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
from ...metadata.rows import array_row_selection_parts, read_whole
from ...perturbation.edistance import (
    EDISTANCE_STATUSES,
    EDISTANCE_TESTED,
    PerturbationEDistance,
    edistance_arrays,
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
from ...storage.async_execution import zarr_decoding_threads
from ...storage.geometry import array_geometry
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.memory_budget import MemoryLedger
from ...utils.warnings import warn
from .perturbation_store import (
    _COORDINATE_KINDS,
    _call_ledger,
    _cell_selection_check_bytes,
    _cell_totals,
    _compared_cells,
    _count_blocks,
    _feature_text,
    _frame_bytes,
    _label_rows,
    _matched_ids,
    _pairs_fingerprint,
    _require_ref,
    _row_features,
    _selected_rows,
    _status_labels,
    _table_write_bytes,
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


def _float64_rows(
    ledger: MemoryLedger, data: zarr.Array, rows: np.ndarray
) -> np.ndarray:
    """Read ascending rows of a two-dimensional array as float64, keeping them charged.

    The rows are read one chunk of rows at a time, so that a read holds the
    values of one chunk in their stored type, not those of every row. Each
    read holds its chunk as stored and decoded beside the rows it returns and
    the chunk that the read before can still hold, which the ledger counts,
    and decodes on the ledger's codec threads.
    """
    n_rows, n_columns = len(rows), int(data.shape[1])
    ledger.hold("the cells' coordinates", 8 * n_rows * n_columns)
    values = np.empty((n_rows, n_columns))
    step = int(data.chunks[0])
    geometry = array_geometry(data)
    fixed, per_row = array_row_selection_parts(data)
    held = 0 if geometry is None else geometry.heldReadBytes(ledger.codec_threads)
    # Before the first chunk, the read before is the ledger's last, whose
    # leftover the ledger counts already.
    ledger.peak(
        "reading the cells' coordinates",
        fixed + per_row * min(n_rows, step) + max(0, held - ledger.left),
    )
    # Where the rows of each chunk start among the rows, once per chunk that
    # holds any.
    edges = np.unique(
        np.searchsorted(rows, np.arange(0, int(data.shape[0]) + step, step))
    )
    with zarr_decoding_threads(ledger.codec_threads):
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            values[start:stop] = data.oindex[rows[start:stop], :]
    if geometry is not None:
        chunks = max(1, len(edges) - 1)
        ledger.leave(geometry.heldReadBytes(ledger.holding_threads(chunks)))
    return values


def _unmatched_targets_warning(
    names: np.ndarray, targets: np.ndarray, status: np.ndarray, assay: str, by: str
) -> str | None:
    """Name the efficiency rows whose target matches no feature, or return None.

    A measured table warns with this message, and so does a saved table when
    ``efficiency`` reuses it. By guide, each example names the guide and the
    target that matches nothing.

    Args:
        names: Each row's name: its target, or its guide.
        targets: Each row's target.
        status: Each row's status, as a position in ``EFFICIENCY_STATUSES``.
        assay: The expression assay.
        by: ``"target"`` or ``"guide"``, which the rows are.
    """
    unmatched = np.flatnonzero(np.asarray(status) == EFFICIENCY_NO_FEATURE)
    if not len(unmatched):
        return None
    examples = [
        repr(str(names[row]))
        if by == "target"
        else f"guide {str(names[row])!r} (target {str(targets[row])!r})"
        for row in unmatched[:5]
    ]
    return (
        f"{len(unmatched)} of {len(names)} rows have no feature of assay "
        f"{assay!r} for their target, for example "
        + ", ".join(examples)
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
        median library size of the compared cells that hold a UMI of
        ``from_assay``, the fraction of the control expression that remains,
        its base-2 logarithm, and a two-sided Mann-Whitney test on the scaled
        expression with Benjamini-Hochberg adjustment. The test's p-value is
        the normal approximation with the corrections for ties and for
        continuity, which is unreliable for groups of a few cells. A knockdown
        leaves a fraction below 1 and an activation a fraction above 1. A row
        whose control cells hold no UMI of the gene has no fraction and the
        status ``"no_control_expression"``, but keeps its p-value, which counts
        in the adjustment.

        Each target is matched to a feature of ``from_assay`` by the feature
        ID that ``id_column`` of the guide assay gives its guides, which 10x
        imports fill from Cell Ranger's feature reference, when its guides
        give one ID and the feature IDs are unique; otherwise by the one
        feature whose name equals the target, when no other feature shares
        that name. Only the target genes are read, feature-major when that
        reads less. A target that matches no feature is kept with the status
        ``"no_feature"`` and named in a warning, beside its guide by guide,
        which a reused table repeats.

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
                ``id_column`` is not a feature column of the guide assay, no
                labelled cell carries control guides only, or no compared cell
                holds a UMI of ``from_assay``.
            PermissionError: If the store is read-only and the table is not
                saved yet.
            MemoryError: If the comparison does not fit the memory budget.
        """
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")

        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        if not isinstance(from_assay, str) or from_assay not in self.assay_names:
            raise ValueError(f"from_assay must name an assay, not {from_assay!r}")
        efficiency_settings(by, id_column)
        ledger = _call_ledger(self, f"Measuring the efficiency of the {by}s")
        rows = _label_rows(self, ledger, labels, by)
        status = rows.status
        row_names = rows.names
        expression = self._get_assay(from_assay)
        feature_ids = _feature_text(ledger, expression, "ids")
        feature_names = _feature_text(ledger, expression, "names")
        row_features = _row_features(
            self, ledger, labels, rows, id_column, feature_ids, feature_names
        )
        self._require_measured_cells(
            expression.name,
            status.input_ref("cell_selection"),
            operation=PerturbationEfficiencyArguments.operation,
        )
        ledger.hold(
            "each row's feature ID", len(row_features) * feature_ids.dtype.itemsize
        )
        row_feature_ids = _matched_ids(row_features, feature_ids)
        features_bytes = feature_ids.nbytes + feature_names.nbytes
        del feature_ids, feature_names
        ledger.release(features_bytes)
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
                read_whole(ledger, stored, "names", item="the saved rows' names"),
                read_whole(ledger, stored, "targets", item="the saved rows' targets"),
                read_whole(ledger, stored, "status", item="the saved rows' statuses"),
                expression.name,
                by,
            )
            if message is not None:
                warn(message)
            return planned.ref
        compared = _compared_cells(self, ledger, labels, rows, by)
        n_compared = len(compared.cells)
        needed_features = np.unique(row_features[row_features >= 0])
        n_features = len(needed_features)
        ledger.doing = f"Comparing {n_features} target genes over {n_compared} cells"
        n_control = int(np.count_nonzero(compared.control))
        library = _cell_totals(
            ledger,
            expression,
            compared.cells,
            item="the compared cells' library sizes",
        )
        if not np.any(library > 0):
            raise ValueError(
                f"None of the {n_compared} compared cells, those that carry a "
                "perturbation alone and the control cells, holds a UMI of assay "
                f"{expression.name!r}, so their expression cannot be scaled to "
                "a median library size; pass a from_assay that holds counts for "
                "these cells"
            )
        self._require_writable(arguments.operation)
        counts_bytes = 8 * n_features * n_compared
        if n_features:
            # The counts are streamed beside the blocks gathered so far, and
            # joined into one array beside them.
            ledger.hold("the target genes' counts", counts_bytes)
            gathered = [
                np.asarray(block, dtype=np.float64)
                for block in _count_blocks(
                    self,
                    ledger,
                    expression,
                    needed_features,
                    compared.cells,
                    scratch_bytes_per_row=8 * n_features,
                    item="reading the target genes' counts",
                )
            ]
            ledger.hold("the target genes' counts", counts_bytes)
            counts = np.concatenate(gathered)
            del gathered
            ledger.release(counts_bytes)
        else:
            counts = np.zeros((n_compared, 0))
        row_columns = np.where(
            row_features >= 0, np.searchsorted(needed_features, row_features), -1
        )
        arrays = efficiency_arrays(
            counts,
            library,
            compared.groups,
            compared.control,
            row_columns,
            ledger=ledger,
        )
        columns = {
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
        }
        del counts, library
        ledger.release(counts_bytes + 8 * n_compared)
        ledger.hold(
            "the efficiency table",
            sum(np.asarray(values).nbytes for values in columns.values()),
        )
        ledger.peak("saving the efficiency table", _table_write_bytes(columns))
        write_cell_data_artifact(
            self.zw,
            planned,
            columns,
            attributes={"status_names": list(EFFICIENCY_STATUSES), "by": by},
        )
        message = _unmatched_targets_warning(
            row_names,
            rows.target_names[rows.codes],
            arrays.status,
            expression.name,
            by,
        )
        if message is not None:
            warn(message)
        logger.info(
            "Measured the efficiency of "
            f"{int(np.count_nonzero(arrays.status == EFFICIENCY_TESTED))} "
            f"of {n_rows} {by}s against {n_control} control cells"
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
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as a call that
                stopped before it finished leaves it.
        """
        _require_ref(
            efficiency, "efficiency", PerturbationEfficiencyArguments.artifact_kind
        )
        status = require_complete_artifact(self.zw, efficiency)
        group = artifact_group(self.zw, efficiency)
        ledger = _call_ledger(self, f"Reading the {efficiency.kind} artifact")

        by = str((status.parameters or {})["by"])
        columns: dict[str, Any] = {}
        if by == "guide":
            columns["guide"] = read_whole(ledger, group, "names", item="the guides")
        columns["target"] = read_whole(ledger, group, "targets", item="the targets")
        features = read_whole(ledger, group, "features", item="the features")
        columns["status"] = _status_labels(
            group,
            EFFICIENCY_STATUSES,
            read_whole(ledger, group, "status", item="the statuses"),
        )
        for name in ("n_cells", "n_control_cells", *_EFFICIENCY_VALUES):
            columns[name] = read_whole(ledger, group, name, item=f"the rows' {name}")
        ledger.peak(
            "the efficiency table", _frame_bytes({**columns, "feature": features})
        )
        columns["feature"] = pd.Series(
            [None if feature == "" else str(feature) for feature in features],
            dtype=object,
        )
        columns = {
            name: columns[name]
            for name in (
                *(("guide",) if by == "guide" else ()),
                "target",
                "feature",
                "status",
                "n_cells",
                "n_control_cells",
                *_EFFICIENCY_VALUES,
            )
        }
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
            MemoryError: If the measurement does not fit the memory budget.
        """
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        from ...graph.feature_projection import resolve_coordinate_inputs

        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        _require_ref(coordinates, "coordinates", *_COORDINATE_KINDS)
        permutations, seed = edistance_settings(by, n_permutations, random_seed)
        ledger = _call_ledger(self, f"Measuring the E-distances of the {by}s")
        rows = _label_rows(self, ledger, labels, by)
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
        compared = _compared_cells(self, ledger, labels, rows, by)
        # Imported coordinates are checked against the order of their cells'
        # IDs as their selection is first read.
        ledger.peak(
            "checking the coordinates' cells against the cell IDs",
            _cell_selection_check_bytes(
                self, selected_ids=coordinates.kind == "imported_coordinates"
            ),
        )
        positions, present = _selected_rows(
            self, ledger, inputs.cell_selection, compared.cells
        )
        n_compared = len(compared.cells)
        if not present.all():
            logger.info(
                f"{int(np.count_nonzero(~present))} of {n_compared} compared cells "
                "have no coordinates and are left out"
            )
        # The measurement needs only the kept cells' coordinate rows, groups
        # and flags, so the arrays of every compared cell are released here.
        n_kept = int(np.count_nonzero(present))
        ledger.hold("the kept cells' rows, groups and flags", 17 * n_kept)
        coordinate_rows = positions[present]
        kept_groups = compared.groups[present]
        kept_control = compared.control[present]
        del compared, positions, present
        ledger.release(26 * n_compared)
        n_control = int(np.count_nonzero(kept_control))
        if n_control < 2:
            raise ValueError(
                f"E-distances need at least two control cells with coordinates, "
                f"and {n_control} of the labels' control cells have them"
            )
        data = as_zarr_array(artifact_group(self.zw, coordinates)["data"], name="data")
        n_dims = int(data.shape[1])
        ledger.doing = (
            f"Measuring the E-distances of {n_kept} cells in {n_dims} dimensions"
        )
        self._require_writable(arguments.operation)
        values = _float64_rows(ledger, data, coordinate_rows)
        rows_bytes = coordinate_rows.nbytes
        del coordinate_rows
        ledger.release(rows_bytes)
        arrays = edistance_arrays(
            values,
            kept_groups,
            kept_control,
            n_rows,
            n_permutations=permutations,
            random_seed=seed,
            nthreads=self.nthreads,
            ledger=ledger,
        )
        del values
        ledger.release(8 * n_dims * n_kept)
        columns = {
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
        }
        ledger.hold(
            "the E-distance table",
            sum(np.asarray(values).nbytes for values in columns.values()),
        )
        ledger.peak("saving the E-distance table", _table_write_bytes(columns))
        write_cell_data_artifact(
            self.zw,
            planned,
            columns,
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
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as a call that
                stopped before it finished leaves it.
        """
        _require_ref(
            edistance, "edistance", PerturbationEDistanceArguments.artifact_kind
        )
        status = require_complete_artifact(self.zw, edistance)
        group = artifact_group(self.zw, edistance)
        ledger = _call_ledger(self, f"Reading the {edistance.kind} artifact")

        parameters = status.parameters or {}
        by = str(parameters["by"])
        columns: dict[str, Any] = {}
        if by == "guide":
            columns["guide"] = read_whole(ledger, group, "names", item="the guides")
        columns["target"] = read_whole(ledger, group, "targets", item="the targets")
        columns["status"] = _status_labels(
            group,
            EDISTANCE_STATUSES,
            read_whole(ledger, group, "status", item="the statuses"),
        )
        for name in (
            "n_cells",
            "n_control_cells",
            "edistance",
            "p_value",
            "adjusted_p_value",
        ):
            columns[name] = read_whole(ledger, group, name, item=f"the rows' {name}")
        ledger.peak("the E-distance table", _frame_bytes(columns))
        return PerturbationEDistance(
            table=pd.DataFrame(columns),
            by=by,
            from_assay=str(edistance.assay),
            n_permutations=int(parameters["n_permutations"]),
            random_seed=int(parameters["random_seed"]),
        )
