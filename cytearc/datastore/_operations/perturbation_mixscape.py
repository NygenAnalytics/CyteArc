from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    MixscapeArguments,
    PerturbationLabelArguments,
    PerturbationSignatureArguments,
)
from ...metadata.artifacts import write_cell_data_artifact
from ...metadata.rows import read_metadata_rows_chunkwise
from ...metadata.selection import cell_selection_indices
from ...perturbation.mixscape import UNLABELLED, MixscapeResult, mixscape_settings
from ...perturbation.signature import PerturbationSignature, signature_settings
from ...storage.artifacts import ArtifactRef, artifact_group, require_complete_artifact
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.numba import numba_threads
from .perturbation_store import (
    _BAND_BYTES,
    _COORDINATE_KINDS,
    _cell_samples,
    _cell_values,
    _count_blocks,
    _plan_cells,
    _positions_in,
    _require_budget,
    _require_cell_column,
    _require_ref,
    _status_labels,
    _stored_samples,
)

if TYPE_CHECKING:
    from ..mapping_datastore import (
        MappingDatastore as _PerturbationMixscapeOperationsBase,
    )
else:
    _PerturbationMixscapeOperationsBase = object


class _PerturbationMixscapeOperationsMixin(_PerturbationMixscapeOperationsBase):
    def _signature_controls(
        self, labels: ArtifactRef, cell_index: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Return which cells of ``cell_index`` the labels call controls.

        Also returns the number of labelled cells that ``cell_index`` lacks.
        """
        status = require_complete_artifact(self.zw, labels)
        labelled = cell_selection_indices(self.zw, status.input_ref("cell_selection"))
        flags = read_array(artifact_group(self.zw, labels), "is_control").astype(bool)
        positions, present = _positions_in(cell_index, labelled)
        is_control = np.zeros(len(cell_index), dtype=bool)
        is_control[positions[present & flags]] = True
        return is_control, int(np.count_nonzero(~present))

    @validation_scoped
    def _perturbation_signature(
        self,
        labels: ArtifactRef,
        coordinates: ArtifactRef,
        *,
        features: ArtifactRef | None = None,
        dims: int = 15,
        n_neighbors: int = 20,
        sample_column: str | None = None,
        scale_factor: float = 10_000.0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Compute each cell's perturbation signature against its nearest controls.

        The first step of Mixscape (Papalexi et al. 2021) removes the
        variation that a cell shares with the control cells around it, such
        as its cell-cycle phase or its replicate. Every cell of the
        coordinates is matched with its ``n_neighbors`` nearest control cells
        of the labels, by Euclidean distance over the first ``dims``
        dimensions of the coordinates, found exactly; a control cell is its
        own nearest control, and of two equally near controls the first in
        store order is taken. The signature of a feature is the cell's
        log-normalized value minus the log of its neighbors' mean normalized
        value, log1p(y_i) - log1p(mean y_j), where y scales each cell's counts
        to ``scale_factor`` total counts, so log1p(y) is Seurat's
        LogNormalize. A knockdown reads negative; Seurat's ``CalcPerturbSig``
        stores the negative, which leaves Mixscape's classes unchanged.

        With ``sample_column``, each cell takes its neighbors from the control
        cells of its own sample, such as its replicate, as Seurat's
        ``split.by`` does. Labelled cells without coordinates have no
        signature, and a log message counts them.

        Args:
            labels: A ``perturbation_labels`` artifact, whose ``"control"``
                cells are the neighbors.
            coordinates: A reduction, batch correction, imported coordinates
                or normalized values, whose cells get a signature.
            features: A ``feature_selection`` artifact of the expression
                assay, the signature's columns. By default, the feature
                selection of the coordinates, such as the highly variable
                genes of a PCA.
            dims: The leading coordinate dimensions that find the neighbors.
            n_neighbors: The control cells averaged for each cell.
            sample_column: A cell metadata column whose samples keep their own
                control cells, or None for one sample.
            scale_factor: The counts each cell is scaled to before log1p.
            invalidate_cache: Compute again instead of reusing a saved
                signature.

        Returns:
            A complete ``perturbation_signature`` artifact scoped to the
            assay of ``features``, with one row per cell of the coordinates.
            Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_signature`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``labels`` is not a ``perturbation_labels``
                artifact, ``coordinates`` is not a coordinate artifact,
                ``features`` is not a feature selection or is missing for
                coordinates without one, a number is out of range,
                ``sample_column`` is not a cell metadata column or leaves a
                cell without a sample, or a sample has fewer than
                ``n_neighbors`` control cells with coordinates.
            PermissionError: If the store is read-only and the signature is
                not saved yet.
            MemoryError: If the control cells' values and each cell's
                neighbors do not fit the memory budget.
        """
        from ...graph.feature_projection import resolve_coordinate_inputs
        from threadpoolctl import threadpool_limits

        from ...perturbation.signature import (
            nearest_controls,
            normalize_counts,
            signature_rows,
        )
        from ...storage.arrays import create_numeric_array, create_zarr_dataset
        from ...storage.artifact_writer import artifact_transaction
        from ...storage.feature_selection import read_feature_selection_indices
        from ...storage.layout import row_sharded_array_spec
        from ...storage.profiles import resolve_storage_profile
        from ...storage.sharding import write_dense_from_row_batches

        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        _require_ref(coordinates, "coordinates", *_COORDINATE_KINDS)
        _require_ref(features, "features", "feature_selection", optional=True)
        resolved_dims, neighbors_k, factor = signature_settings(
            dims, n_neighbors, scale_factor, sample_column
        )
        _require_cell_column(self, sample_column, "sample_column")
        bool_argument(invalidate_cache, "invalidate_cache")
        inputs = resolve_coordinate_inputs(self.zw, coordinates)
        if features is None:
            if inputs.feature_selection is None:
                raise ValueError(
                    f"The {coordinates.kind} coordinates record no feature "
                    "selection; pass features, such as highly variable genes"
                )
            features = inputs.feature_selection
        assay = self._get_assay(str(features.assay))
        feature_index = read_feature_selection_indices(self.zw, assay.name, features)
        data = as_zarr_array(artifact_group(self.zw, coordinates)["data"], name="data")
        n_dims = int(data.shape[1])
        if resolved_dims > n_dims:
            raise ValueError(
                f"dims must be at most the {n_dims} dimensions of the coordinates"
            )
        cell_selection = inputs.cell_selection
        self._require_measured_cells(
            assay.name,
            cell_selection,
            operation=PerturbationSignatureArguments.operation,
        )
        cell_index = cell_selection_indices(self.zw, cell_selection)
        n_cells = len(cell_index)
        is_control, n_missing = self._signature_controls(labels, cell_index)
        cell_samples = _cell_samples(
            self,
            sample_column,
            cell_index,
            reason="Each cell takes its neighbors from the control cells of its "
            "sample, so every cell needs one",
        )
        codes, sample_names = cell_samples.codes, cell_samples.names
        control_counts = np.bincount(codes[is_control], minlength=len(sample_names))
        short = np.flatnonzero(control_counts < neighbors_k)
        if len(short):
            where = (
                "The cells"
                if sample_column is None
                else ", ".join(
                    f"sample {sample_names[code]!r}" for code in short[:5].tolist()
                )
            )
            raise ValueError(
                f"{where} have {int(control_counts[short[0]])} control cells with "
                f"coordinates, fewer than n_neighbors={neighbors_k}; lower "
                "n_neighbors, or label more control cells"
            )
        arguments = PerturbationSignatureArguments(
            labels=labels,
            coordinates=coordinates,
            feature_selection=features,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            sample_assignments_fingerprint=cell_samples.assignments,
            dims=resolved_dims,
            n_neighbors=neighbors_k,
            sample_columns=cell_samples.columns,
            scale_factor=factor,
            invalidate_cache=invalidate_cache,
        )
        n_features = len(feature_index)

        def has_features(_ref: ArtifactRef, stored: zarr.Group) -> bool:
            stored_features = stored.get("features")
            return (
                isinstance(stored_features, zarr.Array)
                and stored_features.shape == (n_features,)
                and stored.attrs.get("sample_names") == sample_names
            )

        planned = _plan_cells(
            self,
            arguments,
            assay=assay.name,
            cell_selection=cell_selection,
            arrays={
                "values": ((n_cells, n_features), "f"),
                "neighbors": ((n_cells, neighbors_k), "i"),
                "is_control": ((n_cells,), "b"),
                "samples": ((n_cells,), "i"),
            },
            invalidate_cache=invalidate_cache,
            reuse_validator=has_features,
        )
        if planned.reused:
            return planned.ref
        if n_missing:
            logger.info(
                f"{n_missing} labelled cells have no coordinates and get no signature"
            )
        controls = np.flatnonzero(is_control)
        n_controls = len(controls)
        # The control cells' scaled values, and each cell's coordinates,
        # neighbors, total and sample.
        resident = n_controls * n_features * 8 + n_cells * (
            8 * resolved_dims + 16 * neighbors_k + 40
        )
        _require_budget(
            resident,
            self.memoryBytes,
            f"The values of {n_controls} control cells in {n_features} features "
            f"and the neighbors of {n_cells} cells hold about",
            remedy="Raise mem_budget, or pass fewer features.",
        )
        self._require_writable(arguments.operation)
        values = np.asarray(data[:, :resolved_dims], dtype=np.float64)
        neighbors = np.empty((n_cells, neighbors_k), dtype=np.int64)
        # The distances' matrix products use as many BLAS threads as the store.
        with threadpool_limits(limits=self.nthreads, user_api="blas"):
            for code in range(len(sample_names)):
                members = np.flatnonzero(codes == code)
                pool = members[is_control[members]]
                nearest = nearest_controls(values[members], values[pool], neighbors_k)
                neighbors[members] = np.searchsorted(controls, pool)[nearest]
        del values
        totals = assay._cell_count_totals(cell_index)
        control_scaled = np.empty((n_controls, n_features))
        scratch = 16 * n_features
        offset = 0
        _, control_blocks = _count_blocks(
            self,
            assay,
            feature_index,
            cell_index[controls],
            resident_bytes=resident,
            scratch_bytes_per_row=scratch,
        )
        for block in control_blocks:
            stop = offset + len(block)
            control_scaled[offset:stop] = normalize_counts(
                block, totals[controls[offset:stop]], factor
            )
            offset = stop
        band_rows = max(1, min(n_cells, (1 << 27) // (4 * n_features)))
        with (
            numba_threads(self.nthreads),
            artifact_transaction(self.zw, planned) as group,
        ):
            group.attrs["sample_names"] = sample_names
            output = create_numeric_array(
                group,
                "values",
                row_sharded_array_spec(
                    (n_cells, n_features),
                    np.float32,
                    profile=resolve_storage_profile(self.zw.store),
                    band_rows=band_rows,
                ),
            )
            block_rows, cell_blocks = _count_blocks(
                self,
                assay,
                feature_index,
                cell_index,
                resident_bytes=resident,
                scratch_bytes_per_row=scratch,
            )

            def signature_blocks() -> Iterator[np.ndarray]:
                start = 0
                for block in cell_blocks:
                    counts = np.asarray(block, dtype=np.float64)
                    stop = start + len(counts)
                    yield signature_rows(
                        counts,
                        totals[start:stop],
                        neighbors[start:stop],
                        control_scaled,
                        factor,
                    )
                    start = stop

            write_dense_from_row_batches(
                output,
                signature_blocks(),
                dtype=np.float32,
                resources=self.resources,
                io=self.storageIo,
                # The writer holds one block of signatures while the next
                # block of counts is read and computed.
                producerReserveBytes=resident + block_rows * scratch,
                requireFinite=True,
                operation=arguments.operation,
            )
            for name, array in (
                ("neighbors", controls[neighbors].astype(np.int32)),
                ("is_control", is_control),
                ("samples", codes.astype(np.int32)),
                ("features", np.asarray(feature_index, dtype=np.int64)),
            ):
                stored = create_zarr_dataset(
                    group,
                    name,
                    (min(max(len(array), 1), 100_000), *array.shape[1:]),
                    array.dtype,
                    array.shape,
                )
                stored[...] = array
        logger.info(
            f"Computed the signatures of {n_cells} cells in {n_features} features "
            f"against the {neighbors_k} nearest of {n_controls} control cells"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_signature(
        self,
        signature: ArtifactRef,
        *,
        features: Sequence[str] | None = None,
    ) -> PerturbationSignature:
        """Read a saved perturbation signature.

        Args:
            signature: A ``perturbation_signature`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.signature`.
            features: The IDs of the features to read, or None for every
                feature of the signature.

        Returns:
            The signature of every cell, with each cell's control flag,
            sample and neighbors.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``signature`` is not a ``perturbation_signature``
                artifact, or ``features`` names a feature it lacks.
            MemoryError: If the values read do not fit the memory budget.
        """
        _require_ref(
            signature, "signature", PerturbationSignatureArguments.artifact_kind
        )
        status = require_complete_artifact(self.zw, signature)
        group = artifact_group(self.zw, signature)
        assay = self._get_assay(str(signature.assay))

        feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)[
            read_array(group, "features")
        ]
        if features is None:
            columns = np.arange(len(feature_ids))
        else:
            if isinstance(features, str) or not all(
                isinstance(name, str) for name in features
            ):
                raise TypeError("features must be a sequence of feature IDs or None")
            columns = pd.Index(feature_ids).get_indexer(list(features))
            unknown = [
                name
                for name, column in zip(features, columns, strict=True)
                if column < 0
            ]
            if unknown:
                raise ValueError(
                    f"The signature has no feature {unknown[0]!r}; it holds the "
                    "features of its feature selection"
                )
        cell_index = cell_selection_indices(self.zw, status.input_ref("cell_selection"))
        n_bytes = len(cell_index) * len(columns) * 4
        _require_budget(
            n_bytes,
            self.memoryBytes,
            f"The signatures of {len(cell_index)} cells in {len(columns)} features "
            "hold",
            remedy="Pass fewer features, or raise mem_budget.",
        )
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index)
        ).astype(str)
        stored = as_zarr_array(group["values"], name="values")
        values = np.asarray(stored.oindex[:, columns])
        sample_names = _stored_samples(group, "sample_names")
        parameters = status.parameters or {}
        sample_columns = list(parameters["sample_columns"])
        index = pd.Index(cell_ids, name="ids")
        return PerturbationSignature(
            # The frame wraps the values as read, the 4 bytes per value that the
            # budget check above counts; pandas would otherwise copy them.
            values=pd.DataFrame(
                values, index=index, columns=feature_ids[columns], copy=False
            ),
            cells=pd.DataFrame(
                {
                    "is_control": read_array(group, "is_control").astype(bool),
                    "sample": np.asarray(sample_names, dtype=object)[
                        read_array(group, "samples")
                    ],
                },
                index=index,
            ),
            neighbors=pd.DataFrame(
                cell_ids[read_array(group, "neighbors")], index=index
            ),
            from_assay=assay.name,
            dims=int(parameters["dims"]),
            n_neighbors=int(parameters["n_neighbors"]),
            scale_factor=float(parameters["scale_factor"]),
            sample_column=sample_columns[0] if sample_columns else None,
        )

    def _mixscape_labels(
        self, labels: ArtifactRef, cell_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return each cell's label, its single target's code, and the targets.

        A cell outside the labels is ``"unlabelled"``; a cell without a single
        target that is not a control has the code -1.
        """
        status = require_complete_artifact(self.zw, labels)
        group = artifact_group(self.zw, labels)
        labelled = cell_selection_indices(self.zw, status.input_ref("cell_selection"))

        values = read_array(group, "values").astype(str)
        single = (read_array(group, "n_targets") == 1) & ~read_array(
            group, "is_control"
        ).astype(bool)
        codes = np.where(single, read_array(group, "target_codes")[:, 0], -1).astype(
            np.int64
        )
        positions, present = _positions_in(cell_index, labelled)
        cell_labels = np.full(len(cell_index), UNLABELLED, dtype=object)
        cell_labels[positions[present]] = values[present]
        cell_codes = np.full(len(cell_index), -1, dtype=np.int64)
        cell_codes[positions[present]] = codes[present]
        return cell_labels, cell_codes, read_array(group, "targets").astype(str)

    @validation_scoped
    def _perturbation_mixscape(
        self,
        signature: ArtifactRef,
        *,
        split_column: str | None = None,
        min_de_genes: int = 5,
        min_cells: int = 5,
        logfc_threshold: float = 0.25,
        min_fraction: float = 0.1,
        p_value_cutoff: float = 0.05,
        scale: bool = True,
        max_iterations: int = 10,
        perturbation_type: str = "KO",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call the cells of each target knocked out or escapers with Mixscape.

        Mixscape (Papalexi et al. 2021) separates, among the cells that carry
        a target alone, those whose target was knocked out from the escapers
        that kept its function, as Seurat's ``RunMixscape`` does. For each
        target, within each split of ``split_column``:

        1. The DE genes compare the target's cells with the control cells on
           log-normalized counts, at the signature's ``scale_factor``, over
           every feature of the signature's assay, as Seurat's
           ``FindMarkers``: a gene is tested when at least ``min_fraction`` of
           the cells of either group express it, rounded to 3 digits, and its
           log2 fold change of the mean normalized values with a pseudocount
           of 1 is at least ``logfc_threshold`` in size, by a two-sided
           Wilcoxon rank-sum test with continuity correction; it is DE when
           its p-value times the number of features is below
           ``p_value_cutoff``. A group of fewer than 3 cells has no DE gene.
        2. With fewer than ``min_de_genes`` DE genes, every cell of the target
           is an escaper, with the status ``"few_de_genes"``.
        3. Otherwise the signatures of the DE genes, from the signature, or
           computed from its neighbors for DE genes outside its features, are
           scaled over the target's and control cells (``scale``). Starting
           with every cell knocked out, each cell is projected onto the
           difference between the mean of the knocked-out cells and the mean
           of the control cells, a mixture of two normal distributions is
           fitted to the projections with the control component fixed at the
           control cells' mean and standard deviation, and a cell is knocked
           out when its likelihood under the fitted component exceeds that
           under the control component. This repeats until the calls stop
           changing or for ``max_iterations``. Fewer than ``min_cells``
           knocked-out cells make every cell an escaper, with the status
           ``"few_cells"``.

        A cell's class is ``"<target> KO"`` (``perturbation_type`` replaces
        ``"KO"``) or ``"<target> NP"``, ``"control"`` for a control cell, and
        its label otherwise, such as ``"Negative"`` or ``"Multiple"``, or
        ``"unlabelled"`` for a cell of the signature outside the labels.

        Args:
            signature: A ``perturbation_signature`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.signature`.
            split_column: A cell metadata column whose splits are classified
                apart, or None for one split.
            min_de_genes: The fewest DE genes that let a target be classified.
            min_cells: The fewest knocked-out cells a target keeps.
            logfc_threshold: The smallest absolute log2 fold change a tested
                gene has.
            min_fraction: The smallest share of either group's cells that
                express a tested gene.
            p_value_cutoff: The adjusted p-value below which a gene is DE.
            scale: Scale each DE gene's signatures before the projections.
            max_iterations: The most iterations of each target.
            perturbation_type: The word that names knocked-out cells.
            invalidate_cache: Classify again instead of reusing saved classes.

        Returns:
            A complete ``mixscape_classes`` artifact with one class per cell
            of the signature. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_mixscape`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``signature`` is not a ``perturbation_signature``
                artifact, a number is out of range, ``perturbation_type`` is
                empty or holds a space, or ``split_column`` is not a cell
                metadata column or leaves a cell without a split.
            PermissionError: If the store is read-only and the classes are not
                saved yet.
            MemoryError: If one feature's values over the classified and
                control cells, or one target's and the control cells'
                signatures in the target's DE genes, do not fit the memory
                budget.
        """
        from ...perturbation.mixscape import (
            MIXSCAPE_STATUSES,
            MixscapeFit,
            band_de_genes,
            classify_target,
            mixscape_classes,
        )
        from ...perturbation.signature import (
            normalize_counts_by_feature,
            signature_of_values,
        )

        _require_ref(
            signature, "signature", PerturbationSignatureArguments.artifact_kind
        )
        de_floor, cell_floor, iterations, fold, fraction, cutoff = mixscape_settings(
            min_de_genes,
            min_cells,
            max_iterations,
            logfc_threshold,
            min_fraction,
            p_value_cutoff,
            scale,
            perturbation_type,
            split_column,
        )
        _require_cell_column(self, split_column, "split_column")
        bool_argument(invalidate_cache, "invalidate_cache")
        source = require_complete_artifact(self.zw, signature)
        stored = artifact_group(self.zw, signature)
        assay = self._get_assay(str(signature.assay))
        scale_factor = float((source.parameters or {})["scale_factor"])
        cell_index = cell_selection_indices(self.zw, source.input_ref("cell_selection"))
        n_cells = len(cell_index)
        cell_labels, target_codes, target_names = self._mixscape_labels(
            source.input_ref("labels"), cell_index
        )
        is_control = read_array(stored, "is_control").astype(bool)
        cell_splits = _cell_samples(
            self,
            split_column,
            cell_index,
            reason="Each split is classified apart, so every cell needs one",
        )
        splits, split_names = cell_splits.codes, cell_splits.names
        arguments = MixscapeArguments(
            signature=signature,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            split_assignments_fingerprint=cell_splits.assignments,
            split_columns=cell_splits.columns,
            min_de_genes=de_floor,
            min_cells=cell_floor,
            logfc_threshold=fold,
            min_fraction=fraction,
            p_value_cutoff=cutoff,
            scale=scale,
            max_iterations=iterations,
            perturbation_type=perturbation_type,
            invalidate_cache=invalidate_cache,
        )
        # One row per target and split with cells of the target, by target.
        n_splits = len(split_names)
        grouped = target_codes >= 0
        cell_keys = target_codes * n_splits + splits
        row_keys = np.unique(cell_keys[grouped])
        row_targets = row_keys // n_splits
        row_splits = row_keys % n_splits
        n_rows = len(row_keys)

        def has_targets(_ref: ArtifactRef, group: zarr.Group) -> bool:
            stored_rows = group.get("status")
            return (
                isinstance(stored_rows, zarr.Array)
                and stored_rows.shape == (n_rows,)
                and group.attrs.get("split_names") == split_names
                # Saved classes are reused only when their status codes mean
                # the statuses of this version.
                and group.attrs.get("status_names") == list(MIXSCAPE_STATUSES)
            )

        planned = _plan_cells(
            self,
            arguments,
            assay=assay.name,
            cell_selection=source.input_ref("cell_selection"),
            arrays={
                "values": ((n_cells,), None),
                "global_class": ((n_cells,), None),
                "posterior": ((n_cells,), "f"),
                "score": ((n_cells,), "f"),
                "splits": ((n_cells,), "i"),
            },
            invalidate_cache=invalidate_cache,
            reuse_validator=has_targets,
        )
        if planned.reused:
            return planned.ref
        n_features = int(assay.feats.N)
        n_groups = n_splits + n_rows
        # Each cell's group: the control cells of its split first, then the
        # cells of each row. The cells of the groups are the members.
        row_of = np.full(max(len(target_names), 1) * n_splits, -1, dtype=np.int64)
        row_of[row_keys] = np.arange(n_rows)
        group_of = np.full(n_cells, -1, dtype=np.int64)
        group_of[is_control] = splits[is_control]
        group_of[grouped] = n_splits + row_of[cell_keys[grouped]]
        members = np.flatnonzero(group_of >= 0)
        n_members = len(members)
        member_groups = group_of[members]
        sizes = np.bincount(member_groups, minlength=n_groups)
        by_group = np.argsort(member_groups, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(sizes)])

        def group_members(group: int) -> np.ndarray:
            """A group's cells, as sorted positions among the members."""
            cells: np.ndarray = by_group[bounds[group] : bounds[group + 1]]
            return cells

        control_members = np.flatnonzero(is_control[members])
        n_controls = len(control_members)
        stored_neighbors = as_zarr_array(stored["neighbors"], name="neighbors")
        n_neighbors = int(stored_neighbors.shape[1])
        # Each cell's class, scores, group and stored neighbors, and each
        # member's total and neighbors among the members.
        resident = (48 + 4 * n_neighbors) * n_cells + 8 * (n_neighbors + 2) * n_members
        # One feature of the DE pass: every member's value, the control cells'
        # sorted values and each group's sums, and as much again for the read.
        per_feature = 2 * (8 * (n_members + n_controls) + 16 * n_groups)
        _require_budget(
            resident + per_feature,
            self.memoryBytes,
            f"The values of one feature over {n_members} cells hold about",
            beside=" with the cells' classes and neighbors",
        )
        width = min(
            n_features,
            (self.memoryBytes - resident) // per_feature,
            max(1, _BAND_BYTES // (8 * max(n_members, 1))),
        )
        self._require_writable(arguments.operation)
        member_totals = assay._cell_count_totals(cell_index[members])
        position_of = np.full(n_cells, -1, dtype=np.int64)
        position_of[members] = np.arange(n_members)
        # Every neighbor is a control cell, so a member.
        member_neighbors = position_of[np.asarray(stored_neighbors[:])[members]]
        signature_values = as_zarr_array(stored["values"], name="values")
        signature_features = read_array(stored, "features").astype(np.int64)
        signature_column = pd.Index(signature_features)
        in_signature = np.zeros(n_features, dtype=bool)
        in_signature[signature_features] = True

        def normalized(
            features: np.ndarray, cells: np.ndarray, held: int
        ) -> np.ndarray:
            """The normalized values of features over members, features x cells."""
            out = np.empty((len(features), len(cells)))
            _, blocks = _count_blocks(
                self,
                assay,
                features,
                cell_index[members[cells]],
                resident_bytes=held,
                scratch_bytes_per_row=8 * len(features),
            )
            offset = 0
            for block in blocks:
                normalize_counts_by_feature(
                    block,
                    member_totals[cells[offset : offset + len(block)]],
                    scale_factor,
                    out,
                    offset,
                )
                offset += len(block)
            return out

        # The DE genes, from one pass over bands of every feature: a row's
        # genes are tested on its cells and the control cells of its split,
        # whose sorted values serve every row of the split.
        testable = [
            row
            for row in range(n_rows)
            if sizes[n_splits + row] >= 3 and sizes[row_splits[row]] >= 3
        ]
        de_parts: list[list[np.ndarray]] = [[] for _ in range(n_rows)]
        every_member = np.arange(n_members)
        with numba_threads(self.nthreads):
            for start in range(0, n_features if testable else 0, width):
                band = np.arange(start, min(start + width, n_features))
                values = normalized(
                    band, every_member, resident + per_feature * len(band)
                )
                found = band_de_genes(
                    values,
                    member_groups,
                    by_group,
                    bounds,
                    testable,
                    row_splits,
                    min_fraction=fraction,
                    logfc_threshold=fold,
                    p_value_cutoff=cutoff,
                    n_features=n_features,
                )
                for row, genes in found.items():
                    de_parts[row].append(band[genes])
        de_lists = [
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            for parts in de_parts
        ]

        def batch_bytes(n_targets: int, n_genes: int, scratch: int) -> int:
            """A batch's signatures, float32, one feature of normalized values
            with its read, and its largest classification."""
            n = n_targets + n_controls
            return 4 * n * n_genes + 16 * n + scratch

        # The rows with enough DE genes are classified in batches, each with one
        # read of the stored signatures and one pass over the DE genes outside
        # them, whose signatures come from each cell's stored neighbors.
        batches: list[list[int]] = []
        batch: list[int] = []
        chosen = np.zeros(n_features, dtype=bool)
        size = (0, 0, 0)
        for row in range(n_rows):
            genes = de_lists[row]
            if len(genes) < de_floor:
                continue
            n_target = int(sizes[n_splits + row])
            scratch = 24 * (n_target + int(sizes[row_splits[row]])) * len(genes)
            grown = (
                size[0] + n_target,
                size[1] + int(np.count_nonzero(~chosen[genes])),
                max(size[2], scratch),
            )
            if batch and resident + batch_bytes(*grown) > self.memoryBytes:
                batches.append(batch)
                batch = []
                chosen[:] = False
                grown = (n_target, len(genes), scratch)
            _require_budget(
                resident + batch_bytes(*grown),
                self.memoryBytes,
                f"The signatures of the {n_target} cells of "
                f"{target_names[row_targets[row]]} and of the {n_controls} control "
                f"cells in its {len(genes)} DE genes hold about",
            )
            batch.append(row)
            chosen[genes] = True
            size = grown
        if batch:
            batches.append(batch)
        logger.debug(
            f"Classifying {sum(len(batch) for batch in batches)} targets and splits "
            f"in {len(batches)} batches"
        )
        fits: list[MixscapeFit | None] = [None] * n_rows
        with numba_threads(self.nthreads):
            for batch in batches:
                genes = np.unique(np.concatenate([de_lists[row] for row in batch]))
                inside = genes[in_signature[genes]]
                outside = genes[~in_signature[genes]]
                cells = np.union1d(
                    np.concatenate([group_members(n_splits + row) for row in batch]),
                    control_members,
                )
                held = resident + batch_bytes(len(cells) - n_controls, len(genes), 0)
                signatures = np.empty((len(cells), len(genes)), dtype=np.float32)
                if len(inside):
                    signatures[:, np.searchsorted(genes, inside)] = (
                        signature_values.oindex[
                            members[cells], signature_column.get_indexer(inside)
                        ]
                    )
                if len(outside):
                    local = np.full(n_members, -1, dtype=np.int64)
                    local[cells] = np.arange(len(cells))
                    nearest = local[member_neighbors[cells]]
                    step = min(
                        1 + max(self.memoryBytes - held, 0) // (16 * len(cells)),
                        max(1, _BAND_BYTES // (8 * len(cells))),
                    )
                    computed = np.searchsorted(genes, outside)
                    for start in range(0, len(outside), step):
                        part = outside[start : start + step]
                        signatures[:, computed[start : start + len(part)]] = (
                            signature_of_values(
                                normalized(part, cells, held),
                                np.arange(len(part)),
                                np.arange(len(cells)),
                                nearest,
                            )
                        )
                for row in batch:
                    targets = group_members(n_splits + row)
                    places = np.searchsorted(
                        cells,
                        np.concatenate([targets, group_members(int(row_splits[row]))]),
                    )
                    values = np.asarray(
                        signatures[
                            np.ix_(places, np.searchsorted(genes, de_lists[row]))
                        ],
                        dtype=np.float64,
                    )
                    fits[row] = classify_target(
                        values[: len(targets)],
                        values[len(targets) :],
                        n_de_genes=len(de_lists[row]),
                        min_de_genes=de_floor,
                        min_cells=cell_floor,
                        max_iterations=iterations,
                        scale=scale,
                    )

        row_cells = [members[group_members(n_splits + row)] for row in range(n_rows)]
        # A row with fewer DE genes than min_de_genes was not classified: its
        # cells are escapers, without values.
        row_fits = [
            classify_target(
                np.empty((len(row_cells[row]), 0)),
                np.empty((int(sizes[row_splits[row]]), 0)),
                n_de_genes=len(de_lists[row]),
                min_de_genes=de_floor,
                min_cells=cell_floor,
                max_iterations=iterations,
                scale=scale,
            )
            if fit is None
            else fit
            for row, fit in enumerate(fits)
        ]
        assigned = mixscape_classes(
            cell_labels,
            is_control,
            row_cells,
            [str(target_names[row_targets[row]]) for row in range(n_rows)],
            row_fits,
            [len(genes) for genes in de_lists],
            perturbation_type,
        )
        global_class = assigned.global_class
        de_offsets = np.concatenate(
            [[0], np.cumsum([len(genes) for genes in de_lists])]
        ).astype(np.int64)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": assigned.classes.astype(str),
                "global_class": global_class.astype(str),
                "posterior": assigned.posterior,
                "score": assigned.score,
                "splits": splits.astype(np.int32),
                "targets": target_names[row_targets].astype(str),
                "row_splits": row_splits.astype(np.int32),
                **{
                    key: getattr(assigned, key)
                    for key in (
                        "status",
                        "n_cells",
                        "n_ko",
                        "n_de_genes",
                        "iterations",
                        "converged",
                        "control_mean",
                        "control_sd",
                        "perturbed_mean",
                        "perturbed_sd",
                        "perturbed_weight",
                    )
                },
                "de_offsets": de_offsets,
                "de_features": (
                    np.concatenate(de_lists).astype(np.int64)
                    if n_rows
                    else np.empty(0, dtype=np.int64)
                ),
            },
            attributes={
                "status_names": list(MIXSCAPE_STATUSES),
                "split_names": split_names,
            },
        )
        logger.info(
            f"Classified {int(np.count_nonzero(grouped))} cells of {n_rows} targets "
            f"and splits: {int(np.count_nonzero(global_class == perturbation_type))} "
            f"{perturbation_type}, {int(np.count_nonzero(global_class == 'NP'))} NP"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_mixscape(self, classes: ArtifactRef) -> MixscapeResult:
        """Read saved Mixscape classes.

        Args:
            classes: A ``mixscape_classes`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.mixscape`.

        Returns:
            Each cell's class, posterior and score, each target's fit in each
            split, and its DE genes.

        Raises:
            TypeError: If ``classes`` is not an ArtifactRef.
            ValueError: If it is not a ``mixscape_classes`` artifact.
        """
        from ...perturbation.mixscape import MIXSCAPE_STATUSES

        _require_ref(classes, "classes", MixscapeArguments.artifact_kind)
        status = require_complete_artifact(self.zw, classes)
        group = artifact_group(self.zw, classes)
        assay = self._get_assay(str(classes.assay))

        cell_index = cell_selection_indices(self.zw, status.input_ref("cell_selection"))
        index = pd.Index(
            np.asarray(
                read_metadata_rows_chunkwise(self.cells, "ids", cell_index)
            ).astype(str),
            name="ids",
        )
        split_names = np.asarray(_stored_samples(group, "split_names"), dtype=object)
        cells = pd.DataFrame(
            {
                "mixscape_class": _cell_values(self, classes).astype(str),
                "global_class": _cell_values(self, classes, "global_class").astype(str),
                "posterior": read_array(group, "posterior"),
                "score": read_array(group, "score"),
                "split": split_names[read_array(group, "splits")],
            },
            index=index,
        )
        row_splits = split_names[read_array(group, "row_splits")]
        targets = read_array(group, "targets").astype(str)
        table: dict[str, Any] = {
            "target": targets,
            "split": row_splits,
            "status": _status_labels(group, MIXSCAPE_STATUSES),
        }
        for name in (
            "n_cells",
            "n_ko",
            "n_de_genes",
            "iterations",
            "converged",
            "control_mean",
            "control_sd",
            "perturbed_mean",
            "perturbed_sd",
            "perturbed_weight",
        ):
            table[name] = read_array(group, name)
        offsets = read_array(group, "de_offsets")
        feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
        rows = np.repeat(np.arange(len(targets)), np.diff(offsets))
        parameters = status.parameters or {}
        split_columns = list(parameters["split_columns"])
        return MixscapeResult(
            cells=cells,
            targets=pd.DataFrame(table),
            de_genes=pd.DataFrame(
                {
                    "target": targets[rows],
                    "split": row_splits[rows],
                    "feature": feature_ids[read_array(group, "de_features")],
                }
            ),
            from_assay=assay.name,
            perturbation_type=str(parameters["perturbation_type"]),
            split_column=split_columns[0] if split_columns else None,
        )
