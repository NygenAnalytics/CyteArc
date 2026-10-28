from collections.abc import Iterable, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ...assay import ATACassay, RNAassay
from ...assay.feature_summary import (
    ensure_feature_summary,
    feature_summary_selected_count,
    feature_summary_values,
)
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_graph_assay_inputs,
)
from ...graph.kinds import require_graph_kind
from ...quality_control.cell_cycle import assign_cell_cycle_phase
from ...quality_control.cell_cycle_genes import match_gene_names
from ...quality_control.filtering import (
    MetricKind,
    MetricPolicy,
    filter_cell_metrics,
    resolve_metric_policies,
    validate_bound_types,
    validate_cell_filter_sources,
    validate_filter_bounds,
    validate_metric_policies,
)
from ...quality_control.hto import (
    _hto_clr_diagnostics,
    _hto_demux_method,
    _validated_positive_quantile,
    hto_demux,
)
from ...metadata.artifacts import (
    artifact_values,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.arguments import (
    CellCycleArguments,
    DoubletScoreArguments,
    HtoIdentityArguments,
    PrevalentPeakArguments,
)
from ...metadata.membership import measured_selection
from ...metadata.rows import (
    apply_missing_mask,
    metadata_missing_mask,
    read_metadata_missing_rows,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import (
    NamedCellArtifact,
    cell_value_array,
    cell_value_spec,
    grouping_value_name,
    require_complete_cluster_labels,
    resolve_cell_aligned_artifact,
    sample_codes,
)
from ...storage.validation_scope import validation_scoped
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    plan_artifact,
)
from ...storage.artifacts import (
    artifact_group,
    artifact_path,
    fingerprint_array,
    fingerprint_strings,
)
from ...storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
    read_feature_selection_indices,
    resolve_feature_selection,
)
from ...storage.refs import ArtifactRef
from ...storage.selections import (
    ValidatedStoredSelection,
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
    selection_mask,
    validate_stored_selection_integrity,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...utils.arguments import integer_argument
from ...utils.arrays import within_bounds
from ...utils.compute import controlled_compute
from ...utils.logging import logger

if TYPE_CHECKING:
    from ...quality_control.doublets import DoubletSampleResult
    from ..mapping_datastore import MappingDatastore as _QualityControlOperationsBase
else:
    _QualityControlOperationsBase = object


_ONE_CLUSTER_DOUBLET_REMEDY = (
    "Pass a clustering with two or more clusters, or heterotypic_fraction=0 to "
    "simulate doublets from any two sampled cells."
)


def _validated_real(
    value: object,
    name: str,
    *,
    low: float,
    high: float = np.inf,
    include_low: bool = True,
) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    in_range = (resolved >= low if include_low else resolved > low) and (
        resolved <= high
    )
    if not np.isfinite(resolved) or not in_range:
        bound = f">= {low:g}" if include_low else f"> {low:g}"
        if np.isfinite(high):
            bound += f" and <= {high:g}"
        raise ValueError(f"{name} must be a finite number {bound}")
    return resolved


def _doublet_sample_columns(sample_column: object) -> tuple[str, ...]:
    """Return the cell metadata columns that split doublet detection into samples."""
    if sample_column is None:
        return ()
    columns = (
        (sample_column,)
        if isinstance(sample_column, str)
        else tuple(sample_column)
        if isinstance(sample_column, Sequence)
        else ()
    )
    if not columns or not all(isinstance(column, str) and column for column in columns):
        raise TypeError(
            "sample_column must be a non-empty string, a sequence of them, or None"
        )
    if len(set(columns)) != len(columns):
        raise ValueError("sample_column must not repeat a column")
    return columns


def _doublet_sample_text(labels: dict[str, str]) -> str:
    return (
        "(" + ", ".join(f"{column}={value!r}" for column, value in labels.items()) + ")"
    )


def _doublet_sample_summary(
    labels: dict[str, str],
    outcome: "DoubletSampleResult",
) -> dict[str, Any]:
    """Describe one sample's doublet simulation and calls as JSON values."""
    plan = outcome.plan
    return {
        "labels": labels,
        "cells": plan.cells,
        "simulatedDoublets": plan.simulated,
        "neighbors": plan.neighbors,
        "adjustedNeighbors": plan.adjusted_neighbors,
        "heterotypicFraction": outcome.heterotypic_fraction,
        "threshold": outcome.threshold,
        "thresholdMethod": outcome.threshold_method,
        "predictedDoublets": outcome.predicted,
        "detectableDoubletFraction": outcome.detectable_fraction,
        "estimatedDoubletRate": outcome.estimated_doublet_rate,
    }


class _QualityControlOperationsMixin(_QualityControlOperationsBase):
    def _run_cell_cycle_scoring_artifact(
        self,
        *,
        assay: RNAassay,
        cell_selection: ArtifactRef,
        feature_names: np.ndarray | None = None,
        feature_snapshot: ArtifactRef | None = None,
        s_genes: list[str] | None = None,
        g2m_genes: list[str] | None = None,
        ctrl_size: int | None = None,
        log_transform: bool | None = None,
        n_bins: int = 50,
        rand_seed: int = 4466,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Create or reuse cell-cycle scores without creating metadata columns.

        The arguments are checked first, then the cells against the assay's
        membership. A read-only store then reuses saved scores and refuses
        only before it computes new ones.
        """
        from ...assay.normalization import resolve_normalization_params

        if s_genes is None:
            from ...quality_control.cell_cycle_genes import s_phase_genes

            s_genes = list(s_phase_genes)
        if g2m_genes is None:
            from ...quality_control.cell_cycle_genes import g2m_phase_genes

            g2m_genes = list(g2m_phase_genes)
        for label, genes in (("s_genes", s_genes), ("g2m_genes", g2m_genes)):
            if isinstance(genes, str) or not all(
                isinstance(gene, str) for gene in genes
            ):
                raise TypeError(f"{label} must be a sequence of gene names")
        control_size = integer_argument(
            min(len(s_genes), len(g2m_genes)) if ctrl_size is None else ctrl_size,
            "ctrl_size",
            minimum=1,
        )
        if log_transform is not None and not isinstance(log_transform, bool):
            raise TypeError("log_transform must be a bool")
        # Scores are logged by default when the normalizer's defaults log.
        log_transform = resolve_normalization_params(
            assay,
            {"log_transform": log_transform},
            caller="run_cell_cycle_scoring",
            default=True,
        )["log_transform"]
        n_bins = integer_argument(n_bins, "n_bins", minimum=2)
        rand_seed = integer_argument(rand_seed, "rand_seed", minimum=0)
        names = np.asarray(
            assay.feats.fetch_all("names") if feature_names is None else feature_names
        )
        # Names match without case sensitivity; a name shared by several
        # features selects all of them, and a renamed gene that its listed
        # symbol misses matches by another HGNC symbol.
        gene_aliases: dict[str, str] = {}

        def indices_for(targets: list[str], label: str) -> list[int]:
            match = match_gene_names(names, targets)
            if match.unmatched:
                logger.warning(
                    f"{len(match.unmatched)} of {len(targets)} {label} were not "
                    "found in the assay feature names"
                )
            if not match.indices:
                raise ValueError(f"None of the {label} match the assay feature names")
            if match.aliases:
                logger.info(
                    f"{len(match.aliases)} of {len(targets)} {label} matched another "
                    "HGNC symbol of the same gene: "
                    + ", ".join(
                        f"{gene!r} as {name!r}" for gene, name in match.aliases.items()
                    )
                )
                gene_aliases.update(match.aliases)
            return match.indices

        s_gene_indices = indices_for(s_genes, "s_genes")
        g2m_gene_indices = indices_for(g2m_genes, "g2m_genes")
        self._require_measured_cells(
            assay.name, cell_selection, operation="run_cell_cycle_scoring"
        )
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            log_transform=log_transform,
            invalidate_cache=invalidate_cache,
        )
        n_cells = feature_summary_selected_count(self.zw, cell_selection)
        arguments = CellCycleArguments(
            feature_summary=summary_ref,
            cell_selection=cell_selection,
            s_gene_indices=tuple(s_gene_indices),
            g2m_gene_indices=tuple(g2m_gene_indices),
            control_size=control_size,
            log_transform=log_transform,
            n_bins=n_bins,
            rand_seed=rand_seed,
            invalidate_cache=invalidate_cache,
            gene_aliases=gene_aliases or None,
        )
        record = arguments.to_record()
        inputs = dict(record.inputs)
        if feature_snapshot is not None:
            inputs["feature_snapshot"] = feature_snapshot
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={
                "s_score": ((n_cells,), "f"),
                "g2m_score": ((n_cells,), "f"),
                "phase": ((n_cells,), None),
            },
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_cell_cycle_scoring")

        cell_idx = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        summary = feature_summary_values(
            self.zw,
            summary_ref,
            n_selected=n_cells,
        )
        # One pass reads the S and G2M genes and their controls.
        s_score, g2m_score = assay._score_feature_sets(
            [
                np.asarray(s_gene_indices, dtype=np.int64),
                np.asarray(g2m_gene_indices, dtype=np.int64),
            ],
            cell_idx,
            summary["avg"],
            ctrl_size=control_size,
            n_bins=n_bins,
            rand_seed=rand_seed,
            log_transform=log_transform,
        )
        phase = np.asarray(assign_cell_cycle_phase(s_score, g2m_score))
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "s_score": np.asarray(s_score),
                "g2m_score": np.asarray(g2m_score),
                "phase": phase,
            },
        )
        return planned.ref

    @validation_scoped
    def _qc_filter(
        self,
        attrs: Iterable[str],
        lows: Iterable[float | None],
        highs: Iterable[float | None],
        *,
        cell_selection: ArtifactRef | None = None,
        keep_bounds: bool = False,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Create an immutable selection from metadata thresholds.

        The requested columns are read from current cell metadata when this
        method is called. Their values are fingerprinted in provenance, and the
        resulting immutable selection is stored. The live ``I`` column is
        snapshotted when ``cell_selection`` is omitted. Pass a prior selection
        explicitly to compose multiple filtering steps.

        Cells whose value is recorded as missing in a nullable column never
        pass its bounds. The fingerprint of such a column's missing-value mask
        is also recorded in provenance.

        Bounds are exclusive unless ``keep_bounds`` is True: ``lows=[200]``
        keeps cells whose value is greater than 200 and removes a cell with
        exactly 200. Scanpy's ``filter_cells(min_genes=200)`` and CyteArc's
        ``features.detected`` keep values equal to their thresholds; pass
        ``keep_bounds=True`` for the same inclusive bounds.

        Args:
            attrs: Names of distinct columns to be used for filtering
            lows: Lower bounds, in the same order as ``attrs``. Each bound is a
                finite number for a numeric or boolean column, text for a text
                column, or None.
            highs: Upper bounds, in the same order as ``attrs``. A lower bound
                cannot exceed its upper bound.
            cell_selection: Optional prior cell-selection artifact.
            keep_bounds: Retain values exactly equal to a bound, which makes
                both bounds inclusive.

        Returns:
            A complete datastore-scoped ``cell_selection`` artifact.

        Raises:
            TypeError: If a bound or ``keep_bounds`` has an invalid type, or a
                bound does not match its column, such as text for a numeric
                column. Bound types are checked before any column is read.
            ValueError: If a bound is invalid, the input selection is empty,
                or no cell passes the filter.
            UnmeasuredCellsError: If a column's assay did not measure a selected cell.
        """
        attrs = list(attrs)
        lows = list(lows)
        highs = list(highs)
        if not (len(attrs) == len(lows) == len(highs)):
            raise ValueError("attrs, lows, and highs must have the same length")
        attrs, _, _ = validate_cell_filter_sources(attrs)
        validate_filter_bounds(lows, highs, keep_bounds=keep_bounds)
        available = set(self.cells.columns)
        missing = [attr for attr in attrs if attr not in available]
        if missing:
            joined = ", ".join(repr(attr) for attr in missing)
            raise KeyError(f"Cell metadata columns not found: {joined}")
        for attr, low, high in zip(attrs, lows, highs, strict=True):
            validate_bound_types(attr, self.cells.get_dtype(attr), low, high)
        self._require_measured_metric_cells(
            attrs, [], cell_selection, operation="filter_cells"
        )
        prior, active_idx = self._filter_input_cells(cell_selection)
        # Provenance fingerprints cover whole columns, so each is read in full.
        input_fingerprints: dict[str, str] = {}
        missing_fingerprints: dict[str, str] = {}
        values_by_attr: dict[str, np.ndarray] = {}
        missing_by_attr: dict[str, np.ndarray] = {}
        for attr, values in zip(
            attrs, self.cells.fetch_all_columns(attrs), strict=True
        ):
            input_fingerprints[attr] = (
                fingerprint_strings(values)
                if values.dtype.kind in {"O", "S", "U"}
                else fingerprint_array(values)
            )
            values_by_attr[attr] = values[active_idx]
            mask_array = metadata_missing_mask(self.cells, attr)
            if mask_array is not None:
                mask = np.asarray(mask_array[:], dtype=bool)
                missing_fingerprints[attr] = fingerprint_array(mask)
                missing_by_attr[attr] = mask[active_idx]
        result = filter_cell_metrics(
            values_by_attr,
            missing_by_attr,
            np.ones(len(active_idx), dtype=bool),
            method="manual",
            lows=lows,
            highs=highs,
            keep_bounds=keep_bounds,
        )
        new_bool = np.zeros(self.cells.N, dtype=bool)
        new_bool[active_idx] = result.retained
        inputs: dict[str, Any] = {
            "prior_cell_selection": prior.ref,
            "metadata_fingerprints": input_fingerprints,
        }
        if missing_fingerprints:
            inputs["missing_mask_fingerprints"] = missing_fingerprints
        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=new_bool,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="filter_cells",
            parameters={
                "attrs": attrs,
                "lows": lows,
                "highs": highs,
                "keep_bounds": keep_bounds,
            },
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        remaining = int(stored.sum())
        logger.info(f"Cell filtering retained {remaining}/{self.cells.N} cells")
        return ref

    def select_cells(
        self,
        values: ArtifactRef,
        *,
        low: float | None = None,
        high: float | None = None,
        include: Sequence[Any] | None = None,
        cell_selection: ArtifactRef | None = None,
        keep_bounds: bool = False,
        invalidate_cache: bool = False,
        value: str | None = None,
    ) -> ArtifactRef:
        """Select cells from one numeric or categorical artifact vector.

        Numeric values use optional bounds. Categorical values use ``include``.
        The artifact is read from its kind's canonical array, for example
        ``phase`` of a ``cell_cycle`` artifact or ``labels`` of a ``cluster_cut``,
        or from the other per-cell array that ``value`` names, such as
        ``predicted_doublets`` of a ``doublet_score`` artifact.
        The artifact must identify its source cell selection in provenance. By
        default the new selection is composed with that source selection. An
        explicit ``cell_selection`` may narrow it further, but cannot add cells
        that were absent from the source artifact. Cells whose value the
        artifact records as missing are never selected.

        Args:
            values: Complete artifact with one scalar value per selected cell.
            low: Optional lower bound.
            high: Optional upper bound.
            include: Categorical values to retain instead of applying bounds.
            cell_selection: Optional prior selection to intersect.
            keep_bounds: Retain values exactly equal to either bound.
            invalidate_cache: Create a fresh result instead of reusing a match.
            value: The per-cell array of ``values`` to read, or None for its
                canonical array. A selection records any other array.

        Returns:
            A complete datastore-scoped ``cell_selection`` artifact.

        Raises:
            ValueError: If no cell is selected.
        """
        if not isinstance(values, ArtifactRef):
            raise TypeError("values must be an ArtifactRef")
        if not isinstance(keep_bounds, bool):
            raise TypeError("keep_bounds must be a boolean")
        # The canonical array is read and recorded as before; any other array
        # is named in the selection's parameters.
        value_name: str | None = None
        if value is not None:
            value_name, _categorical = cell_value_array(values.kind, value)
            if value_name == cell_value_spec(values.kind).name:
                value_name = None

        raw_include: tuple[str | int | float | bool, ...] | None = None
        if include is not None:
            if isinstance(include, str | bytes) or not isinstance(include, Sequence):
                raise TypeError("include must be a sequence of scalar values")
            included: list[str | int | float | bool] = []
            for raw_value in include:
                value = (
                    raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                )
                if not isinstance(value, str | int | float | bool):
                    raise TypeError("include must contain only scalar values")
                if isinstance(value, float) and not np.isfinite(value):
                    raise ValueError("include must contain only finite values")
                included.append(value)
            if not included:
                raise ValueError("include must contain at least one value")
            raw_include = tuple(included)
            if low is not None or high is not None or keep_bounds:
                raise ValueError(
                    "include cannot be combined with low, high, or keep_bounds"
                )

        def resolve_bound(value: float | None, name: str) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite number or None")
            resolved = float(value)
            if not np.isfinite(resolved):
                raise ValueError(f"{name} must be finite or None")
            return resolved

        resolved_low = resolve_bound(low, "low")
        resolved_high = resolve_bound(high, "high")
        if (
            resolved_low is not None
            and resolved_high is not None
            and resolved_low > resolved_high
        ):
            raise ValueError("low cannot exceed high")

        if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        # The one reader of cell-aligned artifacts checks the kind, the cell
        # selection that the artifact records, and that cell_selection lies
        # within it, and reads the requested array for exactly those cells,
        # once what it holds fits the memory budget, as load_cell_values does.
        aligned = resolve_cell_aligned_artifact(
            self.zw,
            values,
            cell_selection=cell_selection,
            value_name=value_name,
            max_bytes=self.memoryBytes,
        )
        source_selection = aligned.source_cell_selection
        prior_selection = aligned.cell_selection
        raw_values = np.asarray(aligned.values)
        value_kind = raw_values.dtype.kind
        if raw_include is None and value_kind not in {"i", "u", "f"}:
            raise TypeError(
                "values artifact must be numeric unless include is provided"
            )
        if raw_include is not None and value_kind not in {
            "b",
            "f",
            "i",
            "O",
            "u",
            "U",
        }:
            raise TypeError("values artifact must contain scalar values")

        resolved_include: tuple[str | int | float | bool, ...] | None = None
        if raw_include is not None:
            if value_kind in {"O", "U"}:
                if not all(isinstance(value, str) for value in raw_include):
                    raise TypeError(
                        "include values must be strings for a string artifact"
                    )
                resolved_include = tuple(sorted(set(raw_include)))
            elif value_kind == "b":
                if not all(isinstance(value, bool) for value in raw_include):
                    raise TypeError(
                        "include values must be booleans for a boolean artifact"
                    )
                resolved_include = tuple(sorted(set(raw_include)))
            elif value_kind in {"i", "u"}:
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in raw_include
                ):
                    raise TypeError(
                        "include values must be integers for an integer artifact"
                    )
                limits = np.iinfo(raw_values.dtype)
                if any(
                    int(value) < limits.min or int(value) > limits.max
                    for value in raw_include
                ):
                    raise ValueError("include contains an out-of-range integer")
                resolved_include = tuple(sorted({int(value) for value in raw_include}))
            else:
                if not all(
                    isinstance(value, Real) and not isinstance(value, bool)
                    for value in raw_include
                ):
                    raise TypeError(
                        "include values must be numeric for a floating artifact"
                    )
                # Float values are already finite, so only an integer beyond
                # the float64 range fails to convert.
                try:
                    normalized = {float(value) for value in raw_include}
                except OverflowError as exc:
                    raise ValueError(
                        "include contains an out-of-range integer"
                    ) from exc
                resolved_include = tuple(sorted(normalized))

        if resolved_include is not None:
            keep = np.isin(raw_values, resolved_include)
        else:
            numeric = np.asarray(raw_values, dtype=np.float64)
            keep = np.isfinite(numeric) & within_bounds(
                numeric, resolved_low, resolved_high, keep_bounds=keep_bounds
            )
        if aligned.missing_mask is not None:
            keep &= ~aligned.missing_mask
        selected = np.zeros(self.cells.N, dtype=bool)
        selected[aligned.cell_idx[keep]] = True
        if not selected.any():
            raise ValueError("select_cells retained no cells")
        prior = validate_stored_selection_integrity(
            self.zw,
            prior_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

        inputs: dict[str, Any] = {
            "values": values,
            "source_cell_selection": source_selection,
            "prior_cell_selection": prior_selection,
        }
        parameters: dict[str, Any] = {
            "low": resolved_low,
            "high": resolved_high,
            "include": resolved_include,
            "keep_bounds": keep_bounds,
        }
        if value_name is not None:
            parameters["value"] = value_name
        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=selected,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="select_cells",
            parameters=parameters,
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        logger.info(f"Cell selection retained {int(stored.sum())}/{self.cells.N} cells")
        return ref

    def select_measured_cells(
        self,
        assay: str,
        *,
        cell_selection: ArtifactRef | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Keep the cells of a selection that an assay measured.

        Args:
            assay: Name of the assay whose measured cells are kept.
            cell_selection: Optional prior cell-selection artifact.
            invalidate_cache: Create a fresh result instead of reusing a match.

        Returns:
            A complete datastore-scoped ``cell_selection`` artifact.

        Raises:
            ValueError: If ``assay`` is unknown or measured no selected cell.
        """
        if not isinstance(assay, str) or not assay:
            raise TypeError("assay must be the name of an assay")
        assay_name = self._get_assay(assay).name
        prior, active_idx = self._filter_input_cells(cell_selection)
        selected = np.zeros(self.cells.N, dtype=bool)
        selected[active_idx] = True
        measured = measured_selection(self.cells, assay_name, selected)
        if measured is None:
            return prior.ref
        column, values, membership_fingerprint = measured
        remaining = int(np.count_nonzero(values))
        if remaining == len(active_idx):
            return prior.ref
        if remaining == 0:
            raise ValueError(
                f"Assay {assay_name!r} measured none of the {len(active_idx)} "
                f"selected cells: cell column {column!r} is False for all of them."
            )
        ref, _stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=values,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="select_measured_cells",
            parameters={"assay": assay_name},
            inputs={
                "prior_cell_selection": prior.ref,
                "membership_fingerprint": membership_fingerprint,
            },
            source_column=column,
            invalidate_cache=invalidate_cache,
        )
        logger.info(
            f"Assay {assay_name} measured {remaining}/{len(active_idx)} selected cells"
        )
        return ref

    def _read_filter_metrics(
        self,
        attrs: list[str],
        artifact_metrics: list[NamedCellArtifact],
        prior: ValidatedStoredSelection,
        active_idx: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
        """Read QC metrics and missing-value masks for the active cells.

        Returns float values and masks keyed by metric name, and the
        fingerprints of the metadata-column masks that exist. An artifact
        metric's reference already identifies its mask.
        """
        values_by_name: dict[str, np.ndarray] = {}
        missing_by_name: dict[str, np.ndarray] = {}
        # Whole columns are read concurrently: chunk-by-chunk reads of several
        # columns are a chain of sequential requests on object stores.
        for attr, column in zip(
            attrs, self.cells.fetch_all_columns(attrs), strict=True
        ):
            values_by_name[attr] = np.asarray(column, dtype=float)[active_idx]
            missing = read_metadata_missing_rows(self.cells, attr, active_idx)
            if missing is not None:
                missing_by_name[attr] = missing
        for source in artifact_metrics:
            resolved = resolve_cell_aligned_artifact(
                self.zw,
                source.artifact,
                cell_selection=prior.ref,
                expected_kind="quality_metric",
            )
            values_by_name[source.name] = np.asarray(resolved.values, dtype=float)
            if resolved.missing_mask is not None:
                missing_by_name[source.name] = resolved.missing_mask
        fingerprints = {
            name: fingerprint_array(missing)
            for name, missing in missing_by_name.items()
            if name in attrs
        }
        return values_by_name, missing_by_name, fingerprints

    def _filter_input_selection(
        self,
        selection: ArtifactRef | None,
    ) -> ValidatedStoredSelection:
        """Resolve the input cell selection, validated exactly once."""
        if selection is None:
            return self._snapshot_cell_selection("I")
        if not isinstance(selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        return validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

    def _filter_input_cells(
        self,
        selection: ArtifactRef | None,
    ) -> tuple[ValidatedStoredSelection, np.ndarray]:
        """Resolve the input cell selection and its active row indices."""
        prior = self._filter_input_selection(selection)
        active_idx = np.flatnonzero(selection_mask(prior)).astype(np.int64, copy=False)
        if len(active_idx) == 0:
            raise ValueError("Cell selection contains no active cells")
        return prior, active_idx

    def _metric_assays(
        self,
        attrs: Sequence[str],
        artifact_metrics: Sequence[NamedCellArtifact],
    ) -> list[str]:
        """Return the assays whose values the QC metrics describe.

        A cell metadata column is a metric of the assay whose preparation
        wrote it (:func:`~cytearc.storage.identity.generated_cell_columns`):
        ``<assay>_nCounts``, ``<assay>_nFeatures``, and the percentage columns
        that the assay records in its ``percentFeatures`` attribute. The
        column is found among each assay's own columns, never by parsing its
        name, so a user column named like a metric is no metric. A
        ``quality_metric`` artifact scoped to an assay is a metric of it.
        """
        from ...storage.identity import generated_cell_columns

        owners = {
            column: name
            for name in self.assay_names
            for column in generated_cell_columns(
                name, self._get_assay(name)._percent_features()
            )
        }
        assays = [owners[attr] for attr in attrs if attr in owners]
        assays.extend(
            source.artifact.assay
            for source in artifact_metrics
            if source.artifact.kind == "quality_metric"
            and source.artifact.scope == "assay"
            and source.artifact.assay is not None
        )
        return list(dict.fromkeys(assays))

    def _metric_kinds(
        self,
        attrs: Sequence[str],
        artifact_metrics: Sequence[NamedCellArtifact],
    ) -> dict[str, MetricKind]:
        """Return the kind of each QC metric whose kind CyteArc wrote.

        Preparation writes ``<assay>_nCounts`` (counts), ``<assay>_nFeatures``
        (feature counts), and the percentage columns that the assay records in
        its ``percentFeatures`` attribute, as ``_metric_assays`` finds them,
        never by parsing a name. A ``quality_metric`` artifact written by
        ``qc.feature_percentage`` is a percentage. Other metrics are left out.
        """
        written: dict[str, MetricKind] = {}
        for name in self.assay_names:
            written[f"{name}_nCounts"] = "counts"
            written[f"{name}_nFeatures"] = "features"
            for column in self._get_assay(name)._percent_features():
                written[column] = "percentage"
        kinds = {attr: written[attr] for attr in attrs if attr in written}
        for source in artifact_metrics:
            status = self._artifacts_inspect(source.artifact)
            if status.operation == "run_feature_percentage":
                kinds[source.name] = "percentage"
        return kinds

    def _require_measured_metric_cells(
        self,
        attrs: Sequence[str],
        artifact_metrics: Sequence[NamedCellArtifact],
        cell_selection: ArtifactRef | None,
        *,
        operation: str,
    ) -> None:
        """Refuse QC metrics of an assay over cells that it did not measure.

        Such a cell holds zero counts, so its metrics are zero and would set
        the bounds of the measured cells. Without ``cell_selection``, the
        live ``I`` column, which the filter would snapshot, is checked
        before anything is written.
        """
        cells: ArtifactRef | str = "I" if cell_selection is None else cell_selection
        for assay in self._metric_assays(attrs, artifact_metrics):
            self._require_measured_cells(
                assay, cells, operation=operation, remedy="cell_selection"
            )

    @validation_scoped
    def _qc_auto_filter(
        self,
        attrs: Iterable[str] | None = None,
        min_p: float = 0.01,
        max_p: float = 0.99,
        *,
        cell_selection: ArtifactRef | None = None,
        artifact_metrics: Iterable[NamedCellArtifact] | None = None,
        invalidate_cache: bool = False,
        method: Literal["mad", "gaussian"] = "mad",
        sample_column: str | None = None,
        sample_artifact: NamedCellArtifact | None = None,
        n_mads: float = 3.0,
        min_cells_per_sample: int = 20,
        metric_policies: dict[str, dict[str, str]] | None = None,
    ) -> ArtifactRef:
        """Create an immutable automatically filtered cell selection.

        Defaults to median absolute deviation (MAD) bounds. Each metric's
        policy sets the scale of its bounds and which bounds apply:

        - Counts and feature counts use log1p values and two-sided bounds.
        - Percentages use raw values and an upper bound only, kept within 0
          to 100.
        - ``metric_policies`` sets the policy of named metrics explicitly.

        The kind of a metric comes from what CyteArc wrote, not from its name:
        the ``<assay>_nCounts`` and ``<assay>_nFeatures`` columns and the
        percentage columns that an assay's preparation wrote, and the
        ``quality_metric`` artifacts of ``qc.feature_percentage``, which are
        percentages. Any other metric, such as an imported ``total_counts``
        or ``pct_counts_mt`` column, gets the policy that its name suggests:
        counts for names ending in ``nCounts`` or ``nFeatures``, percentages
        for names ending in ``percentMito`` or ``percentRibo``, and raw
        two-sided bounds otherwise, with a warning that names those metrics.
        The resolved policies are recorded as ``metric_policies``.

        Bounds are exclusive: a cell whose value equals a bound is removed.
        MAD thresholds are pooled unless a sample source is supplied. Use
        ``method="gaussian"`` for pooled median/std Gaussian quantiles instead.

        Requested columns are read from current cell metadata when this method
        is called. Exact quality-metric artifacts can be supplied alongside
        metadata metrics. Metadata values are fingerprinted and artifact
        references are stored in provenance.

        These are the pipeline filtering rules. A cell whose metric is recorded
        as missing never passes and does not inform the bounds; the
        fingerprints of metadata-column missing-value masks are recorded in
        provenance.
        Non-finite metrics among the remaining active cells, missing sample
        labels among active cells, and an empty result raise ``ValueError``.

        Bounds assume one population of cells. A warning says when automatic
        bounds remove more than half of the selected cells, or when the MAD
        bounds of a log1p metric miss a second population: a separate group
        of cells above the upper bound, or a strongly bimodal metric whose
        low group the bounds keep. Both happen when empty droplets of a raw,
        unfiltered droplet matrix set the bounds; call cells or import the
        filtered matrix first.

        Args:
            attrs: Column names to be used for filtering.
            min_p: Quantile used for the lower threshold (Gaussian path only).
            max_p: Quantile used for the upper threshold (Gaussian path only).
            cell_selection: Optional prior cell-selection artifact.
            artifact_metrics: Named exact ``quality_metric`` artifact vectors.
            method: MAD bounds by default, or explicit pooled Gaussian bounds.
            sample_column: Optional cell-metadata column with sample labels.
                When set, MAD bounds are calculated within each sample.
            sample_artifact: Optional named exact ``hto_identity`` sample vector.
            n_mads: Number of scaled MADs used for bounds.
            min_cells_per_sample: Groups with fewer active cells than this are
                retained without MAD filtering and emit a warning.
            metric_policies: MAD policies of named metrics (MAD only). Each
                maps ``"transform"`` to ``"log1p"`` or ``"identity"`` and
                ``"bound_direction"`` to ``"two_sided"``, ``"lower"``, or
                ``"upper"``, for example
                ``{"total_counts": {"transform": "log1p",
                "bound_direction": "lower"}}``. Unnamed metrics keep their
                default policies.

        Returns:
            A complete datastore-scoped ``cell_selection`` artifact.

        Raises:
            TypeError: If ``metric_policies`` or one of its policies is not a
                mapping.
            ValueError: If ``metric_policies`` names a metric that is not
                filtered, holds an unknown key or value, or is given with
                ``method="gaussian"``.
            UnmeasuredCellsError: If a metric's assay did not measure a selected cell.
        """
        if method not in ("mad", "gaussian"):
            raise ValueError("method must be 'mad' or 'gaussian'")
        if method == "gaussian":
            if sample_column is not None or sample_artifact is not None:
                raise ValueError("Gaussian filtering does not support a sample source")
            if n_mads != 3.0 or min_cells_per_sample != 20:
                raise ValueError(
                    "n_mads and min_cells_per_sample apply only to method='mad'"
                )
            if metric_policies is not None:
                raise ValueError("metric_policies apply only to method='mad'")
        available = set(self.cells.columns)
        if attrs is None:
            attrs = []
            for i in ["nCounts", "nFeatures", "percentMito", "percentRibo"]:
                i = f"{self._defaultAssay}_{i}"
                if i in available:
                    attrs.append(i)

        attrs_list, metric_artifacts, resolved_sample_artifact = (
            validate_cell_filter_sources(
                attrs,
                artifact_metrics,
                sample_column=sample_column,
                sample_artifact=sample_artifact,
            )
        )
        missing = [attr for attr in attrs_list if attr not in available]
        if missing:
            joined = ", ".join(repr(attr) for attr in missing)
            raise KeyError(f"Cell metadata columns not found: {joined}")
        if method == "mad":
            explicit_policies = validate_metric_policies(
                metric_policies,
                [*attrs_list, *(source.name for source in metric_artifacts)],
            )
            return self._auto_filter_cells_mad(
                attrs=attrs_list,
                artifact_metrics=metric_artifacts,
                min_p=min_p,
                max_p=max_p,
                cell_selection=cell_selection,
                invalidate_cache=invalidate_cache,
                sample_column=sample_column,
                sample_artifact=resolved_sample_artifact,
                n_mads=n_mads,
                min_cells_per_sample=min_cells_per_sample,
                explicit_policies=explicit_policies,
            )

        self._require_measured_metric_cells(
            attrs_list, metric_artifacts, cell_selection, operation="auto_filter_cells"
        )
        prior, active_idx = self._filter_input_cells(cell_selection)
        values_by_name, missing_by_name, missing_fingerprints = (
            self._read_filter_metrics(attrs_list, metric_artifacts, prior, active_idx)
        )
        metadata_fingerprints = {
            attr: fingerprint_array(values_by_name[attr]) for attr in attrs_list
        }
        result = filter_cell_metrics(
            values_by_name,
            missing_by_name,
            np.ones(len(active_idx), dtype=bool),
            method="gaussian",
            min_p=min_p,
            max_p=max_p,
        )
        keep = np.zeros(self.cells.N, dtype=bool)
        keep[active_idx] = result.retained

        metric_sources = [
            {"name": attr, "source": "metadataColumn", "column": attr}
            for attr in attrs_list
        ]
        metric_sources.extend(
            {"name": source.name, "source": "artifact"} for source in metric_artifacts
        )
        inputs: dict[str, Any] = {
            "prior_cell_selection": prior.ref,
            "metadata_fingerprints": metadata_fingerprints,
            "artifact_metrics": {
                source.name: source.artifact for source in metric_artifacts
            },
        }
        if missing_fingerprints:
            inputs["missing_mask_fingerprints"] = missing_fingerprints

        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=keep,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="auto_filter_cells",
            parameters={
                "method": "gaussian",
                "attrs": list(values_by_name),
                "metric_sources": metric_sources,
                "min_p": min_p,
                "max_p": max_p,
                "resolved_bounds": result.gaussian_bounds,
            },
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        logger.info(f"Cell filtering retained {int(stored.sum())}/{self.cells.N} cells")
        return ref

    def _auto_filter_cells_mad(
        self,
        *,
        attrs: list[str],
        artifact_metrics: list[NamedCellArtifact],
        min_p: float,
        max_p: float,
        cell_selection: ArtifactRef | None,
        invalidate_cache: bool,
        sample_column: str | None,
        sample_artifact: NamedCellArtifact | None,
        n_mads: float,
        min_cells_per_sample: int,
        explicit_policies: dict[str, MetricPolicy],
    ) -> ArtifactRef:
        if min_p != 0.01 or max_p != 0.99:
            raise ValueError(
                "min_p and max_p apply only to method='gaussian'. "
                "Leave them at their defaults (0.01 and 0.99) and use n_mads "
                "to control MAD bounds"
            )
        if isinstance(n_mads, bool) or not isinstance(n_mads, Real):
            raise TypeError("n_mads must be a positive number")
        resolved_n_mads = float(n_mads)
        if not np.isfinite(resolved_n_mads) or resolved_n_mads <= 0:
            raise ValueError("n_mads must be finite and greater than 0")
        if (
            not isinstance(min_cells_per_sample, int)
            or isinstance(min_cells_per_sample, bool)
            or min_cells_per_sample < 2
        ):
            raise ValueError("min_cells_per_sample must be an integer >= 2")
        if sample_column is not None and sample_column not in self.cells.columns:
            raise ValueError(
                f"sample_column '{sample_column}' not found in cell metadata"
            )
        self._require_measured_metric_cells(
            attrs, artifact_metrics, cell_selection, operation="auto_filter_cells"
        )
        prior, active_idx = self._filter_input_cells(cell_selection)
        sample_options: dict[str, Any] = {}
        if sample_column is not None:
            sample_options = {
                "sample_labels": np.asarray(
                    read_metadata_rows_chunkwise(self.cells, sample_column, active_idx)
                ),
                "sample_missing": read_metadata_missing_rows_chunkwise(
                    self.cells,
                    sample_column,
                    active_idx,
                ),
                "sample_label_name": f"sample_column '{sample_column}'",
            }
        elif sample_artifact is not None:
            resolved_sample = resolve_cell_aligned_artifact(
                self.zw,
                sample_artifact.artifact,
                cell_selection=prior.ref,
                expected_kind="hto_identity",
            )
            sample_options = {
                "sample_labels": np.asarray(resolved_sample.values),
                "sample_missing": resolved_sample.missing_mask,
                "sample_label_name": f"sample_artifact '{sample_artifact.name}'",
            }
        values_by_attr, missing_by_attr, missing_fingerprints = (
            self._read_filter_metrics(attrs, artifact_metrics, prior, active_idx)
        )
        policies = resolve_metric_policies(
            list(values_by_attr),
            kinds=self._metric_kinds(attrs, artifact_metrics),
            explicit=explicit_policies,
        )
        result = filter_cell_metrics(
            values_by_attr,
            missing_by_attr,
            np.ones(len(active_idx), dtype=bool),
            method="mad",
            n_mads=resolved_n_mads,
            min_cells_per_sample=int(min_cells_per_sample),
            metric_policies=policies,
            **sample_options,
        )
        parameters: dict[str, Any] = {
            "method": "mad",
            "attrs": list(values_by_attr),
            "metric_sources": [
                *(
                    {"name": attr, "source": "metadataColumn", "column": attr}
                    for attr in attrs
                ),
                *(
                    {"name": source.name, "source": "artifact"}
                    for source in artifact_metrics
                ),
            ],
            "n_mads": resolved_n_mads,
            "min_cells_per_sample": int(min_cells_per_sample),
            "resolved_bounds": {},
        }
        if sample_column is not None:
            parameters["sample_column"] = sample_column
            parameters["sample_source"] = {
                "name": sample_column,
                "source": "metadataColumn",
                "column": sample_column,
            }
        elif sample_artifact is not None:
            parameters["sample_source"] = {
                "name": sample_artifact.name,
                "source": "artifact",
            }
        else:
            parameters["sample_source"] = {"source": "pooled"}

        if result.mad_provenance is not None:
            # Warnings are logged below rather than recorded.
            parameters.update(
                (key, value)
                for key, value in result.mad_provenance.items()
                if key != "warnings"
            )
        inputs: dict[str, Any] = {
            "prior_cell_selection": prior.ref,
            "qc_metric_fingerprints": {
                attr: fingerprint_array(values_by_attr[attr]) for attr in attrs
            },
            "artifact_metrics": {
                source.name: source.artifact for source in artifact_metrics
            },
        }
        if missing_fingerprints:
            inputs["missing_mask_fingerprints"] = missing_fingerprints
        if sample_column is not None:
            assert result.sample_labels is not None
            inputs["sample_assignments_fingerprint"] = fingerprint_strings(
                result.sample_labels
            )
        elif sample_artifact is not None:
            inputs["sample_artifact"] = sample_artifact.artifact

        if result.mad_provenance is not None:
            for message in result.mad_provenance["warnings"]:
                logger.warning(message)
        keep = np.zeros(self.cells.N, dtype=bool)
        keep[active_idx] = result.retained

        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=keep,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="auto_filter_cells",
            parameters=parameters,
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        logger.info(f"Cell filtering retained {int(stored.sum())}/{self.cells.N} cells")
        return ref

    @validation_scoped
    def _qc_feature_percentage(
        self,
        cell_selection: ArtifactRef,
        features: ArtifactRef,
        *,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Persist the percentage of counts assigned to selected features.

        Args:
            cell_selection: Exact datastore-scoped cell-selection artifact.
            features: Exact assay-scoped feature-selection artifact.
            invalidate_cache: Recompute instead of reusing an exact artifact.

        Returns:
            A complete assay-scoped ``quality_metric`` artifact.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a selected cell.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if features.scope != "assay" or features.assay is None:
            raise ValueError("features must be an assay-scoped ArtifactRef")
        assay = self._get_assay(features.assay)
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if len(cell_index) == 0:
            raise ValueError("cell_selection must select at least one cell")
        # The stored mask of the selection is read in bounded blocks.
        self._require_measured_cells(
            assay.name, cell_selection, operation="run_feature_percentage"
        )
        feature_selection = resolve_feature_selection(
            self.zw,
            assay.name,
            features,
        )
        feature_values = np.asarray(
            artifact_values(
                artifact_group(self.zw, feature_selection),
                "values",
            ),
            dtype=bool,
        )
        feature_index = np.flatnonzero(feature_values).astype(
            np.int64,
            copy=False,
        )

        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind="quality_metric",
            operation="run_feature_percentage",
            parameters={"scale": 100.0},
            inputs={"feature_selection": feature_selection},
            execution_options={"nthreads": assay.nthreads},
            cell_selection=cell_selection,
            arrays={"values": ((len(cell_index),), "f")},
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_feature_percentage")
        values = assay._compute_feature_percentage(cell_index, feature_index)
        write_cell_data_artifact(self.zw, planned, {"values": values})
        return planned.ref

    @validation_scoped
    def _qc_hto_demultiplexing(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        random_seed: int = 0,
        positive_quantile: float = 0.99,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Demultiplex HTO counts into an immutable identity artifact.

        The method follows Seurat's ``HTODemux``: a count is positive when it
        exceeds the ``positive_quantile`` quantile of the negative-binomial
        distribution fitted to its HTO's background cluster. A cell with no
        positive HTO is ``Negative``, one with several is ``Doublet``, and a
        singlet takes the one HTO called positive. Unlike ``HTODemux``, which
        names a singlet after its highest normalized (CLR) value, a singlet is
        therefore never assigned an HTO that its cutoff called negative.

        The artifact's ``values`` array holds the identities. Diagnostic arrays
        hold, for each cell, the HTOs with the highest and the second highest
        CLR value (``max_id`` and ``second_id``, Seurat's ``maxID`` and
        ``secondID``) and the difference of those values (``margin``). With one
        HTO, ``second_id`` is empty and ``margin`` is NaN.

        A background cluster that holds only zero counts gets a cutoff of 0, so
        every nonzero count of that HTO is positive, with a warning. A warning
        also names each HTO whose positive calls overlap its background: fewer
        than half of its positive cells have counts that the background fit
        exceeds with probability ``(1 - positive_quantile) ** 3``, one in a
        million for the default. Such calls, for example of an unused tag or
        of pure background, may be background.

        Args:
            from_assay: HTO assay name (default: ``'HTO'``).
            cell_selection: Explicit cell-selection artifact.
            random_seed: Seed used for HTO demultiplexing.
            positive_quantile: Quantile of each HTO's background distribution
                that a count must exceed to be positive, Seurat's
                ``positive.quantile`` (default 0.99). It must be greater than 0
                and less than 1. A higher quantile, such as 0.999, calls fewer
                background counts positive, and so fewer false doublets, but
                can miss weakly stained tags. The artifact records the
                quantile, so each quantile has its own artifact.
            invalidate_cache: Recompute even when matching provenance exists.

        Returns:
            A complete ``hto_identity`` artifact.

        Raises:
            TypeError: If ``random_seed`` is not an integer or
                ``positive_quantile`` is not a real number.
            ValueError: If ``positive_quantile`` is not finite or not between
                0 and 1.
            UnmeasuredCellsError: If the HTO assay did not measure a selected cell.
        """
        from ...assay.classification import declared_assay_type

        if from_assay is None:
            from_assay = "HTO"
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        # The type this datastore resolved for the assay, which its store
        # records; an assay that the store lacks declares none.
        declared_type = (
            declared_assay_type(self._get_assay(from_assay))
            if from_assay in self.assay_names
            else None
        )
        if declared_type != "HTO":
            raise TypeError(
                "HTO demultiplexing requires an assay declared with type 'HTO'; "
                f"{from_assay!r} is declared as {declared_type!r}"
            )
        assay = self._get_assay(from_assay)
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        positive_quantile = _validated_positive_quantile(positive_quantile)
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = len(cell_index)
        required_cells = assay.feats.N + 1
        if n_cells < required_cells:
            raise ValueError(
                f"HTO demultiplexing requires at least {required_cells} selected cells"
            )
        # The stored mask of the selection is read in bounded blocks.
        self._require_measured_cells(
            assay.name, cell_selection, operation="run_hto_demultiplexing"
        )
        arguments = HtoIdentityArguments(
            cell_selection=cell_selection,
            feature_ids_fingerprint=fingerprint_strings(
                np.asarray(assay.feats.fetch_all("ids"))
            ),
            method=_hto_demux_method(positive_quantile),
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={
                "values": ((n_cells,), None),
                "max_id": ((n_cells,), None),
                "second_id": ((n_cells,), None),
                "margin": ((n_cells,), "f"),
            },
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_hto_demultiplexing")
        matrix_bytes = n_cells * assay.feats.N * np.dtype(np.float64).itemsize
        # Per cell: the cluster label, the identity, and the three diagnostics.
        estimated_peak_bytes = 6 * matrix_bytes + 40 * n_cells
        if estimated_peak_bytes > self.memoryBytes:
            raise MemoryError(
                "HTO demultiplexing needs an estimated "
                f"{estimated_peak_bytes / (1024**2):.1f} MiB for its exact "
                "whole-matrix clustering algorithm, which exceeds the datastore "
                f"memory budget of {self.memoryBytes / (1024**2):.1f} MiB. "
                "Use a smaller explicit cell selection or a larger memory budget."
            )
        counts = pd.DataFrame(
            controlled_compute(assay.rawData[cell_index, :], self.nthreads),
            columns=assay.feats.fetch_all("ids"),
        )
        hto_idents = hto_demux(
            counts,
            random_seed=random_seed,
            positive_quantile=positive_quantile,
        )
        diagnostics = _hto_clr_diagnostics(counts)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": np.asarray(hto_idents.values),
                "max_id": np.asarray(diagnostics["max_id"].to_numpy()),
                "second_id": np.asarray(diagnostics["second_id"].to_numpy()),
                "margin": np.asarray(
                    diagnostics["margin"].to_numpy(), dtype=np.float64
                ),
            },
        )
        return planned.ref

    def _run_doublet_detection_artifact(
        self,
        *,
        source_assay: RNAassay,
        cell_selection: ArtifactRef,
        clusters: ArtifactRef,
        cluster_values: np.ndarray,
        connectivity: ArtifactRef,
        feature_snapshot: ArtifactRef | None = None,
        sample_column: str | Sequence[str] | None = None,
        sample_labels: Sequence[np.ndarray] | None = None,
        cluster_sample_fraction: float = 1.0,
        max_cells_per_cluster: int | None = None,
        simulation_ratio: float = 2.0,
        heterotypic_fraction: float = 0.0,
        save_k: int = 30,
        expected_doublet_rate: float = 0.05,
        threshold: float | None = None,
        smoothing_t: int = 0,
        normalize_scores: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
        one_cluster_remedy: str = _ONE_CLUSTER_DOUBLET_REMEDY,
    ) -> ArtifactRef:
        """Create doublet scores and calls without creating metadata columns.

        ``sample_column`` names one cell metadata column, or several, and
        ``sample_labels`` holds their labels for the selected cells, one array
        per column: the cells that share every label form one sample.
        ``one_cluster_remedy`` ends the error raised when
        ``heterotypic_fraction`` is above 0 and every cell has the same
        cluster label, so a caller can name the settings that it exposes.
        """
        from ...quality_control.doublets import (
            DoubletIndexSettings,
            minmax_doublet_scores,
            plan_doublet_samples,
            score_synthetic_doublets,
            smooth_doublet_scores,
        )
        from ...storage.execution import admit_stream

        require_graph_kind(connectivity)
        # Arguments are checked before any reference, score, or artifact work.
        fraction = _validated_real(
            cluster_sample_fraction,
            "cluster_sample_fraction",
            low=0.0,
            high=1.0,
            include_low=False,
        )
        cap = (
            None
            if max_cells_per_cluster is None
            else integer_argument(
                max_cells_per_cluster, "max_cells_per_cluster", minimum=1
            )
        )
        ratio = _validated_real(
            simulation_ratio, "simulation_ratio", low=0.0, include_low=False
        )
        heterotypic = _validated_real(
            heterotypic_fraction, "heterotypic_fraction", low=0.0, high=1.0
        )
        neighbors_k = integer_argument(save_k, "save_k", minimum=1)
        rate = _validated_real(
            expected_doublet_rate, "expected_doublet_rate", low=0.0, include_low=False
        )
        if rate >= 1:
            raise ValueError(
                "expected_doublet_rate must be a finite number > 0 and < 1"
            )
        cutoff = (
            None
            if threshold is None
            else _validated_real(threshold, "threshold", low=0.0, high=1.0)
        )
        smoothing = integer_argument(smoothing_t, "smoothing_t", minimum=0)
        seed = integer_argument(random_seed, "random_seed", minimum=0)
        if not isinstance(normalize_scores, bool):
            raise TypeError("normalize_scores must be a boolean")
        columns = _doublet_sample_columns(sample_column)
        if len(columns) != (0 if sample_labels is None else len(sample_labels)):
            raise ValueError("Pass one array of sample labels per sample column")
        labels = np.asarray(cluster_values)
        if labels.ndim != 1:
            raise ValueError("Cluster values must contain one label per selected cell")
        # Forced heterotypic doublets need a partner outside the first
        # parent's cluster, so one cluster fails before any planning.
        if heterotypic > 0 and len(np.unique(labels)) < 2:
            raise ValueError(
                "Doublet detection with heterotypic_fraction="
                f"{heterotypic:g} pairs cells from two different "
                "clusters, but every selected cell has the same cluster label. "
                f"{one_cluster_remedy}"
            )
        assay_name = source_assay.name
        connectivity_status = self._require_complete_artifact(
            connectivity,
            connectivity.kind,
            assay=(assay_name if connectivity.scope == "assay" else None),
        )
        if connectivity.kind == "connectivity_map" and (
            connectivity_status.operation != "build_connectivity_map"
        ):
            raise ValueError(
                "Doublet detection requires a build_connectivity_map artifact"
            )
        lineage = resolve_graph_assay_inputs(
            self.zw,
            connectivity,
            assay_name,
        )
        neighbors = lineage.neighbors
        coordinates = lineage.coordinates
        if coordinates.kind == "batch_correction":
            # Within one sample there is no batch to correct, so samples are
            # scored on the PCA that the correction started from.
            if not columns:
                raise ValueError(
                    "Doublet detection on a batch-corrected graph needs "
                    "sample_column, the column of each cell's capture: doublets "
                    "are then simulated and scored within each sample, on the "
                    "PCA that the correction started from. To score every cell "
                    "together, pass an uncorrected PCA graph"
                )
        elif coordinates.kind != "reduction":
            raise ValueError("Doublet detection requires an uncorrected PCA graph")
        active_idx = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if len(labels) != len(active_idx):
            raise ValueError("Cluster values must contain one label per selected cell")
        n_active = len(active_idx)
        codes, samples, assignments = sample_codes(
            columns,
            sample_labels or (),
            n_active,
            reason="Doublets are simulated within each sample, so every selected "
            "cell needs one",
        )
        sizes = np.bincount(codes, minlength=len(samples))
        small = [sample for sample, size in zip(samples, sizes) if size < 2]
        if small:
            raise ValueError(
                "Doublets are simulated within each sample, from pairs of its "
                f"cells, but sample {_doublet_sample_text(small[0])} has one "
                "selected cell. Score a cell selection without such samples, "
                "or pass a sample column that groups them with the rest of "
                "their capture"
            )
        plans = plan_doublet_samples(
            sizes, simulation_ratio=ratio, n_neighbors=neighbors_k
        )
        arguments = DoubletScoreArguments(
            clusters=clusters,
            connectivity_map=connectivity,
            neighbors=neighbors,
            sample_assignments_fingerprint=assignments,
            cluster_sample_fraction=fraction,
            max_cells_per_cluster=cap,
            sample_columns=columns,
            simulation_ratio=ratio,
            heterotypic_fraction=heterotypic,
            # Each sample counts at most its other cells, so the neighbors
            # recorded are the most that any sample uses.
            save_k=max(plan.neighbors for plan in plans),
            expected_doublet_rate=rate,
            threshold=cutoff,
            smoothing_t=smoothing,
            normalize_scores=normalize_scores,
            random_seed=seed,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        inputs = dict(record.inputs)
        if feature_snapshot is not None:
            inputs["feature_snapshot"] = feature_snapshot
        inputs["cell_selection"] = cell_selection
        n_simulated = sum(plan.simulated for plan in plans)
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=assay_name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("values", shape=(n_active,), dtype=np.float64),
                ArrayRequirement("raw_scores", shape=(n_active,), dtype=np.float64),
                ArrayRequirement(
                    "predicted_doublets", shape=(n_active,), dtype=np.bool_
                ),
                ArrayRequirement(
                    "simulated_scores", shape=(n_simulated,), dtype=np.float64
                ),
                ArrayRequirement(
                    "simulated_parents", shape=(n_simulated, 2), dtype=np.int64
                ),
            ),
            required_attributes=(
                AttributeRequirement("samples", expected_types=(list,)),
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_doublet_detection")

        feature_selection = lineage.feature_selection
        # A reduction's lineage always names its normalized feature selection.
        assert feature_selection is not None
        reference_ref = self._mapping_build_reference(neighbors)
        reference = self._mapping_load_reference(reference_ref)

        feature_indices = read_feature_selection_indices(
            self.zw,
            assay_name,
            feature_selection,
        )
        feature_ids = read_metadata_rows_chunkwise(
            source_assay.feats,
            "ids",
            feature_indices,
        )
        if not np.array_equal(
            np.asarray(feature_ids).astype(str), reference.feature_ids
        ):
            raise ValueError(
                "Doublet features do not match the mapping reference order"
            )
        # Each sample's index follows the settings of the graph's ANN index,
        # which records every one of them.
        ann = self._artifacts_inspect(reference.ann_index).parameters or {}
        index = DoubletIndexSettings(
            metric=reference.ann_metric,
            m=ann["ann_m"],
            ef_construction=ann["ann_efc"],
            ef=ann["ann_ef"],
            parallel=ann["ann_parallel"],
        )
        totals = (
            None
            if reference.normalization_parameters["renormalize_subset"]
            else source_assay._cell_count_totals(active_idx)
        )
        result = score_synthetic_doublets(
            source_assay,
            reference,
            active_idx,
            labels,
            codes,
            feature_indices,
            totals,
            index=index,
            cluster_sample_fraction=fraction,
            max_cells_per_cluster=cap,
            simulation_ratio=ratio,
            heterotypic_fraction=heterotypic,
            n_neighbors=neighbors_k,
            expected_doublet_rate=rate,
            threshold=cutoff,
            random_seed=seed,
            resources=self.resources,
        )
        del reference
        if result.scores.shape != (n_active,):
            raise RuntimeError("Doublet scores do not match the selected cells")
        for sample, outcome in zip(samples, result.samples, strict=True):
            if outcome.heterotypic_fraction < heterotypic:
                logger.warning(
                    f"Every parent cell of sample {_doublet_sample_text(sample)} "
                    "has the same cluster label, so its doublets pair any two "
                    f"of its cells instead of forcing heterotypic_fraction="
                    f"{heterotypic:g} of them to pair two clusters"
                )
        values = result.scores
        if smoothing:
            graph_group = artifact_group(self.zw, connectivity)
            edges = as_zarr_array(graph_group["edges"], name="edges")
            weights = as_zarr_array(graph_group["weights"], name="weights")
            # Loading, symmetrizing and row-normalizing may hold several
            # sparse copies.
            graph_bytes = (
                int(edges.nbytes)
                + int(weights.nbytes)
                + 8 * int(weights.shape[0]) * (weights.dtype.itemsize + 8)
                + 128 * (n_active + 1)
            )
            admit_stream(
                self.resources,
                nBlocks=1,
                blockBytes=graph_bytes,
                residentBytes=active_idx.nbytes
                + labels.nbytes
                + 3 * result.scores.nbytes
                + result.simulated_scores.nbytes
                + result.simulated_parents.nbytes
                + feature_ids.nbytes
                + feature_indices.nbytes,
                requested=1,
            )
            graph = self._load_graph_artifact(
                connectivity, symmetric=True, upper_only=False, use_k=None
            )
            if graph.shape != (n_active, n_active):
                raise ValueError("Doublet graph does not match the selected cells")
            values = smooth_doublet_scores(
                graph,
                values,
                power=smoothing,
                normalize=normalize_scores,
            )
        elif normalize_scores:
            values = minmax_doublet_scores(values)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": values,
                "raw_scores": result.raw_scores,
                "predicted_doublets": result.predicted,
                "simulated_scores": result.simulated_scores,
                "simulated_parents": result.simulated_parents,
            },
            attributes={
                "samples": [
                    _doublet_sample_summary(sample, outcome)
                    for sample, outcome in zip(samples, result.samples, strict=True)
                ]
            },
        )
        logger.info(
            f"Stored doublet scores: {int(result.predicted.sum())} of {n_active} "
            f"cells called doublets in {len(samples)} sample(s)"
        )
        return planned.ref

    @validation_scoped
    def _qc_doublets(
        self,
        clusters: ArtifactRef,
        graph: ArtifactRef,
        *,
        from_assay: str | None = None,
        cluster_sample_fraction: float = 1.0,
        max_cells_per_cluster: int | None = None,
        sample_column: str | None = None,
        simulation_ratio: float = 2.0,
        heterotypic_fraction: float = 0.0,
        save_k: int = 30,
        expected_doublet_rate: float = 0.05,
        threshold: float | None = None,
        smoothing_t: int = 0,
        normalize_scores: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Score and call doublets from simulated doublets, within each sample.

        Doublets are simulated by adding the counts of two cells of one
        sample, every cell of the sample by default, so their cell types are
        as common as in the sample. They are normalized like the graph's
        cells and projected onto its PCA. Every cell and simulated doublet is
        then scored on its ``k`` nearest neighbors among the cells and
        simulated doublets of its own sample, with ``k = round(save_k * (1 +
        r))`` for ``r`` simulated doublets per cell: the score is the
        probability that the cell is a doublet given the fraction of
        simulated doublets among those neighbors and ``expected_doublet_rate``,
        Scrublet's doublet score (Wolock, Lopez and Klein, 2019). A cell is a
        predicted doublet when its score is above its sample's threshold:
        ``threshold``, or the minimum between the two modes of the sample's
        simulated-doublet scores (Otsu's threshold when they are not
        bimodal).

        The ``doublet_score`` artifact holds, for each cell of the graph's
        selection, the score in ``values``, the fraction of simulated
        doublets among its neighbors in ``raw_scores``, and the call in
        ``predicted_doublets``; ``ds.select_cells(ref,
        value="predicted_doublets", include=[False])`` keeps the predicted
        singlets. ``simulated_scores`` holds the score of each simulated
        doublet and ``simulated_parents`` the positions of its two parents
        among the selected cells, sample after sample. The ``samples``
        attribute lists each sample's labels, cells, simulated doublets,
        neighbors, threshold and its method, calls, the fraction of simulated
        doublets above the threshold, and Scrublet's estimate of the doublet
        rate.

        Args:
            clusters: Cluster labels over the graph's cell selection. They
                matter only with ``heterotypic_fraction`` above 0 or a parent
                pool smaller than every cell.
            graph: A connectivity map built on a PCA reduction. A map built on
                a Harmony correction of one needs ``sample_column``.
            from_assay: The RNA assay of the graph; required for an
                integrated graph.
            cluster_sample_fraction: The fraction of each cluster of a sample
                that may parent doublets, drawn at random.
            max_cells_per_cluster: The most cells of each cluster of a sample
                that may parent doublets, or None for no limit. A limit makes
                rare clusters parent more doublets than their share of cells.
            sample_column: A cell metadata column naming each cell's capture
                (10x channel or lane), or None to treat every cell as one
                sample. Doublets form only within one capture, so they are
                simulated from, and scored against, cells of the same sample.
                Cells hashed into one capture share it: do not pass hashtag
                identities.
            simulation_ratio: Simulated doublets per cell of each sample.
            heterotypic_fraction: The share of simulated doublets forced to
                pair cells of two clusters; 0 pairs any two cells of a
                sample.
            save_k: The neighbors that the scaling above starts from. A
                sample counts at most its other cells, and the artifact
                records the most that any sample uses.
            expected_doublet_rate: The prior doublet rate of each sample,
                above 0 and below 1.
            threshold: The doublet score from 0 to 1 above which a cell is a
                predicted doublet in every sample, or None to choose one per
                sample.
            smoothing_t: Diffuse ``values`` this many steps over the graph; 0
                keeps each cell's own score.
            normalize_scores: Scale ``values`` to [0, 1]. Neither option
                changes ``raw_scores``, the calls, or the thresholds.
            random_seed: The seed of the simulation and of each sample's
                neighbor index.
            invalidate_cache: Create a fresh result instead of reusing a match.

        Returns:
            A complete assay-scoped ``doublet_score`` artifact.

        Raises:
            ValueError: If ``heterotypic_fraction`` is above 0 with one
                cluster, a sample has fewer than two cells or a missing
                label, or an argument is out of range.
            UnmeasuredCellsError: If the assay did not measure a cell of the graph.
            MemoryError: If a sample's simulation or neighbor index exceeds
                the memory budget.
        """
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        if not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef")
        if sample_column is not None and (
            not isinstance(sample_column, str) or not sample_column
        ):
            raise TypeError("sample_column must be a non-empty string or None")
        if sample_column is not None and sample_column not in self.cells.columns:
            raise ValueError(
                f"sample_column '{sample_column}' not found in cell metadata"
            )
        assay_name = from_assay or graph.assay
        if assay_name is None:
            raise ValueError("from_assay is required for an integrated graph")
        resolve_graph_assay_inputs(self.zw, graph, assay_name)
        assay = self._get_assay(assay_name)
        graph_selection = graph_cell_selection(self.zw, graph)
        validate_stored_selection_integrity(
            self.zw,
            graph_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Doublet detection is only supported for RNA assays; "
                f"received {type(assay).__name__}"
            )
        self._require_measured_cells(
            assay.name,
            graph_selection,
            operation="run_doublet_detection",
            remedy="graph",
        )
        cluster_status = self._artifacts_inspect(clusters)
        if (
            clusters.kind not in {"cluster_labels", "cluster_cut"}
            or not cluster_status.exists
            or not cluster_status.complete
        ):
            raise ValueError("clusters must be a complete clustering artifact")
        raw_cluster_selection = (cluster_status.inputs or {}).get("cell_selection")
        if not isinstance(raw_cluster_selection, dict):
            raise ValueError("Clustering artifact has no cell-selection input")
        try:
            cluster_selection = ArtifactRef.from_dict(raw_cluster_selection)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Clustering artifact cell selection is malformed") from exc
        validate_stored_selection_integrity(
            self.zw,
            cluster_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if not self._selection_artifacts_match(graph_selection, cluster_selection):
            raise ValueError("Cluster and graph cell selections do not match")
        cluster_group = as_zarr_group(
            self.zw[artifact_path(clusters)],
            name=clusters.artifact_id,
        )
        value_name = grouping_value_name(clusters.kind)
        require_complete_cluster_labels(cluster_group, value_name, name="clusters")
        cluster_values = artifact_values(cluster_group, value_name)
        sample_labels = None
        if sample_column is not None:
            rows = read_stored_selection_indices(
                self.zw,
                graph_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            sample_labels = [
                apply_missing_mask(
                    np.asarray(
                        read_metadata_rows_chunkwise(self.cells, sample_column, rows)
                    ),
                    read_metadata_missing_rows_chunkwise(
                        self.cells, sample_column, rows
                    ),
                    labels=True,
                )
            ]
        ref = self._run_doublet_detection_artifact(
            source_assay=assay,
            cell_selection=graph_selection,
            clusters=clusters,
            cluster_values=cluster_values,
            connectivity=graph,
            sample_column=sample_column,
            sample_labels=sample_labels,
            cluster_sample_fraction=cluster_sample_fraction,
            max_cells_per_cluster=max_cells_per_cluster,
            simulation_ratio=simulation_ratio,
            heterotypic_fraction=heterotypic_fraction,
            save_k=save_k,
            expected_doublet_rate=expected_doublet_rate,
            threshold=threshold,
            smoothing_t=smoothing_t,
            normalize_scores=normalize_scores,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        return ref

    @validation_scoped
    def _features_prevalent_peaks(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        top_n: int = 10000,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Feature selection method for ATACassay type assays.

        This method first calculates prevalence of each peak by computing sum of TF-IDF normalized values for each peak
        and then marks `top_n` peaks with the highest prevalence as prevalent peaks.

        Args:
            from_assay: Assay to use for peak selection. If no value is provided
                then `defaultAssay` will be used
            cell_selection: Explicit cells used to calculate peak prevalence.
            top_n: Number of top prevalent peaks to be selected, from 1 to one
                fewer than the number of peaks. (Default: 10000)
        Returns:
            The persisted prevalent-peak feature-selection artifact.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a selected cell.
            PermissionError: If the selection or its feature summary must be
                computed and the store is not opened with ``zarr_mode='r+'``.
                A read-only store returns a saved selection.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        if type(assay) != ATACassay:  # noqa: E721
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to ATACassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        top_n = integer_argument(top_n, "top_n", minimum=1)
        if top_n >= assay.feats.N:
            raise ValueError(
                f"top_n must be less than the number of peaks ({assay.feats.N})"
            )
        self._require_measured_cells(
            assay.name, cell_selection, operation="select_prevalent_peaks"
        )
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            invalidate_cache=invalidate_cache,
        )
        arguments = PrevalentPeakArguments(
            feature_summary=summary_ref,
            top_n=top_n,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay.z)
        planned = _feature_selection_plan(
            self.zw,
            assay=assay.name,
            n_features=assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            self._require_writable("select_prevalent_peaks")
            n_selected = feature_summary_selected_count(self.zw, cell_selection)
            summary = feature_summary_values(
                self.zw,
                summary_ref,
                n_selected=n_selected,
            )
            values = assay._prevalent_peak_mask(summary["prevalence"], top_n)
            _write_feature_selection(
                self.zw,
                planned,
                ordered_feature_ids_fingerprint=feature_ids_fingerprint,
                payload={"values": values},
            )
        return planned.ref

    @validation_scoped
    def _scores_cell_cycle(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        s_genes: list[str] | None = None,
        g2m_genes: list[str] | None = None,
        ctrl_size: int | None = None,
        log_transform: bool | None = None,
        n_bins: int = 50,
        rand_seed: int = 4466,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Computes S and G2M phase scores by taking into account the average
        expression of S and G2M phase genes respectively. Following steps are
        taken for each phase:

        - Gene names match feature names without case, and a name matches
          every feature with it. A gene that its listed symbol misses matches
          by another HGNC symbol of the same gene in
          ``cytearc.quality_control.cell_cycle_previous_symbols``, such as
          FAM64A for PIMREG; the artifact records such genes as
          ``gene_aliases`` and the log names them. A gene matched more than
          once counts once.
        - Normalized expression is log1p transformed when `log_transform` is True.
        - Every feature's mean over the selected cells sets its expression
          bin, as in scanpy's ``score_genes``: with ``N`` features a bin holds
          ``k = round(N / (n_bins - 1))`` of them, and a feature whose mean
          ranks ``r`` (from 1, tied means sharing the lowest rank) falls in bin
          ``r // k``. All features with a mean of 0 share the lowest bin,
          which can leave the next bins empty.
        - For each bin that holds a phase gene, ``ctrl_size`` control genes
          are drawn from all genes of the bin (the whole bin when it holds no
          more), and phase genes among them are dropped. Controls are drawn
          once per bin rather than for each phase gene as in Tirosh et al.
          (2016), and the S and G2M draws each start from
          ``RandomState(rand_seed)``.
        - The average expression of phase genes (Ep) and control genes (Ec) is calculated per cell.
        - A phase score is calculated as ``Ep - Ec``.
        - G1 is assigned when both scores are negative.
        - G2M is assigned when the G2M score exceeds the S score.
        - S is assigned otherwise, including tied non-negative scores.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_selection: Explicit cells to score.
            s_genes: A list of S phase genes. If not provided then CyteArc loads pre-saved genes accessible at
                     `cytearc.quality_control.s_phase_genes`
            g2m_genes: A list of G2M phase genes. If not provided then CyteArc loads pre-saved genes accessible at
                     `cytearc.quality_control.g2m_phase_genes`
            ctrl_size: Controls sampled per bin. None uses the shorter input gene list.
            log_transform: Apply log1p before binning and scoring. None applies it
                           only for ``norm_lib_size``.
            n_bins: Number of bins into which average expression of genes is divided.
            rand_seed: A random seed for sampling control genes. (Default value: 4466)
        Returns:
            A complete ``cell_cycle`` artifact containing S, G2M, and phase values.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a selected cell.
            PermissionError: If the scores or their feature summary must be
                computed and the store is not opened with ``zarr_mode='r+'``.
                A read-only store returns saved scores.
        """
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Cell-cycle scoring can only be applied to an RNAassay; "
                f"received {type(assay).__name__}"
            )
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        return self._run_cell_cycle_scoring_artifact(
            assay=assay,
            cell_selection=cell_selection,
            s_genes=s_genes,
            g2m_genes=g2m_genes,
            ctrl_size=ctrl_size,
            log_transform=log_transform,
            n_bins=n_bins,
            rand_seed=rand_seed,
            invalidate_cache=invalidate_cache,
        )
