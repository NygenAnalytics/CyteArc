"""Differential-abundance operations of ``DataStore.abundance``.

The datastore resolves labels and sample metadata, plans and persists the
artifacts, and delegates the statistics to :mod:`cytearc.abundance`.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import zarr

from ...abundance.milo import (
    annotate_neighborhoods,
    block_membership,
    count_block,
    differential_abundance,
    kth_distances,
    refine_index_cells,
    reverse_neighbors,
    sample_index_cells,
    unique_in_order,
)
from ...abundance.propeller import count_categories, propeller
from ...abundance.results import (
    CategoryTestResult,
    NeighborhoodCounts,
    Neighborhoods,
    NeighborhoodTestResult,
)
from ...features.statistical import value_fingerprint
from ...graph.distances import validate_distance_provenance
from ...graph.feature_projection import resolve_native_graph_inputs
from ...metadata.arguments import (
    CategoryTestArguments,
    NeighborhoodArguments,
    NeighborhoodCountArguments,
    NeighborhoodTestArguments,
)
from ...metadata.rows import read_metadata_missing_rows, read_metadata_rows
from ...metadata.selection import (
    CellField,
    StudyDesign,
    resolve_grouping,
    valid_category_mask,
)
from ...metadata.rows import read_array_rows_chunkwise
from ...neighbors.stages import AnnIndexStage
from ...stats.design import contrast_vector, treatment_design
from ...storage.arrays import create_numeric_array
from ...storage.artifact_writer import artifact_transaction
from ...storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ...storage.layout import _group_zarr_format, row_sharded_array_spec
from ...storage.profiles import resolve_storage_profile
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.validation_scope import validation_scoped
from ...utils.logging import logger
from .abundance_store import (
    NEIGHBORHOOD_ARRAYS,
    read_category_slot,
    read_neighborhood_counts_slot,
    read_neighborhood_test_slot,
    read_neighborhoods_slot,
    write_category_slot,
    write_neighborhood_test_slot,
    write_small_array,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _AbundanceOperationsBase
else:
    _AbundanceOperationsBase = object


# Rows of graph cells or neighbour lists handled per block.
_MILO_BLOCK_ROWS = 1 << 16


def _read_rows(array: zarr.Array, rows: np.ndarray) -> np.ndarray:
    """Read possibly repeated rows of an array, one distinct row at a time."""
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    values = np.asarray(read_array_rows_chunkwise(array, unique))
    return values[inverse.ravel()]


def _sorted_labels(values: np.ndarray) -> list[str]:
    """Return distinct labels in R's factor order: numeric when all are numbers."""
    unique = list(dict.fromkeys(values.tolist()))
    if all(
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool)
        for value in unique
    ):
        return [str(value) for value in sorted(unique)]
    return sorted(str(value) for value in unique)


def _resolve_design_columns(
    study_design: StudyDesign | None,
    sample_by: str | None,
    condition_by: str | None,
) -> tuple[str, str]:
    if study_design is not None:
        if sample_by is not None or condition_by is not None:
            raise ValueError(
                "Pass either study_design or sample_by and condition_by, not both"
            )
        if study_design.condition_by is None:
            raise ValueError("study_design must set condition_by")
        return study_design.sample_by, study_design.condition_by
    if sample_by is None or condition_by is None:
        raise ValueError("sample_by and condition_by are required")
    if sample_by == condition_by:
        raise ValueError("sample_by and condition_by must be different columns")
    return sample_by, condition_by


def sample_conditions(samples: np.ndarray, conditions: np.ndarray) -> dict[str, str]:
    """Map each sample to its one condition.

    Raises:
        ValueError: If a sample has cells from more than one condition.
    """
    pairs = dict.fromkeys(zip(samples.tolist(), conditions.tolist(), strict=True))
    mapping: dict[str, str] = {}
    for sample, condition in pairs:
        previous = mapping.setdefault(sample, condition)
        if previous != condition:
            raise ValueError(
                f"Sample {sample!r} has cells from conditions {previous!r} and "
                f"{condition!r}. Each sample must belong to one condition."
            )
    return mapping


class _AbundanceOperationsMixin(_AbundanceOperationsBase):
    """Propeller and Milo operations behind ``DataStore.abundance``."""

    def _read_complete_labels(
        self, column: str, cell_idx: np.ndarray, *, role: str
    ) -> np.ndarray:
        if column not in self.cells.columns:
            raise KeyError(f"Cell metadata column {column!r} was not found")
        values = np.asarray(read_metadata_rows(self.cells, column, cell_idx))
        missing = read_metadata_missing_rows(self.cells, column, cell_idx)
        if not valid_category_mask(values, missing_mask=missing).all():
            raise ValueError(
                f"The {role} column {column!r} has missing values in the tested "
                "cells. Pass a cell_selection without those cells."
            )
        return np.asarray([str(value) for value in values], dtype=object)

    @validation_scoped
    def _abundance_test_categories(
        self,
        categories: ArtifactRef | CellField,
        *,
        study_design: StudyDesign | None = None,
        sample_by: str | None = None,
        condition_by: str | None = None,
        cell_selection: ArtifactRef | None = None,
        groups: Sequence[Any] | None = None,
        transform: Literal["logit", "asin"] = "logit",
        robust: bool = True,
        trend: bool = False,
        skip_save: bool = False,
        invalidate_cache: bool = False,
    ) -> CategoryTestResult:
        """Test whether category proportions differ between conditions (propeller).

        Cells are counted per sample and category. Proportions are
        logit-transformed (or arcsine square root) and each category is
        tested with limma's moderated statistics: a t-test for two conditions
        and an F-test (ANOVA) for three or more. Samples are the replicates,
        so every sample must belong to one condition and each condition needs
        samples. The method and its defaults follow ``speckle::propeller``.

        Results are persisted as immutable ``category_tests`` artifacts unless
        ``skip_save`` is ``True``. Pass ``result.artifact`` to
        ``abundance.load_category_tests`` for exact retrieval.

        Args:
            categories: Exact categorical artifact, such as a clustering, or a
                cell metadata field.
            study_design: Study design supplying ``sample_by`` and
                ``condition_by``.
            sample_by: Cell metadata column identifying biological samples.
            condition_by: Cell metadata column with each sample's condition.
            cell_selection: Optional frozen selection for a metadata field, or
                a subset of an artifact's stored selection.
            groups: Keep and order only these conditions. With two
                conditions the contrast is the first minus the second; by
                default conditions are sorted.
            transform: ``"logit"`` (default) or ``"asin"``.
            robust: Robust empirical Bayes, turned off with two or fewer
                categories.
            trend: Let the prior variance trend with the mean transformed
                proportion.
            skip_save: Return the result without writing it.
            invalidate_cache: Recompute even when a matching artifact exists.

        Returns:
            A :class:`~cytearc.abundance.results.CategoryTestResult`.

        Raises:
            PermissionError: If a result must be written to a store that is
                not opened with ``zarr_mode='r+'``.
            ValueError: If labels are missing, a sample spans conditions, or
                fewer than two conditions remain.
        """
        sample_column, condition_column = _resolve_design_columns(
            study_design, sample_by, condition_by
        )
        if transform not in ("logit", "asin"):
            raise ValueError("transform must be 'logit' or 'asin'")
        resolved = resolve_grouping(
            self.zw, self.cells, categories, cell_selection=cell_selection
        )
        cell_idx = np.asarray(resolved.cell_idx, dtype=np.int64)
        labels = np.asarray(resolved.labels)
        if not valid_category_mask(labels, missing_mask=resolved.missing_mask).all():
            raise ValueError(
                "categories has missing labels in the tested cells. Pass a "
                "cell_selection without those cells."
            )
        samples = self._read_complete_labels(sample_column, cell_idx, role="sample")
        conditions = self._read_complete_labels(
            condition_column, cell_idx, role="condition"
        )
        group_order: tuple[str, ...] | None = None
        if groups is not None:
            group_order = tuple(str(group) for group in groups)
            if len(set(group_order)) != len(group_order):
                raise ValueError("groups must not repeat a condition")
            keep = np.isin(conditions, np.asarray(group_order, dtype=object))
            cell_idx = cell_idx[keep]
            labels = labels[keep]
            samples = samples[keep]
            conditions = conditions[keep]
        if cell_idx.size == 0:
            raise ValueError("No cells remain to test")
        condition_of = sample_conditions(samples, conditions)
        category_order = _sorted_labels(labels)
        counts, category_names, sample_names = count_categories(
            labels, samples, category_order=category_order
        )
        sample_groups = [condition_of[sample] for sample in sample_names]
        category_input = categories if isinstance(categories, ArtifactRef) else None
        category_field = categories.key if isinstance(categories, CellField) else None
        arguments = CategoryTestArguments(
            categories=category_input,
            cell_selection=resolved.cell_selection,
            cell_selection_fingerprint=value_fingerprint(cell_idx),
            category_fingerprint=value_fingerprint(
                np.asarray([str(value) for value in labels], dtype=object)
            ),
            sample_fingerprint=value_fingerprint(samples),
            condition_fingerprint=value_fingerprint(conditions),
            category_field=category_field,
            sample_by=sample_column,
            condition_by=condition_column,
            groups=group_order,
            transform=transform,
            robust=bool(robust),
            trend=bool(trend),
            n_cells=int(cell_idx.size),
            invalidate_cache=invalidate_cache,
        )
        planned: Any = None
        if not skip_save:
            planned = arguments.plan(
                self.zw,
                scope="datastore",
                invalidate_cache=invalidate_cache,
                required_arrays=("counts", "stats", "df_prior", "s2_prior"),
            )
            if planned.reused:
                logger.info("Reused category proportion tests")
                return read_category_slot(
                    artifact_group(self.zw, planned.ref), artifact=planned.ref
                )
            self._require_writable("test_categories")
        fit = propeller(
            counts,
            categories=category_names,
            samples=sample_names,
            sample_groups=sample_groups,
            groups=group_order,
            transform=transform,
            robust=robust,
            trend=trend,
        )
        result = CategoryTestResult(
            fit=fit,
            categories=category_input,
            category_field=(
                CellField(category_field, kind="categorical")
                if category_field is not None
                else None
            ),
            cell_selection=resolved.cell_selection,
            sample_by=sample_column,
            condition_by=condition_column,
            robust_requested=bool(robust),
            n_cells=int(cell_idx.size),
            artifact=planned.ref if planned is not None else None,
        )
        if planned is not None:
            with artifact_transaction(self.zw, planned) as slot:
                write_category_slot(slot, result, arguments=arguments)
            logger.info(
                f"Stored category proportion tests for {len(category_names)} "
                f"categories over {len(sample_names)} samples"
            )
        return result

    @validation_scoped
    def _abundance_load_category_tests(
        self,
        artifact: ArtifactRef,
    ) -> CategoryTestResult:
        """Return one exact immutable category proportion test."""
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")
        if artifact.kind != "category_tests":
            raise ValueError("artifact must reference category_tests")
        status = inspect_artifact(self.zw, artifact)
        if not status.exists:
            raise KeyError(f"Category test artifact does not exist: {status.path}")
        if not status.complete:
            raise RuntimeError(f"Category test artifact is incomplete: {status.path}")
        if (status.provenance or {}).get("operation") != "test_categories":
            raise ValueError("artifact was not produced by test_categories")
        slot_group = as_zarr_group(self.zw[status.path], name=status.path)
        return read_category_slot(slot_group, artifact=artifact)

    # Milo -----------------------------------------------------------------

    def _abundance_complete_slot(
        self, artifact: ArtifactRef, *, kind: str, operation: str
    ) -> Any:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError(f"artifact must be an ArtifactRef to {kind}")
        if artifact.kind != kind:
            raise ValueError(f"artifact must reference {kind}")
        status = inspect_artifact(self.zw, artifact)
        if not status.exists:
            raise KeyError(f"{kind} artifact does not exist: {status.path}")
        if not status.complete:
            raise RuntimeError(f"{kind} artifact is incomplete: {status.path}")
        if (status.provenance or {}).get("operation") != operation:
            raise ValueError(f"artifact was not produced by {operation}")
        return as_zarr_group(self.zw[status.path], name=status.path)

    def _abundance_graph_cells(self, cell_selection: ArtifactRef) -> np.ndarray:
        return read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)

    def _abundance_codes(
        self, cell_idx: np.ndarray, column: str, *, role: str
    ) -> tuple[np.ndarray, list[str]]:
        """Code a cell column by first appearance, reading it in row blocks."""
        if column not in self.cells.columns:
            raise KeyError(f"Cell metadata column {column!r} was not found")
        codes = np.empty(cell_idx.size, dtype=np.int32)
        order: dict[str, int] = {}
        for start in range(0, cell_idx.size, _MILO_BLOCK_ROWS):
            block = cell_idx[start : start + _MILO_BLOCK_ROWS]
            labels = self._read_complete_labels(column, block, role=role)
            for offset, label in enumerate(labels.tolist()):
                codes[start + offset] = order.setdefault(label, len(order))
        return codes, list(order)

    def _abundance_sample_values(
        self,
        cell_idx: np.ndarray,
        sample_codes: np.ndarray,
        n_samples: int,
        column: str,
    ) -> np.ndarray:
        """Return one value per sample of a column that is constant within samples."""
        if column not in self.cells.columns:
            raise KeyError(f"Cell metadata column {column!r} was not found")
        values: list[Any] = [None] * n_samples
        seen = np.zeros(n_samples, dtype=bool)
        for start in range(0, cell_idx.size, _MILO_BLOCK_ROWS):
            block = cell_idx[start : start + _MILO_BLOCK_ROWS]
            raw = np.asarray(read_metadata_rows(self.cells, column, block))
            missing = read_metadata_missing_rows(self.cells, column, block)
            if raw.dtype.kind not in "iuf" or missing is not None:
                if not valid_category_mask(raw, missing_mask=missing).all():
                    raise ValueError(
                        f"The design column {column!r} has missing values in the "
                        "graph's cells"
                    )
            elif not np.all(np.isfinite(raw)):
                raise ValueError(f"The design column {column!r} has non-finite values")
            codes = sample_codes[start : start + _MILO_BLOCK_ROWS]
            for code, value in zip(codes.tolist(), raw.tolist(), strict=True):
                if not seen[code]:
                    values[code] = value
                    seen[code] = True
                elif values[code] != value:
                    raise ValueError(
                        f"The design column {column!r} varies within a sample. "
                        "Design variables must be constant within each sample."
                    )
        return np.asarray(values)

    @validation_scoped
    def _abundance_make_neighborhoods(
        self,
        neighbors: ArtifactRef,
        *,
        prop: float = 0.1,
        refined: bool = True,
        seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Sample Milo neighbourhoods on a stored kNN graph (``makeNhoods``).

        ``floor(prop * n_cells)`` cells are drawn with ``seed``. With
        ``refined=True`` each drawn cell moves to the cell nearest the median
        position of its k nearest neighbours, found with the graph's ANN
        index, and duplicates are dropped. A neighbourhood is the index cell,
        its k nearest neighbours, and every cell that lists the index cell
        among its own k nearest neighbours, as on miloR's undirected graph.
        The graph is read in row blocks; membership is stored cell-major
        with the Euclidean distance from each index cell to its k-th
        neighbour, which spatial FDR uses.

        Args:
            neighbors: Neighbors artifact built with ``ann_metric="l2"``.
            prop: Fraction of cells drawn as candidate index cells.
            refined: Refine the drawn cells, as miloR does by default.
            seed: Seed of the draw.
            invalidate_cache: Recompute even when a matching artifact exists.

        Returns:
            Reference to the ``neighborhoods`` artifact.

        Raises:
            ValueError: If the graph is not Euclidean or ``prop`` is invalid.
            PermissionError: If the store is read-only and nothing matches.
        """
        if not isinstance(neighbors, ArtifactRef):
            raise TypeError("neighbors must be an ArtifactRef")
        if neighbors.kind != "neighbors":
            raise ValueError("neighbors must reference a neighbors artifact")
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise TypeError("seed must be an integer")
        if not 0 < float(prop) <= 1:
            raise ValueError("prop must be in (0, 1]")
        status = inspect_artifact(self.zw, neighbors)
        if not status.complete:
            raise ValueError("neighbors artifact is not complete")
        inputs = resolve_native_graph_inputs(self.zw, neighbors)
        validate_distance_provenance(self.zw, neighbors)
        if (status.parameters or {}).get("distance_metric") != "l2":
            raise ValueError(
                "Milo neighbourhoods need a Euclidean graph, as miloR builds. "
                "Build the ANN index with ann_metric='l2'."
            )
        assert neighbors.assay is not None
        indices = as_zarr_array(
            artifact_group(self.zw, neighbors)["indices"], name="indices"
        )
        n_cells, k = (int(value) for value in indices.shape)
        arguments = NeighborhoodArguments(
            neighbors=neighbors,
            prop=float(prop),
            refined=bool(refined),
            seed=int(seed),
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=neighbors.assay,
            invalidate_cache=invalidate_cache,
            required_arrays=NEIGHBORHOOD_ARRAYS,
        )
        if planned.reused:
            logger.info("Reused Milo neighbourhoods")
            return planned.ref
        self._require_writable("make_neighborhoods")
        coordinates = as_zarr_array(
            artifact_group(self.zw, inputs.coordinates)["data"], name="data"
        )
        sampled = sample_index_cells(n_cells, float(prop), seed=int(seed))
        if refined:
            ann_parameters = (
                inspect_artifact(self.zw, inputs.ann_index).parameters or {}
            )
            index = self._resolve_ann_index(
                inputs.ann_index,
                "l2",
                int(coordinates.shape[1]),
                expected_count=n_cells,
            )
            index = AnnIndexStage.configure(
                index, ef=int(ann_parameters.get("ann_ef", 50)), threads=1
            )

            def nearest(points: np.ndarray) -> np.ndarray:
                labels, _ = index.knn_query(np.asarray(points, dtype=np.float32), k=1)
                return np.asarray(labels[:, 0], dtype=np.int64)

            index_rows = refine_index_cells(
                sampled,
                _read_rows(indices, sampled),
                lambda cells: _read_rows(coordinates, cells),
                nearest,
            )
            del index
        else:
            index_rows = unique_in_order(sampled)
        n_nhoods = int(index_rows.size)
        index_neighbors = _read_rows(indices, index_rows).astype(np.int64)
        kth = np.empty(n_nhoods)
        for start in range(0, n_nhoods, _MILO_BLOCK_ROWS // k + 1):
            stop = min(start + _MILO_BLOCK_ROWS // k + 1, n_nhoods)
            rows = index_neighbors[start:stop]
            kth[start:stop] = kth_distances(
                _read_rows(coordinates, index_rows[start:stop]),
                _read_rows(coordinates, rows.ravel()).reshape(*rows.shape, -1),
            )
        reverse = reverse_neighbors(index_rows, index_neighbors)
        del index_neighbors
        position = np.full(n_cells, -1, dtype=np.int32)
        position[index_rows] = np.arange(n_nhoods, dtype=np.int32)
        sizes = np.zeros(n_nhoods, dtype=np.int64)
        graph_cells = self._abundance_graph_cells(inputs.cell_selection)
        with artifact_transaction(self.zw, planned) as slot:
            profile = resolve_storage_profile(slot.store)
            zarr_format = _group_zarr_format(slot)
            indptr = create_numeric_array(
                slot,
                "membership_indptr",
                row_sharded_array_spec(
                    (n_cells + 1,),
                    np.int64,
                    profile=profile,
                    band_rows=min(n_cells + 1, 1 << 20),
                    zarr_format=zarr_format,
                ),
            )
            total = 0
            indptr[0:1] = np.zeros(1, dtype=np.int64)
            for start in range(0, n_cells, _MILO_BLOCK_ROWS):
                stop = min(start + _MILO_BLOCK_ROWS, n_cells)
                row_counts, ids = block_membership(
                    start, np.asarray(indices[start:stop]), position, reverse, n_nhoods
                )
                sizes += np.bincount(ids, minlength=n_nhoods)
                indptr[start + 1 : stop + 1] = total + np.cumsum(row_counts)
                total += int(row_counts.sum())
            members = create_numeric_array(
                slot,
                "membership_indices",
                row_sharded_array_spec(
                    (total,),
                    np.int32,
                    profile=profile,
                    band_rows=max(1, min(total, 1 << 22)),
                    zarr_format=zarr_format,
                ),
            )
            for start in range(0, n_cells, _MILO_BLOCK_ROWS):
                stop = min(start + _MILO_BLOCK_ROWS, n_cells)
                _, ids = block_membership(
                    start, np.asarray(indices[start:stop]), position, reverse, n_nhoods
                )
                offset = int(np.asarray(indptr[start : start + 1])[0])
                members[offset : offset + ids.size] = ids
            write_small_array(slot, "index_rows", index_rows.astype(np.int64))
            write_small_array(slot, "index_cells", graph_cells[index_rows])
            write_small_array(slot, "sizes", sizes)
            write_small_array(slot, "kth_distance", kth)
            slot.attrs.update(
                {
                    "neighbors": neighbors.to_dict(),
                    "cell_selection": inputs.cell_selection.to_dict(),
                    "k": k,
                    "n_cells": n_cells,
                    "prop": float(prop),
                    "refined": bool(refined),
                    "seed": int(seed),
                    "n_sampled": int(sampled.size),
                }
            )
        logger.info(f"Stored {n_nhoods} Milo neighbourhoods over {n_cells} cells")
        return planned.ref

    @validation_scoped
    def _abundance_load_neighborhoods(self, artifact: ArtifactRef) -> Neighborhoods:
        """Return one exact immutable set of Milo neighbourhoods."""
        slot = self._abundance_complete_slot(
            artifact, kind="neighborhoods", operation="make_neighborhoods"
        )
        return read_neighborhoods_slot(slot, artifact=artifact)

    def _abundance_iter_membership(
        self, artifact: ArtifactRef
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield cell blocks of a neighbourhood membership: start, counts, ids."""
        slot = artifact_group(self.zw, artifact)
        indptr = as_zarr_array(slot["membership_indptr"], name="membership_indptr")
        members = as_zarr_array(slot["membership_indices"], name="membership_indices")
        n_cells = int(indptr.shape[0]) - 1
        for start in range(0, n_cells, _MILO_BLOCK_ROWS):
            stop = min(start + _MILO_BLOCK_ROWS, n_cells)
            offsets = np.asarray(indptr[start : stop + 1])
            ids = np.asarray(members[int(offsets[0]) : int(offsets[-1])])
            yield start, np.diff(offsets), ids

    @validation_scoped
    def _abundance_count_neighborhoods(
        self,
        neighborhoods: ArtifactRef,
        *,
        sample_by: str,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Count each neighbourhood's cells per sample (``countCells``).

        Samples are ordered by their first appearance along the graph's
        cells. Every graph cell needs a sample label.

        Args:
            neighborhoods: Neighbourhoods artifact.
            sample_by: Cell metadata column identifying biological samples.
            invalidate_cache: Recompute even when a matching artifact exists.

        Returns:
            Reference to the ``neighborhood_counts`` artifact.
        """
        nhoods = self._abundance_load_neighborhoods(neighborhoods)
        cell_idx = self._abundance_graph_cells(nhoods.cell_selection)
        codes, samples = self._abundance_codes(cell_idx, sample_by, role="sample")
        arguments = NeighborhoodCountArguments(
            neighborhoods=neighborhoods,
            sample_fingerprint=value_fingerprint(
                np.asarray(samples, dtype=object)[codes]
            ),
            sample_by=sample_by,
            invalidate_cache=invalidate_cache,
        )
        assert neighborhoods.assay is not None
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=neighborhoods.assay,
            invalidate_cache=invalidate_cache,
            required_arrays=("counts",),
        )
        if planned.reused:
            logger.info("Reused Milo neighbourhood counts")
            return planned.ref
        self._require_writable("count_neighborhoods")
        counts = np.zeros((nhoods.n_neighborhoods, len(samples)), dtype=np.int64)
        for start, row_counts, ids in self._abundance_iter_membership(neighborhoods):
            count_block(counts, row_counts, ids, codes[start : start + row_counts.size])
        with artifact_transaction(self.zw, planned) as slot:
            write_small_array(slot, "counts", counts)
            slot.attrs.update(
                {
                    "samples": samples,
                    "sample_by": sample_by,
                    "neighborhoods": neighborhoods.to_dict(),
                }
            )
        logger.info(
            f"Stored counts of {nhoods.n_neighborhoods} neighbourhoods over "
            f"{len(samples)} samples"
        )
        return planned.ref

    @validation_scoped
    def _abundance_load_neighborhood_counts(
        self, artifact: ArtifactRef
    ) -> NeighborhoodCounts:
        """Return one exact immutable table of neighbourhood counts."""
        slot = self._abundance_complete_slot(
            artifact, kind="neighborhood_counts", operation="count_neighborhoods"
        )
        return read_neighborhood_counts_slot(slot, artifact=artifact)

    @validation_scoped
    def _abundance_test_neighborhoods(
        self,
        counts: ArtifactRef,
        *,
        condition_by: str,
        covariates: Sequence[str] = (),
        levels: Mapping[str, Sequence[Any]] | None = None,
        contrast: Mapping[str, float] | None = None,
        norm_method: Literal["TMM", "RLE", "logMS"] = "TMM",
        min_mean: float = 0.0,
        robust: bool = True,
        legacy: bool = True,
        fdr_weighting: Literal["k-distance", "none"] = "k-distance",
        skip_save: bool = False,
        invalidate_cache: bool = False,
    ) -> NeighborhoodTestResult:
        """Test neighbourhoods for differential abundance (``testNhoods``).

        The design is ``~ covariates + condition_by`` with treatment coding,
        built from sample-level values that must be constant within each
        sample. Counts are normalized (TMM by default), NB dispersions are
        estimated with ``estimateDisp``, and each neighbourhood is tested with
        edgeR's quasi-likelihood F-test, using the legacy method that miloR
        2.6.0 requests unless ``legacy=False``. The edgeR steps run in
        edgePython, installed with the ``milo`` extra. Without ``contrast`` the last
        design column is tested, as miloR does. P-values are corrected with
        Benjamini-Hochberg (``fdr``) and with miloR's spatial FDR, weighted by
        each neighbourhood's k-distance (``spatial_fdr``).

        miloR computes spatial FDR from every neighbourhood even when
        ``min.mean`` leaves some untested, which misaligns its weights; here
        it uses the tested neighbourhoods only.

        Args:
            counts: Neighbourhood counts artifact.
            condition_by: Sample-level condition column; it is the last term.
            covariates: Sample-level columns placed before the condition.
            levels: Level order of categorical design columns; the first
                level is the reference. Others are sorted.
            contrast: Weights over design columns, for example
                ``{"conditiontreated": 1}``; the last column by default.
            norm_method: ``"TMM"``, ``"RLE"``, or ``"logMS"``.
            min_mean: Test neighbourhoods whose mean count is at least this.
            robust: Robust quasi-likelihood moderation.
            legacy: edgeR's legacy quasi-likelihood method.
            fdr_weighting: ``"k-distance"`` or ``"none"``.
            skip_save: Return the result without writing it.
            invalidate_cache: Recompute even when a matching artifact exists.

        Returns:
            A :class:`~cytearc.abundance.results.NeighborhoodTestResult`.

        Raises:
            ImportError: If edgePython is not installed.
            ValueError: If a design value varies within a sample or the design
                is not of full rank.
        """
        if fdr_weighting not in ("k-distance", "none"):
            raise ValueError("fdr_weighting must be 'k-distance' or 'none'")
        if norm_method not in ("TMM", "RLE", "logMS"):
            raise ValueError("norm_method must be 'TMM', 'RLE', or 'logMS'")
        covariate_columns = tuple(str(column) for column in covariates)
        if condition_by in covariate_columns:
            raise ValueError("condition_by must not also be a covariate")
        count_table = self._abundance_load_neighborhood_counts(counts)
        nhoods = self._abundance_load_neighborhoods(count_table.neighborhoods)
        cell_idx = self._abundance_graph_cells(nhoods.cell_selection)
        sample_codes, samples = self._abundance_codes(
            cell_idx, count_table.sample_by, role="sample"
        )
        if tuple(samples) != count_table.samples:
            raise ValueError(
                f"The samples of {count_table.sample_by!r} changed since the counts "
                "were made; rerun abundance.count_neighborhoods"
            )
        variables = {
            column: self._abundance_sample_values(
                cell_idx, sample_codes, len(samples), column
            )
            for column in (*covariate_columns, condition_by)
        }
        level_order = (
            None
            if levels is None
            else {
                key: tuple(str(value) for value in order)
                for key, order in levels.items()
            }
        )
        design = treatment_design(variables, levels=level_order)
        weights = (
            {design.columns[-1]: 1.0}
            if contrast is None
            else {str(key): float(value) for key, value in contrast.items()}
        )
        vector = contrast_vector(design, weights)
        arguments = NeighborhoodTestArguments(
            counts=counts,
            design_fingerprint=value_fingerprint(design.matrix),
            condition_by=condition_by,
            covariates=covariate_columns,
            levels=level_order,
            contrast=None if contrast is None else weights,
            norm_method=norm_method,
            min_mean=float(min_mean),
            robust=bool(robust),
            legacy=bool(legacy),
            fdr_weighting=fdr_weighting,
            invalidate_cache=invalidate_cache,
        )
        planned: Any = None
        if not skip_save:
            assert counts.assay is not None
            planned = arguments.plan(
                self.zw,
                scope="assay",
                assay=counts.assay,
                invalidate_cache=invalidate_cache,
                required_arrays=("stats", "tested", "design"),
            )
            if planned.reused:
                logger.info("Reused Milo neighbourhood tests")
                return read_neighborhood_test_slot(
                    artifact_group(self.zw, planned.ref), artifact=planned.ref
                )
            self._require_writable("test_neighborhoods")
        fit = differential_abundance(
            count_table.counts,
            design.matrix,
            vector,
            kth_distance=nhoods.kth_distance if fdr_weighting == "k-distance" else None,
            norm_method=norm_method,
            min_mean=float(min_mean),
            robust=bool(robust),
            legacy=bool(legacy),
        )
        sample_index = pd.Index(samples, name=count_table.sample_by)
        result = NeighborhoodTestResult(
            counts=counts,
            neighborhoods=count_table.neighborhoods,
            sample_by=count_table.sample_by,
            condition_by=condition_by,
            covariates=covariate_columns,
            design=pd.DataFrame(
                design.matrix, index=sample_index, columns=list(design.columns)
            ),
            contrast=weights,
            table=pd.DataFrame(
                {
                    "index_cell": nhoods.index_cells,
                    "size": nhoods.sizes,
                    "log_fc": fit.log_fc,
                    "log_cpm": fit.log_cpm,
                    "f_statistic": fit.f_statistic,
                    "p_value": fit.p_value,
                    "fdr": fit.fdr,
                    "spatial_fdr": fit.spatial_fdr,
                    "tested": fit.tested,
                },
                index=pd.RangeIndex(nhoods.n_neighborhoods, name="neighborhood"),
            ),
            norm_factors=pd.Series(
                fit.norm_factors, index=sample_index, name="norm_factor"
            ),
            lib_size=pd.Series(fit.lib_size, index=sample_index, name="lib_size"),
            common_dispersion=fit.common_dispersion,
            average_ql_dispersion=fit.average_ql_dispersion,
            norm_method=norm_method,
            min_mean=float(min_mean),
            robust=bool(robust),
            legacy=bool(legacy),
            fdr_weighting=fdr_weighting,
            artifact=planned.ref if planned is not None else None,
        )
        if planned is not None:
            with artifact_transaction(self.zw, planned) as slot:
                write_neighborhood_test_slot(slot, result, tested=fit.tested)
            logger.info(f"Stored Milo tests of {int(fit.tested.sum())} neighbourhoods")
        return result

    @validation_scoped
    def _abundance_load_neighborhood_tests(
        self, artifact: ArtifactRef
    ) -> NeighborhoodTestResult:
        """Return one exact immutable Milo test."""
        slot = self._abundance_complete_slot(
            artifact, kind="neighborhood_tests", operation="test_neighborhoods"
        )
        return read_neighborhood_test_slot(slot, artifact=artifact)

    @validation_scoped
    def _abundance_annotate_neighborhoods(
        self,
        neighborhoods: ArtifactRef,
        labels: ArtifactRef | CellField,
    ) -> pd.DataFrame:
        """Return each neighbourhood's majority label (``annotateNhoods``).

        Args:
            neighborhoods: Neighbourhoods or neighbourhood tests artifact.
            labels: Categorical artifact or cell metadata field over the
                graph's cells. Cells with missing labels are left out.

        Returns:
            One row per neighbourhood with ``label``, its ``fraction`` of the
            neighbourhood's labelled cells, and the cell count of each label.
        """
        if (
            isinstance(neighborhoods, ArtifactRef)
            and neighborhoods.kind == "neighborhood_tests"
        ):
            neighborhoods = self._abundance_load_neighborhood_tests(
                neighborhoods
            ).neighborhoods
        nhoods = self._abundance_load_neighborhoods(neighborhoods)
        resolved = resolve_grouping(
            self.zw, self.cells, labels, cell_selection=nhoods.cell_selection
        )
        values = np.asarray(resolved.labels)
        if values.size != nhoods.n_cells:
            raise ValueError("labels do not cover the neighbourhoods' graph cells")
        valid = valid_category_mask(values, missing_mask=resolved.missing_mask)
        text = np.asarray([str(value) for value in values], dtype=object)
        levels = tuple(str(value) for value in unique_in_order(text[valid]))
        lookup = {level: idx for idx, level in enumerate(levels)}
        codes = np.fromiter(
            (
                lookup[label] if ok else -1
                for label, ok in zip(text, valid, strict=True)
            ),
            dtype=np.int64,
            count=text.size,
        )
        label_counts = np.zeros((nhoods.n_neighborhoods, len(levels)), dtype=np.int64)
        for start, row_counts, ids in self._abundance_iter_membership(neighborhoods):
            count_block(
                label_counts, row_counts, ids, codes[start : start + row_counts.size]
            )
        annotation = annotate_neighborhoods(label_counts, levels)
        frame = pd.DataFrame(
            label_counts,
            index=pd.RangeIndex(nhoods.n_neighborhoods, name="neighborhood"),
            columns=list(levels),
        )
        frame.insert(0, "fraction", annotation.fraction)
        frame.insert(0, "label", annotation.label)
        return frame
