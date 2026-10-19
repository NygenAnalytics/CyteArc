"""``DataStore.abundance``: propeller and Milo on a store, kept as artifacts.

propeller (Phipson et al. 2022) tests the proportions of annotated cell
categories between conditions. Milo (Dann et al. 2022) builds overlapping
neighbourhoods on a stored kNN graph, counts their cells per sample, and
tests the counts with a spatial FDR (Lun, Richard and Marioni 2017). The
statistics live in :mod:`cytearc.abundance`; this module validates the
design, streams the stored graph in row bands, and keeps every result as an
immutable artifact.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
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
from ...graph.feature_projection import resolve_coordinate_inputs
from ...metadata.arguments import (
    CategoryTestArguments,
    NeighborhoodArguments,
    NeighborhoodCountArguments,
    NeighborhoodTestArguments,
)
from ...metadata.rows import (
    metadata_column_fingerprint,
    read_array_rows_chunkwise,
    read_metadata_missing_rows,
    read_metadata_rows,
)
from ...metadata.selection import (
    CellField,
    StudyDesign,
    resolve_grouping,
    valid_category_mask,
)
from ...neighbors.stages import AnnIndexStage
from ...stats.design import contrast_vector, factor_labels, treatment_design
from ...storage.artifact_writer import AttributeRequirement, artifact_transaction
from ...storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    artifact_group,
    fingerprint_array,
    inspect_artifact,
    provenance_hash,
)
from ...storage.geometry import array_geometry
from ...storage.partition import scan_band
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import float_argument, integer_argument
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

_CATEGORY_ARRAYS = ("counts", "stats", "df_prior", "s2_prior")
_TEST_ARRAYS = (
    "stats",
    "tested",
    "index_cells",
    "sizes",
    "design",
    "norm_factors",
    "lib_size",
)
# Most graph rows per band. Membership intermediates hold several int64
# copies of a band's neighbour lists, so whole 64 MiB scan bands are split.
_MAX_BAND_ROWS = 65_536
# Index cells whose neighbour coordinates are held at once.
_INDEX_CELL_BLOCK = 4_096
# Elements per chunk of the membership ids, whose length is known only at
# the end of the pass that writes them.
_MEMBERSHIP_CHUNK = 1 << 20


def _band_rows(array: Any) -> int:
    return min(
        scan_band(array_geometry(array), fallback=_MAX_BAND_ROWS), _MAX_BAND_ROWS
    )


def _distinct_rows(array: Any, rows: np.ndarray) -> np.ndarray:
    """Read possibly repeated rows of a stored array, one chunk at a time."""
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return np.asarray(read_array_rows_chunkwise(array, unique))[inverse]


def _single_condition_samples(samples: np.ndarray, conditions: np.ndarray) -> None:
    """Reject samples whose cells carry more than one condition."""
    pairs = pd.DataFrame({"sample": samples, "condition": conditions})
    spread = pairs.groupby("sample", sort=True)["condition"].nunique()
    mixed = spread.index[spread > 1].tolist()
    if mixed:
        raise ValueError(
            "Every sample must belong to one condition; "
            f"{len(mixed)} samples span several, for example {mixed[:5]}. Leave "
            "out the cells of a sample's other conditions with cell_selection."
        )


class _AbundanceOperationsMixin(_AbundanceOperationsBase):
    def _abundance_artifact(
        self, artifact: ArtifactRef, kind: str, operation: str
    ) -> tuple[ArtifactStatus, zarr.Group]:
        """Return the status and group of a complete abundance artifact."""
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")
        if artifact.kind != kind:
            raise ValueError(f"artifact must reference {kind}")
        status = inspect_artifact(self.zw, artifact)
        if not status.exists:
            raise KeyError(f"Artifact does not exist: {status.path}")
        if not status.complete:
            raise RuntimeError(f"Artifact is incomplete: {status.path}")
        if status.operation != operation:
            raise ValueError(f"artifact was not produced by {operation}")
        return status, as_zarr_group(self.zw[status.path], name=status.path)

    def _abundance_cell_rows(self, cell_selection: ArtifactRef) -> np.ndarray:
        return read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)

    def _abundance_column(
        self, column: str, rows: np.ndarray, *, role: str
    ) -> np.ndarray:
        """Read a label column over ``rows``, refusing missing values."""
        if column not in self.cells.columns:
            raise KeyError(f"{role} column {column!r} is not in the cell metadata")
        values = np.asarray(read_metadata_rows(self.cells, column, rows))
        missing = read_metadata_missing_rows(self.cells, column, rows)
        valid = valid_category_mask(values, missing_mask=missing)
        if not valid.all():
            raise ValueError(
                f"{role} column {column!r} has missing values in "
                f"{int((~valid).sum())} cells; leave those cells out with a cell "
                "selection"
            )
        return values

    def _abundance_iter_membership(
        self, group: zarr.Group
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield each row band of a neighbourhood membership.

        A band is its first graph cell, the number of neighbourhoods of each
        of its cells, and their ids.
        """
        indptr = as_zarr_array(group["membership_indptr"], name="membership_indptr")
        indices = as_zarr_array(group["membership_indices"], name="membership_indices")
        n_cells = int(indptr.shape[0]) - 1
        band = _band_rows(indptr)
        for start in range(0, n_cells, band):
            stop = min(start + band, n_cells)
            offsets = np.asarray(indptr[start : stop + 1], dtype=np.int64)
            ids = np.asarray(indices[int(offsets[0]) : int(offsets[-1])])
            yield start, np.diff(offsets), ids

    def _abundance_neighborhoods_ref(self, artifact: ArtifactRef) -> ArtifactRef:
        """Return the neighbourhoods that an abundance artifact derives from."""
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("neighborhoods must be an ArtifactRef")
        if artifact.kind == "neighborhood_tests":
            status, _ = self._abundance_artifact(
                artifact, "neighborhood_tests", "test_neighborhoods"
            )
            artifact = status.input_ref("counts")
        if artifact.kind == "neighborhood_counts":
            status, _ = self._abundance_artifact(
                artifact, "neighborhood_counts", "count_neighborhoods"
            )
            artifact = status.input_ref("neighborhoods")
        if artifact.kind != "neighborhoods":
            raise ValueError(
                "neighborhoods must reference neighborhoods, neighborhood_counts, "
                "or neighborhood_tests"
            )
        return artifact

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

        Cells are counted per category and sample, each sample's proportions
        are transformed (``"logit"`` or ``"asin"``), and each category is
        tested with an empirical Bayes moderated t-test for two conditions
        or F-test for more (Phipson et al. 2022). Samples are the
        replicates, so every sample must belong to one condition. Categories,
        samples, and conditions are ordered as R's ``factor`` orders them;
        ``groups`` keeps only those conditions, in that order, and with two
        the t-statistic is the first minus the second.

        The result is saved as an immutable ``category_tests`` artifact
        unless ``skip_save`` is True. Writing requires ``zarr_mode='r+'``; a
        matching saved result is reused from a read-only store.

        Args:
            categories: Categorical cell labels: a cluster or label artifact,
                or a cell-metadata field.
            study_design: Supplies ``sample_by`` and ``condition_by``.
            sample_by: Cell-metadata column of the samples.
            condition_by: Cell-metadata column of the conditions.
            cell_selection: Cells to count. Defaults to every cell of a
                metadata field, or to the artifact's own cells.
            groups: Conditions to test, in order.
            transform: Variance-stabilizing transform of the proportions.
            robust: Robust empirical Bayes moderation.
            trend: Let the prior variance trend with the mean.
            skip_save: Return the result without writing it.
            invalidate_cache: Recompute even when a matching result exists.

        Raises:
            ValueError: If a design column is missing or conflicts with
                ``study_design``, a label is missing, a sample spans
                conditions, or fewer than two conditions remain.
            PermissionError: If the result must be written to a read-only
                store. The check runs before the test is computed.
        """
        if study_design is not None:
            if study_design.subject_by is not None or study_design.pair_by is not None:
                raise ValueError(
                    "propeller treats samples as independent; drop subject_by and "
                    "pair_by from study_design"
                )
            if sample_by is not None and sample_by != study_design.sample_by:
                raise ValueError("sample_by conflicts with study_design.sample_by")
            if (
                condition_by is not None
                and study_design.condition_by is not None
                and condition_by != study_design.condition_by
            ):
                raise ValueError(
                    "condition_by conflicts with study_design.condition_by"
                )
            sample_by = study_design.sample_by
            condition_by = condition_by or study_design.condition_by
        if sample_by is None:
            raise ValueError("sample_by is required, directly or through study_design")
        if condition_by is None:
            raise ValueError(
                "condition_by is required, directly or through study_design"
            )
        if transform not in ("logit", "asin"):
            raise ValueError("transform must be 'logit' or 'asin'")
        requested_groups = (
            None
            if groups is None
            else tuple(str(label) for label in factor_labels(list(groups))[0])
        )

        resolved = resolve_grouping(
            self.zw, self.cells, categories, cell_selection=cell_selection
        )
        cell_idx = np.asarray(resolved.cell_idx, dtype=np.int64)
        valid = valid_category_mask(resolved.labels, missing_mask=resolved.missing_mask)
        if not valid.all():
            raise ValueError(
                f"categories have missing values in {int((~valid).sum())} cells; "
                "leave those cells out with a cell selection"
            )
        category_labels, category_order = factor_labels(
            resolved.labels, name="categories"
        )
        sample_labels, sample_order = factor_labels(
            self._abundance_column(sample_by, cell_idx, role="Sample"),
            name="samples",
        )
        condition_labels, condition_order = factor_labels(
            self._abundance_column(condition_by, cell_idx, role="Condition"),
            name="conditions",
        )
        _single_condition_samples(sample_labels, condition_labels)

        category_ref = categories if isinstance(categories, ArtifactRef) else None
        category_field = (
            CellField(categories.key, kind="categorical")
            if isinstance(categories, CellField)
            else None
        )
        arguments = CategoryTestArguments(
            categories=category_ref,
            cell_selection=resolved.cell_selection,
            cell_selection_fingerprint=value_fingerprint(cell_idx),
            category_fingerprint=value_fingerprint(category_labels),
            sample_fingerprint=value_fingerprint(sample_labels),
            condition_fingerprint=value_fingerprint(condition_labels),
            category_field=None if category_field is None else category_field.key,
            sample_by=sample_by,
            condition_by=condition_by,
            groups=requested_groups,
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
                required_arrays=_CATEGORY_ARRAYS,
                required_attributes=(
                    AttributeRequirement("stat_columns", expected_types=(list,)),
                ),
            )
            if planned.reused:
                logger.info("Reused propeller category tests")
                return read_category_slot(
                    artifact_group(self.zw, planned.ref), artifact=planned.ref
                )
            self._require_writable("test_categories")

        counts, category_names, sample_names = count_categories(
            category_labels,
            sample_labels,
            category_order=category_order,
            sample_order=sample_order,
        )
        condition_of = dict(zip(sample_labels, condition_labels, strict=True))
        fit = propeller(
            counts,
            categories=category_names,
            samples=sample_names,
            sample_groups=[condition_of[sample] for sample in sample_names],
            groups=condition_order if requested_groups is None else requested_groups,
            transform=transform,
            robust=bool(robust),
            trend=bool(trend),
        )
        result = CategoryTestResult(
            fit=fit,
            categories=category_ref,
            category_field=category_field,
            cell_selection=resolved.cell_selection,
            sample_by=sample_by,
            condition_by=condition_by,
            robust_requested=bool(robust),
            n_cells=int(cell_idx.size),
            artifact=None if planned is None else planned.ref,
        )
        if planned is not None:
            with artifact_transaction(self.zw, planned) as group:
                write_category_slot(group, result, arguments=arguments)
            logger.info(
                f"Stored propeller tests of {len(category_names)} categories over "
                f"{len(fit.samples)} samples"
            )
        return result

    @validation_scoped
    def _abundance_load_category_tests(
        self, artifact: ArtifactRef
    ) -> CategoryTestResult:
        """Return one saved propeller result exactly as it was computed."""
        _, group = self._abundance_artifact(
            artifact, "category_tests", "test_categories"
        )
        return read_category_slot(group, artifact=artifact)

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
        """Build Milo neighbourhoods on a stored kNN graph.

        ``floor(prop * n)`` of the graph's ``n`` cells are drawn with
        ``seed``. With ``refined``, each is moved to the cell nearest the
        coordinate-wise median of its k nearest neighbours, found with the
        graph's ANN index, and repeats are dropped (Dann et al. 2022). A
        neighbourhood is its index cell's order-1 neighbourhood on the
        undirected graph: the index cell, its k nearest neighbours, and the
        cells that list it among theirs. Membership is built and stored in
        row bands of the graph.

        The graph must come from ``graph.neighbors`` on an
        ``ann_metric="l2"`` index, so that distances are Euclidean as in
        miloR. The random draw uses NumPy's generator and so differs from
        miloR's for the same seed.

        Args:
            neighbors: A neighbors artifact.
            prop: Share of cells drawn as candidate index cells, in (0, 1].
            refined: Move candidates to the cell nearest their neighbours'
                median.
            seed: Seed of the draw.
            invalidate_cache: Rebuild even when matching neighbourhoods exist.

        Returns:
            The ``neighborhoods`` artifact; load it with
            ``abundance.load_neighborhoods``.

        Raises:
            ValueError: If the graph is not Euclidean or ``prop`` draws no
                cell.
            PermissionError: If neighbourhoods must be written to a read-only
                store.
        """
        if not isinstance(neighbors, ArtifactRef):
            raise TypeError("neighbors must be an ArtifactRef")
        if neighbors.kind != "neighbors":
            raise ValueError("neighbors must reference a neighbors artifact")
        status = inspect_artifact(self.zw, neighbors)
        if not status.complete:
            raise ValueError("Neighbors artifact is unavailable or incomplete")
        metric = (status.parameters or {}).get("distance_metric")
        if status.operation != "query_neighbors" or metric != "l2":
            raise ValueError(
                "Milo needs Euclidean neighbors: build them with graph.neighbors "
                f"on an ann_metric='l2' index (this graph's metric is {metric!r})"
            )
        resolved_prop = float_argument(prop, "prop")
        if not 0 < resolved_prop <= 1:
            raise ValueError("prop must be in (0, 1]")
        resolved_seed = integer_argument(seed, "seed")
        arguments = NeighborhoodArguments(
            neighbors=neighbors,
            prop=resolved_prop,
            refined=bool(refined),
            seed=resolved_seed,
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

        coordinates = status.input_ref("coordinates")
        cell_selection = resolve_coordinate_inputs(self.zw, coordinates).cell_selection
        cell_idx = self._abundance_cell_rows(cell_selection)
        knn = as_zarr_array(
            artifact_group(self.zw, neighbors)["indices"], name="indices"
        )
        points = as_zarr_array(
            artifact_group(self.zw, coordinates)["data"], name="data"
        )
        n_cells, k = (int(extent) for extent in knn.shape)
        if cell_idx.size != n_cells or int(points.shape[0]) != n_cells:
            raise ValueError("Neighbors, coordinates, and cells do not align")

        sampled = sample_index_cells(n_cells, resolved_prop, seed=resolved_seed)
        if refined:
            index_rows = refine_index_cells(
                sampled,
                read_array_rows_chunkwise(knn, sampled).astype(np.int64),
                lambda cells: _distinct_rows(points, cells),
                self._abundance_nearest_cells(status, n_cells, int(points.shape[1])),
            )
        else:
            index_rows = unique_in_order(sampled).astype(np.int64)
        n_nhoods = int(index_rows.size)
        index_neighbors = read_array_rows_chunkwise(knn, index_rows).astype(np.int64)
        kth_distance = np.empty(n_nhoods, dtype=np.float64)
        for start in range(0, n_nhoods, _INDEX_CELL_BLOCK):
            stop = min(start + _INDEX_CELL_BLOCK, n_nhoods)
            block = index_neighbors[start:stop]
            kth_distance[start:stop] = kth_distances(
                _distinct_rows(points, index_rows[start:stop]),
                _distinct_rows(points, block.ravel()).reshape(*block.shape, -1),
            )
        reverse = reverse_neighbors(index_rows, index_neighbors)
        del index_neighbors
        position = np.full(n_cells, -1, dtype=np.int32)
        position[index_rows] = np.arange(n_nhoods, dtype=np.int32)

        sizes = np.zeros(n_nhoods, dtype=np.int64)
        band = _band_rows(knn)
        with artifact_transaction(self.zw, planned) as group:
            indptr = group.create_array(
                "membership_indptr",
                shape=(n_cells + 1,),
                chunks=(max(1, min(band, n_cells + 1)),),
                dtype=np.int64,
            )
            indices = group.create_array(
                "membership_indices",
                shape=(0,),
                chunks=(_MEMBERSHIP_CHUNK,),
                dtype=np.int32,
            )
            indptr[0:1] = np.zeros(1, dtype=np.int64)
            offset = 0
            for start in range(0, n_cells, band):
                stop = min(start + band, n_cells)
                row_counts, ids = block_membership(
                    start,
                    np.asarray(knn[start:stop], dtype=np.int64),
                    position,
                    reverse,
                    n_nhoods,
                )
                indptr[start + 1 : stop + 1] = offset + np.cumsum(row_counts)
                if ids.size:
                    indices.resize((offset + ids.size,))
                    indices[offset : offset + ids.size] = ids
                offset += int(ids.size)
                sizes += np.bincount(ids, minlength=n_nhoods)
            write_small_array(group, "index_rows", index_rows.astype(np.int64))
            write_small_array(group, "index_cells", cell_idx[index_rows])
            write_small_array(group, "sizes", sizes)
            write_small_array(group, "kth_distance", kth_distance)
            group.attrs.update(
                {
                    "neighbors": neighbors.to_dict(),
                    "cell_selection": cell_selection.to_dict(),
                    "k": k,
                    "n_cells": n_cells,
                    "prop": resolved_prop,
                    "refined": bool(refined),
                    "seed": resolved_seed,
                }
            )
        logger.info(
            f"Stored {n_nhoods} Milo neighbourhoods of {n_cells} cells "
            f"(median size {float(np.median(sizes)):g})"
        )
        return planned.ref

    def _abundance_nearest_cells(
        self, neighbors: ArtifactStatus, n_cells: int, dims: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return a search of the graph's ANN index for the nearest cell to points."""
        ann_ref = neighbors.input_ref("ann_index")
        ann_status = inspect_artifact(self.zw, ann_ref)
        if not ann_status.complete:
            raise ValueError("The graph's ANN index is unavailable or incomplete")
        ann_ef = (ann_status.parameters or {}).get("ann_ef")
        if isinstance(ann_ef, bool) or not isinstance(ann_ef, int) or ann_ef < 1:
            raise ValueError(
                "ANN artifact has no valid ann_ef search depth. Re-run graph.ann_index."
            )
        index = AnnIndexStage.configure(
            self._resolve_ann_index(ann_ref, "l2", dims, expected_count=n_cells),
            ef=ann_ef,
            threads=self.nthreads,
        )

        def nearest(medians: np.ndarray) -> np.ndarray:
            labels, _ = index.knn_query(np.asarray(medians, dtype=np.float32), k=1)
            return np.asarray(labels, dtype=np.int64)[:, 0]

        return nearest

    @validation_scoped
    def _abundance_load_neighborhoods(self, artifact: ArtifactRef) -> Neighborhoods:
        """Return saved Milo neighbourhoods: index cells, sizes, and k-distances."""
        _, group = self._abundance_artifact(
            artifact, "neighborhoods", "make_neighborhoods"
        )
        return read_neighborhoods_slot(group, artifact=artifact)

    @validation_scoped
    def _abundance_count_neighborhoods(
        self,
        neighborhoods: ArtifactRef,
        *,
        sample_by: str,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Count each neighbourhood's cells per sample.

        Samples are the labels of ``sample_by`` over the graph's cells, in
        order of first appearance, as miloR's ``countCells`` orders them. The
        stored membership is streamed in row bands.

        Args:
            neighborhoods: A neighborhoods artifact.
            sample_by: Cell-metadata column of the samples.
            invalidate_cache: Recount even when matching counts exist.

        Returns:
            The ``neighborhood_counts`` artifact; load it with
            ``abundance.load_neighborhood_counts``.

        Raises:
            ValueError: If a graph cell has no sample.
            PermissionError: If counts must be written to a read-only store.
        """
        _, nhood_group = self._abundance_artifact(
            neighborhoods, "neighborhoods", "make_neighborhoods"
        )
        if not isinstance(sample_by, str) or sample_by not in self.cells.columns:
            raise KeyError(f"Sample column {sample_by!r} is not in the cell metadata")
        arguments = NeighborhoodCountArguments(
            neighborhoods=neighborhoods,
            sample_fingerprint=metadata_column_fingerprint(self.cells, sample_by),
            sample_by=sample_by,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=neighborhoods.assay,
            invalidate_cache=invalidate_cache,
            required_arrays=("counts",),
            required_attributes=(
                AttributeRequirement("samples", expected_types=(list,)),
            ),
        )
        if planned.reused:
            logger.info("Reused Milo neighbourhood counts")
            return planned.ref
        self._require_writable("count_neighborhoods")

        nhoods = read_neighborhoods_slot(nhood_group, artifact=neighborhoods)
        cell_idx = self._abundance_cell_rows(nhoods.cell_selection)
        samples: list[str] = []
        counts = np.zeros((nhoods.n_neighborhoods, 0), dtype=np.int64)
        for start, row_counts, ids in self._abundance_iter_membership(nhood_group):
            rows = cell_idx[start : start + row_counts.size]
            labels, _ = factor_labels(
                self._abundance_column(sample_by, rows, role="Sample"),
                name="samples",
            )
            known = set(samples)
            new = [
                label
                for label in unique_in_order(labels).tolist()
                if label not in known
            ]
            if new:
                samples.extend(new)
                counts = np.hstack(
                    [counts, np.zeros((counts.shape[0], len(new)), dtype=np.int64)]
                )
            codes = pd.Index(samples).get_indexer(labels)
            count_block(counts, row_counts, ids, codes)

        with artifact_transaction(self.zw, planned) as group:
            group.attrs.update(
                {
                    "samples": list(samples),
                    "sample_by": sample_by,
                    "neighborhoods": neighborhoods.to_dict(),
                }
            )
            write_small_array(group, "counts", counts)
        logger.info(
            f"Stored counts of {counts.shape[0]} neighbourhoods over "
            f"{len(samples)} samples"
        )
        return planned.ref

    @validation_scoped
    def _abundance_load_neighborhood_counts(
        self, artifact: ArtifactRef
    ) -> NeighborhoodCounts:
        """Return saved neighbourhood x sample counts."""
        _, group = self._abundance_artifact(
            artifact, "neighborhood_counts", "count_neighborhoods"
        )
        return read_neighborhood_counts_slot(group, artifact=artifact)

    def _abundance_sample_values(
        self,
        cell_idx: np.ndarray,
        sample_by: str,
        columns: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Return each column's value per sample over the graph's cells.

        Raises:
            ValueError: If a column varies within a sample.
        """
        per_sample: dict[str, dict[str, Any]] = {column: {} for column in columns}
        label_of: dict[str, dict[str, str]] = {column: {} for column in columns}
        band = _MAX_BAND_ROWS
        for start in range(0, cell_idx.size, band):
            rows = cell_idx[start : start + band]
            samples, _ = factor_labels(
                self._abundance_column(sample_by, rows, role="Sample"),
                name="samples",
            )
            for column in columns:
                values = self._abundance_column(column, rows, role="Design")
                labels, _ = factor_labels(values, name=column)
                pairs = pd.DataFrame(
                    {"sample": samples, "label": labels}
                ).drop_duplicates()
                for position, sample, label in zip(
                    pairs.index.tolist(),
                    pairs["sample"].tolist(),
                    pairs["label"].tolist(),
                    strict=True,
                ):
                    value = values[position]
                    seen = label_of[column].setdefault(sample, label)
                    if seen != label:
                        raise ValueError(
                            f"Design column {column!r} must be constant within each "
                            f"sample; sample {sample!r} has {seen!r} and {label!r}"
                        )
                    per_sample[column].setdefault(sample, value)
        return per_sample

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
        """Test neighbourhood counts between conditions (Milo).

        The design is the treatment-coded ``~ covariates + condition_by`` on
        the samples, whose columns must be constant within each sample.
        Without ``contrast`` the last design column is tested, as in
        miloR's ``testNhoods``: with two conditions, the second level against
        the first. ``levels`` fixes level orders, the reference first, and
        ``contrast`` gives weights over the design columns. Neighbourhoods
        with a mean count of at least ``min_mean`` are tested with edgeR's
        quasi-likelihood F-test, run by edgePython, after ``norm_method``
        normalization. The spatial FDR weights each neighbourhood by the
        inverse of its k-distance (Lun, Richard and Marioni 2017);
        ``fdr_weighting="none"`` leaves it missing.

        The result is saved as an immutable ``neighborhood_tests`` artifact
        unless ``skip_save`` is True.

        Args:
            counts: A neighborhood_counts artifact.
            condition_by: Cell-metadata column of the conditions.
            covariates: Further sample-level columns, such as a batch.
            levels: Level order of design columns.
            contrast: Weights over design columns.
            norm_method: ``"TMM"``, ``"RLE"``, or ``"logMS"``.
            min_mean: Smallest mean count of a tested neighbourhood.
            robust: Robust quasi-likelihood dispersion prior.
            legacy: edgeR's legacy quasi-likelihood method, which miloR uses.
            fdr_weighting: ``"k-distance"`` or ``"none"``.
            skip_save: Return the result without writing it.
            invalidate_cache: Recompute even when a matching result exists.

        Raises:
            ImportError: If edgePython is not installed.
            ValueError: If the design is invalid or not full rank, or the
                samples no longer match the counts.
            PermissionError: If the result must be written to a read-only
                store.
        """
        _, counts_group = self._abundance_artifact(
            counts, "neighborhood_counts", "count_neighborhoods"
        )
        if not isinstance(condition_by, str):
            raise TypeError("condition_by must be a column name")
        if isinstance(covariates, str):
            raise TypeError("covariates must be a sequence of column names")
        covariate_names = tuple(str(name) for name in covariates)
        variables = (*covariate_names, condition_by)
        if len(set(variables)) != len(variables):
            raise ValueError("condition_by and covariates must be distinct columns")
        if norm_method not in ("TMM", "RLE", "logMS"):
            raise ValueError("norm_method must be 'TMM', 'RLE', or 'logMS'")
        if fdr_weighting not in ("k-distance", "none"):
            raise ValueError("fdr_weighting must be 'k-distance' or 'none'")
        resolved_min_mean = float_argument(min_mean, "min_mean")
        if resolved_min_mean < 0:
            raise ValueError("min_mean must be non-negative")
        requested_levels = (
            None
            if levels is None
            else {
                str(name): tuple(str(label) for label in factor_labels(list(order))[0])
                for name, order in levels.items()
            }
        )
        requested_contrast = (
            None
            if contrast is None
            else {str(column): float(weight) for column, weight in contrast.items()}
        )

        table = read_neighborhood_counts_slot(counts_group, artifact=counts)
        nhoods_ref = table.neighborhoods
        _, nhood_group = self._abundance_artifact(
            nhoods_ref, "neighborhoods", "make_neighborhoods"
        )
        nhoods = read_neighborhoods_slot(nhood_group, artifact=nhoods_ref)
        if table.sample_by in variables:
            raise ValueError("The sample column cannot be a design column")
        per_sample = self._abundance_sample_values(
            self._abundance_cell_rows(nhoods.cell_selection),
            table.sample_by,
            variables,
        )
        observed = set(per_sample[condition_by])
        if observed != set(table.samples):
            raise ValueError(
                f"The samples of {table.sample_by!r} no longer match the counts; "
                "recount with abundance.count_neighborhoods"
            )
        design = treatment_design(
            {
                name: np.asarray([per_sample[name][sample] for sample in table.samples])
                for name in variables
            },
            levels=requested_levels,
        )
        effective_contrast = (
            {design.columns[-1]: 1.0}
            if requested_contrast is None
            else requested_contrast
        )
        weights = contrast_vector(design, effective_contrast)
        arguments = NeighborhoodTestArguments(
            counts=counts,
            design_fingerprint=provenance_hash(
                {
                    "samples": list(table.samples),
                    "columns": list(design.columns),
                    "matrix": fingerprint_array(design.matrix),
                }
            ),
            condition_by=condition_by,
            covariates=covariate_names,
            levels=requested_levels,
            contrast=requested_contrast,
            norm_method=norm_method,
            min_mean=resolved_min_mean,
            robust=bool(robust),
            legacy=bool(legacy),
            fdr_weighting=fdr_weighting,
            invalidate_cache=invalidate_cache,
        )
        planned: Any = None
        if not skip_save:
            planned = arguments.plan(
                self.zw,
                scope="assay",
                assay=counts.assay,
                invalidate_cache=invalidate_cache,
                required_arrays=_TEST_ARRAYS,
                required_attributes=(
                    AttributeRequirement("stat_columns", expected_types=(list,)),
                ),
            )
            if planned.reused:
                logger.info("Reused Milo neighbourhood tests")
                return read_neighborhood_test_slot(
                    artifact_group(self.zw, planned.ref), artifact=planned.ref
                )
            self._require_writable("test_neighborhoods")

        fit = differential_abundance(
            table.counts,
            design.matrix,
            weights,
            kth_distance=nhoods.kth_distance if fdr_weighting == "k-distance" else None,
            norm_method=norm_method,
            min_mean=resolved_min_mean,
            robust=bool(robust),
            legacy=bool(legacy),
        )
        sample_index = pd.Index(list(table.samples), name=table.sample_by)
        result = NeighborhoodTestResult(
            counts=counts,
            neighborhoods=nhoods_ref,
            sample_by=table.sample_by,
            condition_by=condition_by,
            covariates=covariate_names,
            design=pd.DataFrame(
                design.matrix, index=sample_index, columns=list(design.columns)
            ),
            contrast=dict(effective_contrast),
            table=pd.DataFrame(
                {
                    "index_cell": np.asarray(nhoods.index_cells, dtype=np.int64),
                    "size": np.asarray(nhoods.sizes, dtype=np.int64),
                    "log_fc": fit.log_fc,
                    "log_cpm": fit.log_cpm,
                    "f_statistic": fit.f_statistic,
                    "p_value": fit.p_value,
                    "fdr": fit.fdr,
                    "spatial_fdr": fit.spatial_fdr,
                    "tested": np.asarray(fit.tested, dtype=bool),
                },
                index=pd.RangeIndex(nhoods.n_neighborhoods, name="neighborhood"),
            ),
            norm_factors=pd.Series(
                fit.norm_factors, index=sample_index, name="norm_factor"
            ),
            lib_size=pd.Series(fit.lib_size, index=sample_index, name="lib_size"),
            common_dispersion=float(fit.common_dispersion),
            average_ql_dispersion=fit.average_ql_dispersion,
            norm_method=norm_method,
            min_mean=resolved_min_mean,
            robust=bool(robust),
            legacy=bool(legacy),
            fdr_weighting=fdr_weighting,
            artifact=None if planned is None else planned.ref,
        )
        if planned is not None:
            with artifact_transaction(self.zw, planned) as group:
                write_neighborhood_test_slot(group, result, tested=fit.tested)
            logger.info(
                f"Stored Milo tests of {int(fit.tested.sum())} neighbourhoods; "
                f"{int(np.sum(fit.spatial_fdr < 0.1))} at spatial FDR < 0.1"
            )
        return result

    @validation_scoped
    def _abundance_load_neighborhood_tests(
        self, artifact: ArtifactRef
    ) -> NeighborhoodTestResult:
        """Return one saved Milo test exactly as it was computed."""
        _, group = self._abundance_artifact(
            artifact, "neighborhood_tests", "test_neighborhoods"
        )
        return read_neighborhood_test_slot(group, artifact=artifact)

    @validation_scoped
    def _abundance_annotate_neighborhoods(
        self, neighborhoods: ArtifactRef, labels: ArtifactRef | CellField
    ) -> pd.DataFrame:
        """Label each neighbourhood with the most common label among its cells.

        ``neighborhoods`` may also be counts or a test derived from them.
        Ties go to the label seen first along the graph's cells, as in
        miloR's ``annotateNhoods``. Cells without a label are left out, and
        a neighbourhood without labelled cells gets no label.

        Returns:
            One row per neighbourhood with the majority ``label`` and its
            ``fraction`` of the neighbourhood's labelled cells.
        """
        nhoods_ref = self._abundance_neighborhoods_ref(neighborhoods)
        _, nhood_group = self._abundance_artifact(
            nhoods_ref, "neighborhoods", "make_neighborhoods"
        )
        nhoods = read_neighborhoods_slot(nhood_group, artifact=nhoods_ref)
        resolved = resolve_grouping(
            self.zw, self.cells, labels, cell_selection=nhoods.cell_selection
        )
        if not np.array_equal(
            resolved.cell_idx, self._abundance_cell_rows(nhoods.cell_selection)
        ):
            raise ValueError("labels do not cover the graph's cells")
        valid = valid_category_mask(resolved.labels, missing_mask=resolved.missing_mask)
        names, _ = factor_labels(resolved.labels[valid], name="labels")
        levels = tuple(str(level) for level in unique_in_order(names).tolist())
        codes = np.full(valid.size, -1, dtype=np.int64)
        codes[valid] = pd.Index(levels).get_indexer(names)
        label_counts = np.zeros((nhoods.n_neighborhoods, len(levels)), dtype=np.int64)
        for start, row_counts, ids in self._abundance_iter_membership(nhood_group):
            count_block(
                label_counts, row_counts, ids, codes[start : start + row_counts.size]
            )
        annotation = annotate_neighborhoods(label_counts, levels)
        return pd.DataFrame(
            {"label": annotation.label, "fraction": annotation.fraction},
            index=pd.RangeIndex(nhoods.n_neighborhoods, name="neighborhood"),
        )
