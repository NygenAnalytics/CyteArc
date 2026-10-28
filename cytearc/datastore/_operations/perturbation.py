import hashlib
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any


import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    GuideCallArguments,
    GuideCallImportArguments,
    GuideThresholdArguments,
    PerturbationEfficiencyArguments,
    PerturbationLabelArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import sample_codes
from ...perturbation.calling import threshold_calls, top_guide_calls
from ...perturbation.guide_calls import (
    GuideCalls,
    guide_call_arrays,
    guide_calls_frames,
    summarize_guide_counts,
    validated_guide_ids,
)
from ...perturbation.guide_thresholds import (
    DEFAULT_FRACTION_THRESHOLD,
    DEFAULT_MIN_UMI,
    FITTED_METHODS,
    METHOD_PARAMETERS,
    METHODS,
    GuideThresholds,
    call_thresholds,
    guide_threshold_frames,
)
from ...perturbation.efficiency import (
    EFFICIENCY_STATUSES,
    PerturbationEfficiency,
    efficiency_arrays,
    target_features,
)
from ...perturbation.labels import (
    POLICIES,
    PerturbationLabels,
    guide_targets,
    perturbation_label_arrays,
    perturbation_label_frames,
)
from ...perturbation.histograms import (
    MAX_COUNT,
    GuideCountHistograms,
    guide_count_histograms,
    histogram_bytes,
)
from ...perturbation.mixtures import (
    FIT_STATUSES,
    MAX_ITERATIONS,
    FITTED,
    NO_CELLS,
    NO_FIT,
    NO_POSITIVE,
    TOO_SPARSE,
    fit_gaussian,
    fit_poisson_gaussian,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_strings,
    require_complete_artifact,
)
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import float_argument, integer_argument
from ...utils.logging import logger
from ...utils.warnings import warn

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _PerturbationOperationsBase
else:
    _PerturbationOperationsBase = object

_GUIDE_ASSAY = "CRISPR"
_GUIDE_CALL_ARRAYS = (
    "values",
    "n_guides",
    "top_guide",
    "top_umi",
    "top_fraction",
    "total_umi",
    "guide_codes",
    "guide_umis",
)
# Barcodes of the call file that are not selected cells, named in the
# artifact's attributes and in the warning.
_UNMATCHED_EXAMPLES = 5
# Bytes held for each selected cell beside the streamed counts: its ID and
# position, its label, and its summary and call arrays.
_BYTES_PER_CELL = 160
# Bytes held for each call of the file: its row, guide, UMIs, and order.
_BYTES_PER_CALL = 48


# Fits named in the warning about guides that call nothing.
_FIT_EXAMPLES = 5
_STATUS_TEXT = {
    NO_CELLS: "have no cell with a UMI of the guide",
    TOO_SPARSE: "have too few cells or counts to fit",
    NO_FIT: "have no admissible fit",
    NO_POSITIVE: "have no guide population above the background",
}
# Above this share of called cells with several guides, a gaussian fit over
# every cell has likely set thresholds among the ambient counts.
_GAUSSIAN_MULTIPLE_SHARE = 0.5
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


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _numba_threads(nthreads: int) -> Iterator[None]:
    """Run the parallel fits on ``nthreads`` Numba threads, then restore them."""
    import numba

    previous = numba.get_num_threads()
    available = int(getattr(numba.config, "NUMBA_NUM_THREADS"))
    numba.set_num_threads(min(max(1, int(nthreads)), available))
    try:
        yield
    finally:
        numba.set_num_threads(previous)


def _stored_samples(group: zarr.Group) -> list[str | None]:
    """Return the sample labels saved with guide fits, None for one sample."""
    labels = group.attrs["samples"]
    if not isinstance(labels, list):
        raise ValueError("The guide_thresholds artifact has no list of samples")
    return [None if label is None else str(label) for label in labels]


def _called_cells(histograms: GuideCountHistograms, table: np.ndarray) -> int:
    """Return how many calls a table of smallest called counts makes."""
    entries = np.repeat(
        np.arange(len(histograms.offsets) - 1), np.diff(histograms.offsets)
    )
    limits = np.asarray(table, dtype=np.int64).reshape(-1)[entries]
    called = (limits > 0) & (histograms.counts >= limits)
    return int(histograms.cells[called].sum())


def _unfitted_warning(
    status: np.ndarray, samples: list[str | None], guide_ids: np.ndarray
) -> str | None:
    """Describe the fits that call no cell, or return None when every fit calls."""
    unfitted = status != FITTED
    total = int(np.count_nonzero(unfitted))
    if not total:
        return None
    parts = [
        f"{int(np.count_nonzero(status == code))} {text}"
        for code, text in _STATUS_TEXT.items()
        if np.any(status == code)
    ]
    rows, columns = np.nonzero(unfitted)
    examples = [
        repr(str(guide_ids[column]))
        + ("" if samples[row] is None else f" in sample {samples[row]!r}")
        for row, column in zip(
            rows[:_FIT_EXAMPLES], columns[:_FIT_EXAMPLES], strict=True
        )
    ]
    return (
        f"{total} of {status.size} guide fits call no cell: "
        + ", ".join(parts)
        + ". For example "
        + ", ".join(examples)
        + ". perturbation.load_guide_thresholds lists every status."
    )


class _PerturbationOperationsMixin(_PerturbationOperationsBase):
    def _guide_assay(self, from_assay: str | None) -> Any:
        """Return the guide assay after checking its declared type and counts."""
        from ...assay.classification import declared_assay_type

        name = _GUIDE_ASSAY if from_assay is None else from_assay
        if not isinstance(name, str):
            raise TypeError("from_assay must be an assay name or None")
        declared = (
            declared_assay_type(self._get_assay(name))
            if name in self.assay_names
            else None
        )
        if declared != _GUIDE_ASSAY:
            raise TypeError(
                "Guide calls need an assay declared with type 'CRISPR'; "
                f"{name!r} is declared as {declared!r}"
            )
        assay = self._get_assay(name)
        count_dtype = np.dtype(assay.rawData.dtype)
        if count_dtype.kind != "u":
            raise TypeError(
                "Guide calls need non-negative integer counts, but assay "
                f"{name!r} stores {count_dtype} counts"
            )
        return assay

    @validation_scoped
    def _perturbation_import_guide_calls(
        self,
        path: str | os.PathLike[str],
        *,
        cell_selection: ArtifactRef,
        from_assay: str | None = None,
        barcode_prefix: str | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import Cell Ranger's guide calls for a set of cells.

        Cell Ranger's CRISPR Guide Capture analysis calls the guides of each
        cell and writes them to ``crispr_analysis/protospacer_calls_per_cell.csv``.
        The import joins each row's barcode to a selected cell and saves the
        calls as a ``guide_calls`` artifact, which CyteArc's own guide calling
        writes too. A selected cell absent from the file has no call, as Cell
        Ranger leaves out cells without a confident call. The guide counts of
        the assay give each cell's top guide, its UMIs, and the cell's total
        guide UMIs, read in row blocks within the memory budget.

        A cell's ID is ``barcode_prefix`` followed by the barcode, matched
        exactly. A merged store prefixes each sample's barcodes with its name
        and two underscores, so import each sample's file with its own prefix
        and a cell selection of that sample. Barcodes that match no selected
        cell are counted, saved in the artifact's ``n_unmatched_barcodes`` and
        ``unmatched_barcode_examples`` attributes, and named in a warning.

        The artifact's ``values`` hold the called guide's ID for a cell with
        one call, ``"Negative"`` for a cell with none, and ``"Multiple"`` for a
        cell with several. Its other per-cell arrays are ``n_guides``,
        ``top_guide`` (the guide with the most UMIs, an empty string for a cell
        with no guide UMIs), ``top_umi``, ``top_fraction`` (the top guide's
        share of the cell's guide UMIs, NaN without UMIs) and ``total_umi``,
        and the two-dimensional ``guide_codes`` and ``guide_umis``, which hold
        each cell's called guides, most UMIs first, as positions among the
        assay's guides, padded with -1 and 0. ``guides`` holds the guide IDs
        in assay order. ``select_cells`` reads the one-dimensional arrays, and
        ``artifacts.load_values`` reads every per-cell array.

        Args:
            path: Cell Ranger's ``protospacer_calls_per_cell.csv``, plain or
                gzip-compressed.
            cell_selection: The cells that the calls label.
            from_assay: The guide assay, declared with type ``CRISPR``
                (default: ``'CRISPR'``).
            barcode_prefix: Text that precedes each barcode in the cell IDs,
                such as ``"lane1__"`` in a merged store. None adds nothing.
            invalidate_cache: Import again instead of reusing saved calls.

        Returns:
            A complete assay-scoped ``guide_calls`` artifact. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls`
            or plot it with ``ds.plots.guide_calls``.

        Raises:
            TypeError: If an argument has the wrong type, or the assay is not
                declared ``CRISPR`` or does not store unsigned integer counts.
            ValueError: If the call file is malformed, a guide ID is empty,
                repeated, or a label reserved for cells without one guide, or a
                called guide is not a feature of the assay.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            PermissionError: If the store is read-only and the calls are not
                saved yet.
            MemoryError: If the calls and the cell summaries do not fit the
                memory budget.
        """
        from ...readers.cellranger import read_protospacer_calls

        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path to Cell Ranger's call file")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if barcode_prefix is not None and not isinstance(barcode_prefix, str):
            raise TypeError("barcode_prefix must be text or None")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        assay = self._guide_assay(from_assay)
        guide_ids = validated_guide_ids(assay.feats.fetch_all("ids"))
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = len(cell_index)
        if not n_cells:
            raise ValueError("cell_selection selects no cell")
        self._require_measured_cells(
            assay.name,
            cell_selection,
            operation=GuideCallImportArguments.operation,
        )
        arguments = GuideCallImportArguments(
            cell_selection=cell_selection,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            guide_ids_fingerprint=fingerprint_strings(guide_ids.astype(str)),
            file_sha256=_file_sha256(path),
            barcode_prefix=barcode_prefix,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        n_guide_ids = len(guide_ids)

        def has_guides(_ref: ArtifactRef, group: zarr.Group) -> bool:
            guides = group.get("guides")
            return isinstance(guides, zarr.Array) and guides.shape == (n_guide_ids,)

        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={
                "values": ((n_cells,), None),
                "n_guides": ((n_cells,), "i"),
                "top_guide": ((n_cells,), None),
                "top_umi": ((n_cells,), "u"),
                "top_fraction": ((n_cells,), "f"),
                "total_umi": ((n_cells,), "u"),
                "guide_codes": ((n_cells, None), "i"),
                "guide_umis": ((n_cells, None), "u"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("n_file_barcodes", "n_unmatched_barcodes"),
            reuse_validator=has_guides,
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)

        file_calls = read_protospacer_calls(path)
        n_calls = len(file_calls.features)
        resident = _BYTES_PER_CELL * n_cells + _BYTES_PER_CALL * n_calls
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Importing {n_calls} guide calls for {n_cells} cells holds about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget, or pass a smaller "
                "cell_selection."
            )
        guide_positions = pd.Index(guide_ids.astype(str)).get_indexer(
            file_calls.features.astype(str)
        )
        unknown = np.flatnonzero(guide_positions < 0)
        if len(unknown):
            names = set(np.asarray(assay.feats.fetch_all("names")).astype(str))
            examples = pd.unique(file_calls.features[unknown])[:5].tolist()
            hint = (
                "; they match guide names, so the file names guides by name "
                "while the assay's IDs differ"
                if all(example in names for example in examples)
                else ""
            )
            raise ValueError(
                f"{len(pd.unique(file_calls.features[unknown]))} called guides are "
                f"not features of assay {assay.name!r}, for example "
                + ", ".join(repr(example) for example in examples)
                + hint
            )
        cell_ids = np.asarray(self.cells.fetch_all("ids"))[cell_index].astype(str)
        prefix = "" if barcode_prefix is None else barcode_prefix
        row_cells = pd.Index(cell_ids).get_indexer(
            np.char.add(prefix, file_calls.barcodes.astype(str))
        )
        unmatched = np.flatnonzero(row_cells < 0)
        examples = file_calls.barcodes[unmatched[:_UNMATCHED_EXAMPLES]].tolist()
        matched_calls = row_cells[file_calls.rows] >= 0
        arrays = guide_call_arrays(
            row_cells[file_calls.rows][matched_calls],
            guide_positions[matched_calls],
            file_calls.umis[matched_calls],
            n_cells=n_cells,
            guide_ids=guide_ids,
        )
        blocks = assay.rawData[cell_index, :]._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Summarising guide counts",
            prefetch=None,
            row_mask=None,
            resident_bytes=resident + arrays.guide_codes.nbytes * 2,
            memory_bytes=self.memoryBytes,
        )
        summary = summarize_guide_counts(blocks, n_cells, n_guide_ids)
        top_guide = np.full(n_cells, "", dtype=object)
        counted = summary.top_code >= 0
        top_guide[counted] = guide_ids[summary.top_code[counted]]
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": arrays.values,
                "n_guides": arrays.n_guides,
                "top_guide": top_guide,
                "top_umi": summary.top_umi,
                "top_fraction": summary.top_fraction,
                "total_umi": summary.total_umi,
                "guide_codes": arrays.guide_codes,
                "guide_umis": arrays.guide_umis,
                "guides": guide_ids,
            },
            attributes={
                "n_file_barcodes": int(len(file_calls.barcodes)),
                "n_unmatched_barcodes": int(len(unmatched)),
                "unmatched_barcode_examples": [str(value) for value in examples],
            },
        )
        if len(unmatched):
            logger.warning(
                f"{len(unmatched)} of {len(file_calls.barcodes)} barcodes of the call "
                "file match no selected cell, for example "
                + ", ".join(repr(str(value)) for value in examples)
                + ". Check barcode_prefix and cell_selection."
            )
        logger.info(
            f"Imported guide calls of {int(np.count_nonzero(arrays.n_guides))} of "
            f"{n_cells} cells"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_guide_calls(self, calls: ArtifactRef) -> GuideCalls:
        """Read saved guide calls as a table of cells and a table of calls.

        Args:
            calls: A ``guide_calls`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls`.

        Returns:
            The calls, with one row per cell and one row per called guide of a
            cell, and the guide IDs of the assay.

        Raises:
            TypeError: If ``calls`` is not an ArtifactRef.
            ValueError: If it is not a ``guide_calls`` artifact.
            MemoryError: If its values do not fit the datastore memory budget.
        """
        if not isinstance(calls, ArtifactRef):
            raise TypeError("calls must be an ArtifactRef")
        if calls.kind != GuideCallImportArguments.artifact_kind:
            raise ValueError(
                f"calls must be a guide_calls artifact, not {calls.kind!r}"
            )
        status = require_complete_artifact(self.zw, calls)
        group = artifact_group(self.zw, calls)
        guide_ids = np.asarray(as_zarr_array(group["guides"], name="guides")[:]).astype(
            str
        )
        # The canonical labels are read by default; the others by name.
        loaded = {
            name: self._artifacts_load_values(
                calls, value=None if name == "values" else name
            )
            for name in _GUIDE_CALL_ARRAYS
        }
        cells, call_table = guide_calls_frames(
            loaded["values"].cell_ids,
            **{name: np.asarray(values.values) for name, values in loaded.items()},
            guide_ids=guide_ids,
        )
        return GuideCalls(
            cells=cells,
            calls=call_table,
            guides=pd.Index(guide_ids, name="guide"),
            source=str(status.operation),
        )

    def _guide_sample_labels(
        self, sample_column: str, cell_index: np.ndarray
    ) -> np.ndarray:
        """Read each selected cell's label in a cell metadata column."""
        labels: np.ndarray = apply_missing_mask(
            np.asarray(
                read_metadata_rows_chunkwise(self.cells, sample_column, cell_index)
            ),
            read_metadata_missing_rows_chunkwise(self.cells, sample_column, cell_index),
            labels=True,
        )
        return labels

    def _plan_guide_thresholds(
        self,
        *,
        assay: str,
        cell_selection: ArtifactRef,
        dataset_fingerprint: str,
        guide_ids_fingerprint: str,
        sample_assignments_fingerprint: str | None,
        method: str,
        sample_columns: tuple[str, ...],
        n_samples: int,
        n_guides: int,
        invalidate_cache: bool,
    ) -> PlannedArtifact:
        """Plan the ``guide_thresholds`` artifact of a fitted method."""
        arguments = GuideThresholdArguments(
            cell_selection=cell_selection,
            dataset_fingerprint=dataset_fingerprint,
            guide_ids_fingerprint=guide_ids_fingerprint,
            sample_assignments_fingerprint=sample_assignments_fingerprint,
            method=method,
            sample_columns=sample_columns,
            invalidate_cache=invalidate_cache,
        )
        grid = (n_samples, n_guides)

        def holds_histograms(_ref: ArtifactRef, group: zarr.Group) -> bool:
            guides = group.get("guides")
            offsets = group.get("histogram_offsets")
            counts = group.get("histogram_counts")
            return (
                isinstance(guides, zarr.Array)
                and guides.shape == (n_guides,)
                and isinstance(offsets, zarr.Array)
                and isinstance(counts, zarr.Array)
                and int(np.asarray(offsets[-1:])[0]) == counts.shape[0]
            )

        return arguments.plan(
            self.zw,
            scope="assay",
            assay=assay,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("n_cells", shape=(n_samples,), dtype_kind="i"),
                ArrayRequirement("status", shape=grid, dtype_kind="i"),
                ArrayRequirement("threshold", shape=grid, dtype_kind="u"),
                ArrayRequirement("n_umi_cells", shape=grid, dtype_kind="i"),
                ArrayRequirement("max_umi", shape=grid, dtype_kind="u"),
                ArrayRequirement("iterations", shape=grid, dtype_kind="i"),
                *(
                    ArrayRequirement(name, shape=grid, dtype_kind="f")
                    for name in METHOD_PARAMETERS[method]
                ),
                ArrayRequirement(
                    "histogram_offsets",
                    shape=(n_samples * n_guides + 1,),
                    dtype_kind="i",
                ),
                ArrayRequirement("histogram_counts", shape=(None,), dtype_kind="u"),
                ArrayRequirement("histogram_cells", shape=(None,), dtype_kind="i"),
            ),
            required_attributes=(
                AttributeRequirement("samples", expected_types=(list,)),
                AttributeRequirement(
                    "status_names",
                    predicate=lambda value: list(value) == list(FIT_STATUSES),
                ),
            ),
            reuse_validator=holds_histograms,
        )

    def _fit_guide_thresholds(
        self,
        assay: Any,
        planned: PlannedArtifact,
        *,
        cell_index: np.ndarray,
        codes: np.ndarray,
        samples: list[str | None],
        method: str,
        guide_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, GuideCountHistograms]:
        """Count the UMIs of each guide, fit it in every sample, and save the fits.

        Returns the fitted thresholds, the statuses and the histograms.
        """
        n_cells = len(cell_index)
        n_samples = len(samples)
        n_guides = len(guide_ids)
        resident = histogram_bytes(n_samples, n_guides, n_cells) + 16 * n_cells
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Fitting guide thresholds for {n_cells} cells in {n_samples} "
                f"samples and {n_guides} guides holds about {resident} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget, or pass a smaller cell_selection or fewer samples."
            )
        blocks = assay.rawData[cell_index, :]._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Counting guide UMIs",
            prefetch=None,
            row_mask=None,
            resident_bytes=resident,
            memory_bytes=self.memoryBytes,
        )
        histograms = guide_count_histograms(
            blocks, codes, n_samples=n_samples, n_guides=n_guides
        )
        with _numba_threads(self.nthreads):
            fits = (
                fit_poisson_gaussian(histograms)
                if method == "poisson_gaussian"
                else fit_gaussian(histograms)
            )
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "guides": guide_ids,
                "n_cells": histograms.n_cells,
                "status": fits.status,
                "threshold": fits.threshold.astype(np.uint32),
                "n_umi_cells": histograms.nonzero_cells(),
                "max_umi": histograms.max_counts().astype(np.uint32),
                "iterations": fits.iterations,
                **{name: getattr(fits, name) for name in METHOD_PARAMETERS[method]},
                "histogram_offsets": histograms.offsets,
                "histogram_counts": histograms.counts.astype(np.uint32),
                "histogram_cells": histograms.cells,
            },
            attributes={"samples": samples, "status_names": list(FIT_STATUSES)},
        )
        stopped = int(np.count_nonzero(fits.iterations >= MAX_ITERATIONS))
        if stopped:
            logger.info(
                f"{stopped} guide fits stopped at {MAX_ITERATIONS} iterations "
                "before their loss settled"
            )
        return fits.threshold, fits.status, histograms

    def _stored_guide_status(
        self, thresholds: ArtifactRef
    ) -> tuple[np.ndarray, list[str | None]]:
        """Read the statuses and samples of saved fits."""
        group = artifact_group(self.zw, thresholds)
        status = np.asarray(as_zarr_array(group["status"], name="status")[:])
        return status, _stored_samples(group)

    def _stored_guide_fits(
        self, thresholds: ArtifactRef
    ) -> tuple[np.ndarray, np.ndarray, GuideCountHistograms, list[str | None]]:
        """Read the thresholds, statuses, histograms and samples of saved fits."""
        group = artifact_group(self.zw, thresholds)

        def array(name: str) -> zarr.Array:
            return as_zarr_array(group[name], name=name)

        # The thresholds, statuses and offsets as int64, and each count and
        # its cells as int64.
        needed = 24 * int(np.prod(array("threshold").shape)) + 16 * int(
            array("histogram_counts").shape[0]
        )
        if needed > self.memoryBytes:
            raise MemoryError(
                f"Reading the saved guide fits holds about {needed} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget."
            )

        def read(name: str) -> np.ndarray:
            return np.asarray(array(name)[:])

        threshold = read("threshold").astype(np.int64)
        histograms = GuideCountHistograms(
            n_cells=read("n_cells").astype(np.int64),
            n_guides=threshold.shape[1],
            offsets=read("histogram_offsets").astype(np.int64),
            counts=read("histogram_counts").astype(np.int64),
            cells=read("histogram_cells").astype(np.int64),
        )
        return threshold, read("status"), histograms, _stored_samples(group)

    def _warn_about_guide_calls(
        self,
        method: str,
        status: np.ndarray | None,
        samples: list[str | None],
        guide_ids: np.ndarray,
        n_guides: np.ndarray,
    ) -> None:
        """Warn about fits that call nothing and about gaussian fits among ambient counts."""
        message = (
            None if status is None else _unfitted_warning(status, samples, guide_ids)
        )
        if message is not None:
            warn(message)
        if method != "gaussian":
            return
        called = int(np.count_nonzero(n_guides > 0))
        several = int(np.count_nonzero(n_guides > 1))
        if called and several > _GAUSSIAN_MULTIPLE_SHARE * called:
            warn(
                f"With method='gaussian', {several / called:.0%} of the cells with "
                "a call have several guides. A fit over every cell sets low "
                "thresholds when most cells hold no UMI of a guide; check "
                "plots.guide_thresholds, or call with method='poisson_gaussian'."
            )

    @validation_scoped
    def _perturbation_call_guides(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        method: str = "poisson_gaussian",
        min_umi: int | None = None,
        fraction_threshold: float | None = None,
        sample_column: str | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call the guides of each cell from its guide counts.

        Every cell holds a few ambient guide molecules, so a guide is called in
        a cell only when its UMIs stand out. Four methods decide:

        - ``"poisson_gaussian"`` (default) fits each guide's UMI counts in the
          cells that hold any, as a Poisson background and a Gaussian on log2
          counts (Replogle et al. 2022, as crispat and pertpy fit it), and calls
          the guide from the smallest count whose Gaussian posterior exceeds
          one half. ``min_umi`` defaults to 1.
        - ``"gaussian"`` fits two Gaussians with one shared variance to
          log10(count + 1) of every cell, as Cell Ranger describes, and calls
          from the smallest count where the higher one wins. ``min_umi``
          defaults to 3, Cell Ranger's floor. When most cells hold no UMI of a
          guide, the fit can split them from the cells with ambient UMIs and
          call those; a warning names this case.
        - ``"fixed"`` calls every guide from ``min_umi`` UMIs, which it needs.
        - ``"top_guide"`` calls each cell's top guide when its share of the
          cell's guide UMIs exceeds ``fraction_threshold`` (default 0.8) and
          it has ``min_umi`` UMIs (default 5), as SCEPTRE's maximum rule does.
          A cell gets at most one call; a cell whose top guide falls short is
          labelled ``"Negative"``.

        The fitted methods fit each guide in each sample of ``sample_column``
        separately and save the fits, the thresholds and each guide's
        histogram of counts as a ``guide_thresholds`` artifact, which the calls
        record as an input. ``min_umi`` raises each threshold and belongs to
        the calls, so calling again with another floor reuses the fits and
        reads the counts once. The counts are read in row blocks within the
        memory budget: once to count each guide's UMIs and once to call.

        The calls are saved as a ``guide_calls`` artifact, as
        :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls`
        saves Cell Ranger's: ``values`` labels each cell with its one called
        guide, ``"Negative"`` or ``"Multiple"``, beside each cell's number of
        calls, top guide, top UMIs, share and total guide UMIs, and its called
        guides with their UMIs.

        Args:
            cell_selection: The cells to call.
            from_assay: The guide assay, declared with type ``CRISPR``
                (default: ``'CRISPR'``).
            method: ``"poisson_gaussian"``, ``"gaussian"``, ``"fixed"`` or
                ``"top_guide"``.
            min_umi: The fewest UMIs of a called guide. None takes the
                method's default; ``"fixed"`` needs a value.
            fraction_threshold: For ``"top_guide"`` only, the share of the
                cell's guide UMIs that its top guide must exceed, at least 0
                and below 1. None takes 0.8.
            sample_column: For the fitted methods only, a cell metadata
                column whose labels split the cells into samples that are fitted
                separately, such as GEM wells. Every selected cell needs a
                label. None fits all selected cells together.
            invalidate_cache: Compute again instead of reusing saved fits and
                calls.

        Returns:
            A complete assay-scoped ``guide_calls`` artifact. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls`,
            read its fits with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_guide_thresholds`,
            and plot them with ``ds.plots.guide_thresholds``.

        Raises:
            TypeError: If an argument has the wrong type, or the assay is not
                declared ``CRISPR`` or does not store unsigned integer counts.
            ValueError: If an argument is out of range or does not apply to
                the method, a sample column is unknown or lacks a label for a
                selected cell, or a guide ID is empty, repeated, or reserved.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            PermissionError: If the store is read-only and the calls are not
                saved yet.
            MemoryError: If the histograms or the calls do not fit the memory
                budget.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(method, str) or method not in METHODS:
            raise ValueError(
                "method must be 'poisson_gaussian', 'gaussian', 'fixed' or "
                f"'top_guide', not {method!r}"
            )
        fitted = method in FITTED_METHODS
        if min_umi is not None:
            resolved_min_umi = integer_argument(
                min_umi, "min_umi", minimum=1, maximum=MAX_COUNT
            )
        elif method == "fixed":
            raise ValueError(
                "method='fixed' needs min_umi, the UMI count from which every "
                "guide is called"
            )
        else:
            resolved_min_umi = DEFAULT_MIN_UMI[method]
        fraction = DEFAULT_FRACTION_THRESHOLD
        if method != "top_guide" and fraction_threshold is not None:
            raise ValueError("fraction_threshold applies only to method='top_guide'")
        if fraction_threshold is not None:
            fraction = float_argument(fraction_threshold, "fraction_threshold")
            if not 0.0 <= fraction < 1.0:
                raise ValueError("fraction_threshold must be at least 0 and below 1")
        if sample_column is not None:
            if not fitted:
                raise ValueError(
                    "sample_column applies only to method='poisson_gaussian' and "
                    "method='gaussian', which fit each sample alone"
                )
            if not isinstance(sample_column, str) or not sample_column:
                raise TypeError("sample_column must be a non-empty string or None")
            if sample_column not in self.cells.columns:
                raise ValueError(
                    f"sample_column '{sample_column}' not found in cell metadata"
                )
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        assay = self._guide_assay(from_assay)
        guide_ids = validated_guide_ids(assay.feats.fetch_all("ids"))
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = len(cell_index)
        if not n_cells:
            raise ValueError("cell_selection selects no cell")
        self._require_measured_cells(
            assay.name,
            cell_selection,
            operation=GuideCallArguments.operation,
        )
        columns = () if sample_column is None else (sample_column,)
        codes, labels, assignments = sample_codes(
            columns,
            [self._guide_sample_labels(column, cell_index) for column in columns],
            n_cells,
            reason="Guide thresholds are fitted within each sample, so every "
            "selected cell needs one",
        )
        samples: list[str | None] = [
            None if sample_column is None else labels_by_column[sample_column]
            for labels_by_column in labels
        ]
        dataset_fingerprint = self._ensure_dataset_fingerprint(assay.name)
        guide_ids_fingerprint = fingerprint_strings(guide_ids.astype(str))
        n_guide_ids = len(guide_ids)
        thresholds = (
            self._plan_guide_thresholds(
                assay=assay.name,
                cell_selection=cell_selection,
                dataset_fingerprint=dataset_fingerprint,
                guide_ids_fingerprint=guide_ids_fingerprint,
                sample_assignments_fingerprint=assignments,
                method=method,
                sample_columns=columns,
                n_samples=len(samples),
                n_guides=n_guide_ids,
                invalidate_cache=invalidate_cache,
            )
            if fitted
            else None
        )
        arguments = GuideCallArguments(
            cell_selection=cell_selection,
            dataset_fingerprint=dataset_fingerprint,
            guide_ids_fingerprint=guide_ids_fingerprint,
            guide_thresholds=None if thresholds is None else thresholds.ref,
            method=method,
            min_umi=resolved_min_umi,
            invalidate_cache=invalidate_cache,
            fraction_threshold=fraction if method == "top_guide" else None,
        )
        record = arguments.to_record()

        def has_guides(_ref: ArtifactRef, group: zarr.Group) -> bool:
            guides = group.get("guides")
            return isinstance(guides, zarr.Array) and guides.shape == (n_guide_ids,)

        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={
                "values": ((n_cells,), None),
                "n_guides": ((n_cells,), "i"),
                "top_guide": ((n_cells,), None),
                "top_umi": ((n_cells,), "u"),
                "top_fraction": ((n_cells,), "f"),
                "total_umi": ((n_cells,), "u"),
                "guide_codes": ((n_cells, None), "i"),
                "guide_umis": ((n_cells, None), "u"),
            },
            invalidate_cache=invalidate_cache,
            reuse_validator=has_guides,
        )
        status: np.ndarray | None = None
        if planned.reused:
            if thresholds is not None:
                status, samples = self._stored_guide_status(thresholds.ref)
            stored = artifact_group(self.zw, planned.ref)
            self._warn_about_guide_calls(
                method,
                status,
                samples,
                guide_ids,
                np.asarray(as_zarr_array(stored["n_guides"], name="n_guides")[:]),
            )
            return planned.ref
        self._require_writable(arguments.operation)
        if thresholds is None:
            table = np.full(
                (1, n_guide_ids),
                resolved_min_umi if method == "fixed" else 0,
                dtype=np.int64,
            )
            n_calls = (
                int(
                    np.asarray(self.cells.fetch_all(f"{assay.name}_nFeatures"))[
                        cell_index
                    ].sum()
                )
                if method == "fixed"
                else n_cells
            )
        else:
            if thresholds.reused:
                threshold, status, histograms, samples = self._stored_guide_fits(
                    thresholds.ref
                )
            else:
                threshold, status, histograms = self._fit_guide_thresholds(
                    assay,
                    thresholds,
                    cell_index=cell_index,
                    codes=codes,
                    samples=samples,
                    method=method,
                    guide_ids=guide_ids,
                )
            table = call_thresholds(threshold, resolved_min_umi)
            n_calls = _called_cells(histograms, table)
            # The histograms are saved; the second pass does not need them.
            del histograms
        resident = _BYTES_PER_CELL * n_cells + _BYTES_PER_CALL * n_calls
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Calling guides makes {n_calls} calls for {n_cells} cells and "
                f"holds about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget, or pass a smaller "
                "cell_selection."
            )
        blocks = assay.rawData[cell_index, :]._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Calling guides",
            prefetch=None,
            row_mask=None,
            resident_bytes=resident,
            memory_bytes=self.memoryBytes,
        )
        calls, summary = threshold_calls(blocks, codes, table)
        if method == "top_guide":
            calls = top_guide_calls(
                summary, min_umi=resolved_min_umi, fraction_threshold=fraction
            )
        arrays = guide_call_arrays(
            calls.cells, calls.codes, calls.umis, n_cells=n_cells, guide_ids=guide_ids
        )
        top_guide = np.full(n_cells, "", dtype=object)
        counted = summary.top_code >= 0
        top_guide[counted] = guide_ids[summary.top_code[counted]]
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": arrays.values,
                "n_guides": arrays.n_guides,
                "top_guide": top_guide,
                "top_umi": summary.top_umi,
                "top_fraction": summary.top_fraction,
                "total_umi": summary.total_umi,
                "guide_codes": arrays.guide_codes,
                "guide_umis": arrays.guide_umis,
                "guides": guide_ids,
            },
        )
        self._warn_about_guide_calls(
            method, status, samples, guide_ids, arrays.n_guides
        )
        n_called = int(np.count_nonzero(arrays.n_guides))
        logger.info(
            f"Called guides in {n_called} of {n_cells} cells, "
            f"{int(np.count_nonzero(arrays.n_guides > 1))} of them with several"
        )
        if method == "top_guide":
            short = int(
                np.count_nonzero((summary.total_umi > 0) & (arrays.n_guides == 0))
            )
            if short:
                logger.info(
                    f"{short} cells hold guide UMIs but no call: their top guide "
                    f"holds at most {fraction:g} of their guide UMIs or fewer than "
                    f"{resolved_min_umi} UMIs"
                )
        return planned.ref

    @validation_scoped
    def _perturbation_load_guide_thresholds(
        self,
        thresholds: ArtifactRef,
        *,
        guides: Sequence[str] | None = None,
    ) -> GuideThresholds:
        """Read the fitted thresholds of guide calls, with each guide's histogram.

        Args:
            thresholds: Guide calls of
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.call_guides`
                with ``method="poisson_gaussian"`` or ``"gaussian"``, or their
                ``guide_thresholds`` artifact. Read through the calls, the
                table also holds the smallest called count once ``min_umi``
                applies.
            guides: The IDs of the guides to read; None reads every guide.

        Returns:
            One row per sample and guide with its status, threshold and fitted
            parameters, and one row per sample, guide and UMI count with its
            cells.

        Raises:
            TypeError: If ``thresholds`` is not an ArtifactRef, or ``guides``
                is not a sequence of guide IDs.
            ValueError: If ``thresholds`` holds no fitted thresholds, or a
                guide is not a guide of the assay.
            MemoryError: If the tables do not fit the memory budget.
        """
        if not isinstance(thresholds, ArtifactRef):
            raise TypeError("thresholds must be an ArtifactRef")
        if guides is not None and (
            isinstance(guides, str)
            or not isinstance(guides, Sequence)
            or not all(isinstance(guide, str) for guide in guides)
        ):
            raise TypeError("guides must be a sequence of guide IDs or None")
        min_umi: int | None = None
        source = thresholds
        if thresholds.kind == GuideCallArguments.artifact_kind:
            calls = require_complete_artifact(self.zw, thresholds)
            if calls.operation == GuideCallImportArguments.operation:
                raise ValueError(
                    "Guide calls imported from Cell Ranger have no fitted "
                    "thresholds; Cell Ranger writes them to "
                    "crispr_analysis/protospacer_umi_thresholds.csv"
                )
            parameters = calls.parameters or {}
            if parameters.get("method") == "fixed":
                raise ValueError(
                    "Guide calls of method='fixed' fit no thresholds: they call "
                    f"every guide from min_umi={parameters.get('min_umi')} UMIs"
                )
            if parameters.get("method") == "top_guide":
                raise ValueError(
                    "Guide calls of method='top_guide' fit no thresholds: they "
                    "call each cell's top guide by its share of the cell's "
                    "guide UMIs"
                )
            min_umi = int(parameters["min_umi"])
            source = calls.input_ref("guide_thresholds")
        elif thresholds.kind != GuideThresholdArguments.artifact_kind:
            raise ValueError(
                "thresholds must be guide calls of perturbation.call_guides or a "
                f"guide_thresholds artifact, not a {thresholds.kind} artifact"
            )
        status = require_complete_artifact(self.zw, source)
        fit_parameters = status.parameters or {}
        method = str(fit_parameters["method"])
        sample_columns = list(fit_parameters.get("sample_columns") or [])
        group = artifact_group(self.zw, source)

        def array(name: str) -> zarr.Array:
            return as_zarr_array(group[name], name=name)

        guide_ids = np.asarray(array("guides")[:]).astype(str)
        if guides is None:
            positions = np.arange(len(guide_ids))
        else:
            positions = pd.Index(guide_ids).get_indexer(list(guides))
            unknown = [
                guide for guide, at in zip(guides, positions, strict=True) if at < 0
            ]
            if unknown:
                raise ValueError(
                    f"{len(unknown)} guides are not guides of the fits, for example "
                    + ", ".join(repr(guide) for guide in unknown[:5])
                )
        offsets = np.asarray(array("histogram_offsets")[:]).astype(np.int64)
        samples = _stored_samples(group)
        n_samples = len(samples)
        entries = (
            np.arange(n_samples)[:, None] * len(guide_ids) + positions[None, :]
        ).reshape(-1)
        sizes = offsets[entries + 1] - offsets[entries]
        needed = 40 * int(sizes.sum()) + 160 * len(entries)
        if needed > self.memoryBytes:
            raise MemoryError(
                f"Reading the thresholds of {len(positions)} guides in {n_samples} "
                f"samples holds about {needed} bytes, more than the memory budget "
                f"of {self.memoryBytes} bytes. Pass guides to read fewer."
            )
        if guides is None:
            counts = np.asarray(array("histogram_counts")[:])
            cells = np.asarray(array("histogram_cells")[:])
        else:
            counts_array, cells_array = (
                array("histogram_counts"),
                array("histogram_cells"),
            )
            starts = offsets[entries]
            counts = np.concatenate(
                [
                    np.asarray(counts_array[start : start + size])
                    for start, size in zip(starts, sizes, strict=True)
                ]
                or [np.zeros(0, dtype=np.uint32)]
            )
            cells = np.concatenate(
                [
                    np.asarray(cells_array[start : start + size])
                    for start, size in zip(starts, sizes, strict=True)
                ]
                or [np.zeros(0, dtype=np.int64)]
            )
        selected_offsets = np.zeros(len(entries) + 1, dtype=np.int64)
        np.cumsum(sizes, out=selected_offsets[1:])
        table, histograms = guide_threshold_frames(
            samples=samples,
            guides=guide_ids[positions],
            n_cells=np.asarray(array("n_cells")[:]),
            status=np.asarray(array("status")[:])[:, positions],
            threshold=np.asarray(array("threshold")[:])[:, positions],
            n_umi_cells=np.asarray(array("n_umi_cells")[:])[:, positions],
            max_umi=np.asarray(array("max_umi")[:])[:, positions],
            iterations=np.asarray(array("iterations")[:])[:, positions],
            parameters={
                name: np.asarray(array(name)[:])[:, positions]
                for name in METHOD_PARAMETERS[method]
            },
            histogram_offsets=selected_offsets,
            histogram_counts=counts,
            histogram_cells=cells,
            min_umi=min_umi,
        )
        return GuideThresholds(
            table=table,
            histograms=histograms,
            guides=pd.Index(guide_ids, name="guide"),
            method=method,
            sample_column=sample_columns[0] if sample_columns else None,
            min_umi=min_umi,
        )

    @validation_scoped
    def _perturbation_labels(
        self,
        calls: ArtifactRef,
        *,
        controls: Sequence[str],
        target_column: str | None = "target_gene_name",
        targets: Mapping[str, str] | None = None,
        policy: str = "single",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Label each cell by the genes that its called guides target.

        Each guide targets one gene, and control guides, such as non-targeting
        guides, target none. A cell's label is ``"Negative"`` without a called
        guide, ``"control"`` with control guides only, the gene when its
        guides target one gene, with or without control guides, and
        ``"Multiple"`` when they target several. With
        ``policy="combinations"``, a cell whose guides target several genes is
        labelled with the genes, sorted and joined by ``"+"``, such as
        ``"GENE1+GENE2"``.

        Each guide's target comes from ``target_column`` of the guide assay's
        features, which 10x imports fill from Cell Ranger's feature reference
        (``target_gene_name``, with ``"Non-Targeting"`` for non-targeting
        guides), and ``targets`` replaces the column's target of the guides
        it names. Control guides are named by their target, never guessed.

        The labels are saved as a ``perturbation_labels`` artifact for the
        cells of the calls: ``values`` holds the labels beside ``guide_label``
        (the cells' labels in the calls), ``is_control``, ``n_targets``, and
        ``target_codes``, which holds each cell's genes as positions in the
        artifact's ``targets``, padded with -1, so a cell's full set of genes
        is kept under either policy. ``select_cells`` reads the
        one-dimensional arrays, and ``artifacts.load_values`` reads every
        per-cell array.

        Args:
            calls: A ``guide_calls`` artifact, imported from Cell Ranger or made
                by :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.call_guides`.
            controls: The targets of control guides, such as
                ``["Non-Targeting"]``.
            target_column: The feature column of the guide assay that holds
                each guide's target, or None to take every target from
                ``targets``.
            targets: Targets by guide ID that replace the column's, or None.
            policy: ``"single"`` or ``"combinations"``.
            invalidate_cache: Label again instead of reusing saved labels.

        Returns:
            A complete assay-scoped ``perturbation_labels`` artifact. Read it
            with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_labels`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``calls`` is not a ``guide_calls`` artifact,
                ``controls`` names no target or a blank one, neither
                ``target_column`` nor ``targets`` is given, the target column
                is not a feature column, ``targets`` names a guide that is not
                in the assay or gives an empty target, a target is a reserved
                label or, with ``"combinations"``, holds ``"+"``, a control
                matches no target, a called guide has no target, or ``policy``
                is unknown.
            PermissionError: If the store is read-only and the labels are not
                saved yet.
            MemoryError: If the labelling does not fit the memory budget.
        """
        if not isinstance(calls, ArtifactRef):
            raise TypeError("calls must be an ArtifactRef")
        if calls.kind != GuideCallImportArguments.artifact_kind:
            raise ValueError(
                f"calls must be a guide_calls artifact, not {calls.kind!r}"
            )
        if (
            isinstance(controls, str)
            or not isinstance(controls, Sequence)
            or not all(isinstance(name, str) for name in controls)
        ):
            raise TypeError(
                "controls must be a sequence of target names, such as ['Non-Targeting']"
            )
        control_names = tuple(sorted({name.strip() for name in controls}))
        if not control_names or "" in control_names:
            raise ValueError(
                "controls must name at least one target, and no name may be blank"
            )
        if target_column is not None and (
            not isinstance(target_column, str) or not target_column
        ):
            raise TypeError("target_column must be a feature column name or None")
        if targets is not None and (
            not isinstance(targets, Mapping)
            or not all(
                isinstance(guide, str) and isinstance(target, str)
                for guide, target in targets.items()
            )
        ):
            raise TypeError("targets must map guide IDs to target names, or be None")
        if target_column is None and not targets:
            raise ValueError(
                "Pass target_column, targets, or both, so that every called guide "
                "has a target"
            )
        if not isinstance(policy, str) or policy not in POLICIES:
            raise ValueError(
                f"policy must be 'single' or 'combinations', not {policy!r}"
            )
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        status = require_complete_artifact(self.zw, calls)
        assay = self._get_assay(str(calls.assay))
        group = artifact_group(self.zw, calls)
        guide_ids = np.asarray(as_zarr_array(group["guides"], name="guides")[:]).astype(
            str
        )
        column = None
        if target_column is not None:
            if target_column not in assay.feats.columns:
                raise ValueError(
                    f"target_column {target_column!r} is not a feature column of "
                    f"assay {assay.name!r}"
                )
            column = np.asarray(assay.feats.fetch_all(target_column), dtype=object)
        resolved = guide_targets(
            guide_ids, column, targets, control_names, policy=policy
        )
        arguments = PerturbationLabelArguments(
            guide_calls=calls,
            targets_fingerprint=fingerprint_strings(resolved.guide_names().astype(str)),
            controls=control_names,
            policy=policy,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        codes = as_zarr_array(group["guide_codes"], name="guide_codes")
        n_cells, width = (int(size) for size in codes.shape)
        n_names = len(resolved.names)
        n_guide_ids = len(guide_ids)

        def has_targets(_ref: ArtifactRef, stored: zarr.Group) -> bool:
            names = stored.get("targets")
            guide_codes = stored.get("guide_target_codes")
            return (
                isinstance(names, zarr.Array)
                and names.shape == (n_names,)
                and isinstance(guide_codes, zarr.Array)
                and guide_codes.shape == (n_guide_ids,)
            )

        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=status.input_ref("cell_selection"),
            arrays={
                "values": ((n_cells,), None),
                "guide_label": ((n_cells,), None),
                "is_control": ((n_cells,), "b"),
                "n_targets": ((n_cells,), "i"),
                "target_codes": ((n_cells, None), "i"),
            },
            invalidate_cache=invalidate_cache,
            reuse_validator=has_targets,
        )
        if planned.reused:
            return planned.ref
        # The calls as int32, their targets and two sorted copies as int64, and
        # each cell's two labels.
        needed = n_cells * (28 * width + 160)
        if needed > self.memoryBytes:
            raise MemoryError(
                f"Labelling {n_cells} cells holds about {needed} bytes, more than "
                f"the memory budget of {self.memoryBytes} bytes. Raise mem_budget."
            )
        guide_codes = np.asarray(codes[:])
        called = np.unique(guide_codes[guide_codes >= 0])
        untargeted = called[resolved.guide_codes[called] < 0]
        if len(untargeted):
            raise ValueError(
                f"{len(untargeted)} called guides have no target, for example "
                + ", ".join(repr(str(guide)) for guide in guide_ids[untargeted[:5]])
                + "; pass targets or choose a target_column that names them"
            )
        self._require_writable(arguments.operation)
        arrays = perturbation_label_arrays(guide_codes, resolved, policy=policy)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": arrays.values,
                "guide_label": np.asarray(self._artifacts_load_values(calls).values),
                "is_control": arrays.is_control,
                "n_targets": arrays.n_targets,
                "target_codes": arrays.target_codes,
                "targets": resolved.names,
                "target_is_control": resolved.is_control,
                "guide_target_codes": resolved.guide_codes,
            },
        )
        logger.info(
            f"Labelled {n_cells} cells: "
            f"{int(np.count_nonzero(arrays.n_targets == 1))} with one gene, "
            f"{int(np.count_nonzero(arrays.n_targets > 1))} with several, "
            f"{int(np.count_nonzero(arrays.is_control))} controls"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_labels(self, labels: ArtifactRef) -> PerturbationLabels:
        """Read saved perturbation labels as tables of cells, targets and guides.

        Args:
            labels: A ``perturbation_labels`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.labels`.

        Returns:
            One row per cell, one row per cell and gene its guides target, one
            row per guide with its target, and the controls and policy.

        Raises:
            TypeError: If ``labels`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_labels`` artifact.
            MemoryError: If its values do not fit the datastore memory budget.
        """
        if not isinstance(labels, ArtifactRef):
            raise TypeError("labels must be an ArtifactRef")
        if labels.kind != PerturbationLabelArguments.artifact_kind:
            raise ValueError(
                f"labels must be a perturbation_labels artifact, not {labels.kind!r}"
            )
        status = require_complete_artifact(self.zw, labels)
        group = artifact_group(self.zw, labels)
        calls = artifact_group(self.zw, status.input_ref("guide_calls"))

        def read(stored: zarr.Group, name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(stored[name], name=name)[:])

        loaded = {
            name: self._artifacts_load_values(
                labels, value=None if name == "values" else name
            )
            for name in (
                "values",
                "guide_label",
                "is_control",
                "n_targets",
                "target_codes",
            )
        }
        cells, targets, guides = perturbation_label_frames(
            loaded["values"].cell_ids,
            values=np.asarray(loaded["values"].values),
            guide_label=np.asarray(loaded["guide_label"].values),
            is_control=np.asarray(loaded["is_control"].values),
            n_targets=np.asarray(loaded["n_targets"].values),
            target_codes=np.asarray(loaded["target_codes"].values),
            target_names=read(group, "targets").astype(str),
            target_is_control=read(group, "target_is_control"),
            guide_ids=read(calls, "guides").astype(str),
            guide_target_codes=read(group, "guide_target_codes"),
        )
        parameters = status.parameters or {}
        return PerturbationLabels(
            cells=cells,
            targets=targets,
            guides=guides,
            controls=tuple(str(name) for name in parameters["controls"]),
            policy=str(parameters["policy"]),
        )

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
        ``"no_feature"`` and named in a warning.

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
                the labels have no control cell.
            PermissionError: If the store is read-only and the table is not
                saved yet.
            MemoryError: If the target genes of the compared cells do not fit
                the memory budget.
        """
        from ...assay.normalization import feature_value_blocks
        from ...metadata.selection import NormalizationSpec

        if not isinstance(labels, ArtifactRef):
            raise TypeError("labels must be an ArtifactRef")
        if labels.kind != PerturbationLabelArguments.artifact_kind:
            raise ValueError(
                f"labels must be a perturbation_labels artifact, not {labels.kind!r}"
            )
        if not isinstance(from_assay, str) or from_assay not in self.assay_names:
            raise ValueError(f"from_assay must name an assay, not {from_assay!r}")
        if not isinstance(by, str) or by not in ("target", "guide"):
            raise ValueError(f"by must be 'target' or 'guide', not {by!r}")
        if id_column is not None and (not isinstance(id_column, str) or not id_column):
            raise TypeError("id_column must be a feature column name or None")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        status = require_complete_artifact(self.zw, labels)
        guide_assay = self._get_assay(str(labels.assay))
        expression = self._get_assay(from_assay)
        label_group = artifact_group(self.zw, labels)
        calls_group = artifact_group(self.zw, status.input_ref("guide_calls"))

        def read(stored: zarr.Group, name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(stored[name], name=name)[:])

        target_names = read(label_group, "targets").astype(str)
        target_is_control = read(label_group, "target_is_control").astype(bool)
        guide_targets_of = read(label_group, "guide_target_codes").astype(np.int64)
        guide_ids = read(calls_group, "guides").astype(str)
        if by == "target":
            row_codes = np.flatnonzero(~target_is_control)
            row_names = target_names[row_codes]
        else:
            guided = guide_targets_of >= 0
            row_guides = np.flatnonzero(
                guided & ~target_is_control[np.maximum(guide_targets_of, 0)]
            )
            row_codes = guide_targets_of[row_guides]
            row_names = guide_ids[row_guides]
        column = None
        if id_column is not None:
            if id_column not in guide_assay.feats.columns:
                raise ValueError(
                    f"id_column {id_column!r} is not a feature column of assay "
                    f"{guide_assay.name!r}"
                )
            column = np.asarray(guide_assay.feats.fetch_all(id_column), dtype=object)
        self._require_measured_cells(
            expression.name,
            status.input_ref("cell_selection"),
            operation=PerturbationEfficiencyArguments.operation,
        )
        feature_ids = np.asarray(expression.feats.fetch_all("ids")).astype(str)
        feature_names = np.asarray(expression.feats.fetch_all("names")).astype(str)
        row_features = target_features(
            target_names, guide_targets_of, column, feature_ids, feature_names
        )[row_codes]
        row_feature_ids = np.where(
            row_features >= 0, feature_ids[np.maximum(row_features, 0)], ""
        )
        arguments = PerturbationEfficiencyArguments(
            labels=labels,
            dataset_fingerprint=self._ensure_dataset_fingerprint(expression.name),
            target_features_fingerprint=fingerprint_strings(
                np.char.add(np.char.add(row_names.astype(str), "\t"), row_feature_ids)
            ),
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
            return planned.ref
        n_targets = read(label_group, "n_targets")
        target_codes = read(label_group, "target_codes")
        control = read(label_group, "is_control").astype(bool)
        if by == "target":
            row_of = np.full(len(target_names), -1, dtype=np.int64)
            row_of[row_codes] = np.arange(n_rows)
            groups = np.where(
                n_targets == 1, row_of[np.maximum(target_codes[:, 0], 0)], -1
            )
        else:
            row_of_guide = pd.Index(row_names)
            single = np.asarray(
                self._artifacts_load_values(labels, value="guide_label").values
            ).astype(str)
            groups = row_of_guide.get_indexer(single).astype(np.int64)
        if not control.any():
            raise ValueError(
                "The labels have no control cell; label cells with controls that "
                "the guides target"
            )
        compared = np.flatnonzero((groups >= 0) | control)
        cell_index = read_stored_selection_indices(
            self.zw,
            status.input_ref("cell_selection"),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)[compared]
        needed_features = np.unique(row_features[row_features >= 0])
        resident = len(compared) * (16 * len(needed_features) + 40)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Comparing {len(needed_features)} target genes over "
                f"{len(compared)} cells holds about {resident} bytes, more than "
                f"the memory budget of {self.memoryBytes} bytes. Raise mem_budget."
            )
        self._require_writable(arguments.operation)
        if len(needed_features):
            _, blocks = feature_value_blocks(
                expression,
                needed_features,
                cell_index,
                NormalizationSpec(source="raw"),
                nthreads=self.nthreads,
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
                "targets": target_names[row_codes],
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
        unmatched = row_names[row_features < 0]
        if len(unmatched):
            warn(
                f"{len(unmatched)} of {n_rows} rows have no feature of assay "
                f"{expression.name!r} for their target, for example "
                + ", ".join(repr(str(name)) for name in unmatched[:5])
                + ". Check id_column, or the target names against the feature names."
            )
        logger.info(
            f"Measured the efficiency of {int(np.count_nonzero(arrays.status == 0))} "
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
        if not isinstance(efficiency, ArtifactRef):
            raise TypeError("efficiency must be an ArtifactRef")
        if efficiency.kind != PerturbationEfficiencyArguments.artifact_kind:
            raise ValueError(
                "efficiency must be a perturbation_efficiency artifact, not "
                f"{efficiency.kind!r}"
            )
        require_complete_artifact(self.zw, efficiency)
        group = artifact_group(self.zw, efficiency)

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        by = str(group.attrs["by"])
        features = read("features").astype(str)
        columns: dict[str, Any] = {}
        if by == "guide":
            columns["guide"] = read("names").astype(str)
        columns["target"] = read("targets").astype(str)
        columns["feature"] = pd.Series(
            [None if feature == "" else feature for feature in features], dtype=object
        )
        columns["status"] = np.asarray(EFFICIENCY_STATUSES, dtype=object)[
            read("status")
        ]
        for name in ("n_cells", "n_control_cells", *_EFFICIENCY_VALUES):
            columns[name] = read(name)
        return PerturbationEfficiency(
            table=pd.DataFrame(columns),
            by=by,
            from_assay=str(efficiency.assay),
        )
