import hashlib
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    GuideCallArguments,
    GuideCallImportArguments,
    GuideThresholdArguments,
    MixscapeArguments,
    PerturbationEDistanceArguments,
    PerturbationEfficiencyArguments,
    PerturbationLabelArguments,
    PerturbationSignatureArguments,
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
from ...perturbation.edistance import (
    EDISTANCE_STATUSES,
    PerturbationEDistance,
    edistance_arrays,
    edistance_memory_bytes,
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
from ...perturbation.mixscape import MixscapeResult
from ...perturbation.signature import PerturbationSignature
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
    ArtifactStatus,
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
# The Mixscape class of a cell of the signature that the labels do not cover.
_UNLABELLED = "unlabelled"
# The most bytes of normalized values that one band of Mixscape's features
# holds, whatever the budget, so that a large budget does not fill memory.
_MIXSCAPE_BAND_BYTES = 1 << 29
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


@dataclass(frozen=True, slots=True)
class _LabelRows:
    """The rows of a per-perturbation table, read from perturbation labels.

    Attributes:
        status: The labels' artifact status.
        target_names: Every target of the labels, control targets included.
        guide_targets: Each guide's position in ``target_names``, -1 without
            a target.
        codes: Each row's target, as a position in ``target_names``.
        names: Each row's name: the target, or the guide.
    """

    status: ArtifactStatus
    target_names: np.ndarray
    guide_targets: np.ndarray
    codes: np.ndarray
    names: np.ndarray


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


def _stored_samples(group: zarr.Group, name: str = "samples") -> list[str | None]:
    """Return the sample labels saved in an attribute, None for one sample."""
    labels = group.attrs.get(name)
    if not isinstance(labels, list):
        raise ValueError(
            f"The {group.attrs.get('kind')} artifact has no list of samples"
        )
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

    def _label_rows(self, labels: ArtifactRef, by: str) -> _LabelRows:
        """Read the rows of a per-perturbation table: each target or each guide.

        A row is a target that is not a control, or with ``by="guide"`` a guide
        whose target is not a control, in assay order.
        """
        status = require_complete_artifact(self.zw, labels)
        group = artifact_group(self.zw, labels)
        calls = artifact_group(self.zw, status.input_ref("guide_calls"))

        def read(stored: zarr.Group, name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(stored[name], name=name)[:])

        target_names = read(group, "targets").astype(str)
        is_control = read(group, "target_is_control").astype(bool)
        guide_targets = read(group, "guide_target_codes").astype(np.int64)
        if by == "target":
            codes = np.flatnonzero(~is_control)
            names = target_names[codes]
        else:
            guides = np.flatnonzero(
                (guide_targets >= 0) & ~is_control[np.maximum(guide_targets, 0)]
            )
            codes = guide_targets[guides]
            names = read(calls, "guides").astype(str)[guides]
        return _LabelRows(
            status=status,
            target_names=target_names,
            guide_targets=guide_targets,
            codes=codes,
            names=names,
        )

    def _label_groups(
        self, labels: ArtifactRef, rows: _LabelRows, by: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the row that each labelled cell carries alone, or -1, and its control flag.

        A cell carries a target alone when its called guides target that gene
        and no other, and a guide alone when it is the cell's one called guide.

        Raises:
            ValueError: If no labelled cell is a control cell.
        """
        group = artifact_group(self.zw, labels)

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        control = read("is_control").astype(bool)
        if not control.any():
            raise ValueError(
                "The labels have no control cell; label cells with controls that "
                "the guides target"
            )
        if by == "target":
            row_of = np.full(len(rows.target_names), -1, dtype=np.int64)
            row_of[rows.codes] = np.arange(len(rows.codes))
            single = read("n_targets") == 1
            first = read("target_codes")[:, 0]
            return np.where(single, row_of[np.maximum(first, 0)], -1), control
        guide_labels = np.asarray(
            self._artifacts_load_values(labels, value="guide_label").values
        ).astype(str)
        return pd.Index(rows.names).get_indexer(guide_labels).astype(np.int64), control

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
        rows = self._label_rows(labels, by)
        status = rows.status
        target_names, guide_targets_of = rows.target_names, rows.guide_targets
        row_codes, row_names = rows.codes, rows.names
        guide_assay = self._get_assay(str(labels.assay))
        expression = self._get_assay(from_assay)
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
        groups, control = self._label_groups(labels, rows, by)
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
        from ...graph.feature_projection import (
            NATIVE_COORDINATE_KINDS,
            resolve_coordinate_inputs,
        )

        if not isinstance(labels, ArtifactRef):
            raise TypeError("labels must be an ArtifactRef")
        if labels.kind != PerturbationLabelArguments.artifact_kind:
            raise ValueError(
                f"labels must be a perturbation_labels artifact, not {labels.kind!r}"
            )
        if not isinstance(coordinates, ArtifactRef):
            raise TypeError("coordinates must be an ArtifactRef")
        if coordinates.kind not in NATIVE_COORDINATE_KINDS:
            raise ValueError(
                "coordinates must be a reduction, batch_correction, "
                f"imported_coordinates or normalized artifact, not {coordinates.kind!r}"
            )
        if not isinstance(by, str) or by not in ("target", "guide"):
            raise ValueError(f"by must be 'target' or 'guide', not {by!r}")
        permutations = integer_argument(
            n_permutations, "n_permutations", minimum=0, maximum=1_000_000
        )
        seed = integer_argument(random_seed, "random_seed", minimum=0)
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        rows = self._label_rows(labels, by)
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
        groups, control = self._label_groups(labels, rows, by)

        def indices(selection: ArtifactRef) -> np.ndarray:
            return read_stored_selection_indices(
                self.zw,
                selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)

        labelled = indices(rows.status.input_ref("cell_selection"))
        located = indices(inputs.cell_selection)
        compared = np.flatnonzero((groups >= 0) | control)
        positions = np.searchsorted(located, labelled[compared])
        present = positions < len(located)
        present[present] = located[positions[present]] == labelled[compared][present]
        if not present.all():
            logger.info(
                f"{int(np.count_nonzero(~present))} of {len(compared)} compared cells "
                "have no coordinates and are left out"
            )
        kept = compared[present]
        n_control = int(np.count_nonzero(control[kept]))
        if n_control < 2:
            raise ValueError(
                f"E-distances need at least two control cells with coordinates, "
                f"and {n_control} of the labels' control cells have them"
            )
        data = as_zarr_array(artifact_group(self.zw, coordinates)["data"], name="data")
        n_dims = int(data.shape[1])
        resident = edistance_memory_bytes(
            len(kept), n_dims, n_control, nthreads=self.nthreads
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"The E-distances of {len(kept)} cells in {n_dims} dimensions "
                f"hold about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        self._require_writable(arguments.operation)
        values = np.asarray(data.oindex[positions[present], :], dtype=np.float64)
        arrays = edistance_arrays(
            values,
            groups[kept],
            control[kept],
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
            f"Measured the E-distances of {int(np.count_nonzero(arrays.status == 0))} "
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
        if not isinstance(edistance, ArtifactRef):
            raise TypeError("edistance must be an ArtifactRef")
        if edistance.kind != PerturbationEDistanceArguments.artifact_kind:
            raise ValueError(
                "edistance must be a perturbation_edistance artifact, not "
                f"{edistance.kind!r}"
            )
        status = require_complete_artifact(self.zw, edistance)
        group = artifact_group(self.zw, edistance)

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        by = str(group.attrs["by"])
        columns: dict[str, Any] = {}
        if by == "guide":
            columns["guide"] = read("names").astype(str)
        columns["target"] = read("targets").astype(str)
        columns["status"] = np.asarray(EDISTANCE_STATUSES, dtype=object)[read("status")]
        for name in (
            "n_cells",
            "n_control_cells",
            "edistance",
            "p_value",
            "adjusted_p_value",
        ):
            columns[name] = read(name)
        parameters = status.parameters or {}
        return PerturbationEDistance(
            table=pd.DataFrame(columns),
            by=by,
            n_permutations=int(parameters["n_permutations"]),
            random_seed=int(parameters["random_seed"]),
        )

    def _signature_controls(
        self, labels: ArtifactRef, cell_index: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Return which cells of ``cell_index`` the labels call controls.

        Also returns the number of labelled cells that ``cell_index`` lacks.
        """
        status = require_complete_artifact(self.zw, labels)
        labelled = read_stored_selection_indices(
            self.zw,
            status.input_ref("cell_selection"),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        flags = np.asarray(
            as_zarr_array(
                artifact_group(self.zw, labels)["is_control"], name="is_control"
            )[:],
            dtype=bool,
        )
        positions = np.searchsorted(cell_index, labelled)
        present = positions < len(cell_index)
        present[present] = cell_index[positions[present]] == labelled[present]
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
        from ...assay.normalization import feature_value_blocks
        from ...graph.feature_projection import (
            NATIVE_COORDINATE_KINDS,
            resolve_coordinate_inputs,
        )
        from ...metadata.selection import NormalizationSpec
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

        if not isinstance(labels, ArtifactRef):
            raise TypeError("labels must be an ArtifactRef")
        if labels.kind != PerturbationLabelArguments.artifact_kind:
            raise ValueError(
                f"labels must be a perturbation_labels artifact, not {labels.kind!r}"
            )
        if not isinstance(coordinates, ArtifactRef):
            raise TypeError("coordinates must be an ArtifactRef")
        if coordinates.kind not in NATIVE_COORDINATE_KINDS:
            raise ValueError(
                "coordinates must be a reduction, batch_correction, "
                f"imported_coordinates or normalized artifact, not {coordinates.kind!r}"
            )
        if features is not None:
            if not isinstance(features, ArtifactRef):
                raise TypeError("features must be an ArtifactRef or None")
            if features.kind != "feature_selection":
                raise ValueError(
                    "features must be a feature_selection artifact, not "
                    f"{features.kind!r}"
                )
        resolved_dims = integer_argument(dims, "dims", minimum=1)
        neighbors_k = integer_argument(n_neighbors, "n_neighbors", minimum=1)
        factor = float_argument(scale_factor, "scale_factor")
        if factor <= 0:
            raise ValueError("scale_factor must be above 0")
        if sample_column is not None:
            if not isinstance(sample_column, str) or not sample_column:
                raise TypeError("sample_column must be a non-empty string or None")
            if sample_column not in self.cells.columns:
                raise ValueError(
                    f"sample_column '{sample_column}' not found in cell metadata"
                )
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
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
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = len(cell_index)
        is_control, n_missing = self._signature_controls(labels, cell_index)
        columns = () if sample_column is None else (sample_column,)
        codes, sample_labels, assignments = sample_codes(
            columns,
            [self._guide_sample_labels(column, cell_index) for column in columns],
            n_cells,
            reason="Each cell takes its neighbors from the control cells of its "
            "sample, so every cell needs one",
        )
        sample_names = [
            None if sample_column is None else labels_by_column[sample_column]
            for labels_by_column in sample_labels
        ]
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
            features=features,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            sample_assignments_fingerprint=assignments,
            dims=resolved_dims,
            n_neighbors=neighbors_k,
            sample_columns=columns,
            scale_factor=factor,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        n_features = len(feature_index)

        def has_features(_ref: ArtifactRef, stored: zarr.Group) -> bool:
            stored_features = stored.get("features")
            return (
                isinstance(stored_features, zarr.Array)
                and stored_features.shape == (n_features,)
                and stored.attrs.get("sample_names") == sample_names
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
        if resident > self.memoryBytes:
            raise MemoryError(
                f"The values of {n_controls} control cells in {n_features} "
                f"features and the neighbors of {n_cells} cells hold about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget, or pass fewer features."
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
        _, control_blocks = feature_value_blocks(
            assay,
            feature_index,
            cell_index[controls],
            NormalizationSpec(source="raw"),
            nthreads=self.nthreads,
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
            _numba_threads(self.nthreads),
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
            block_rows, cell_blocks = feature_value_blocks(
                assay,
                feature_index,
                cell_index,
                NormalizationSpec(source="raw"),
                nthreads=self.nthreads,
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
        if not isinstance(signature, ArtifactRef):
            raise TypeError("signature must be an ArtifactRef")
        if signature.kind != PerturbationSignatureArguments.artifact_kind:
            raise ValueError(
                "signature must be a perturbation_signature artifact, not "
                f"{signature.kind!r}"
            )
        status = require_complete_artifact(self.zw, signature)
        group = artifact_group(self.zw, signature)
        assay = self._get_assay(str(signature.assay))

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)[
            read("features")
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
        cell_index = read_stored_selection_indices(
            self.zw,
            status.input_ref("cell_selection"),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        n_bytes = len(cell_index) * len(columns) * 4
        if n_bytes > self.memoryBytes:
            raise MemoryError(
                f"The signatures of {len(cell_index)} cells in {len(columns)} "
                f"features hold {n_bytes} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Pass fewer features, or raise mem_budget."
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
            values=pd.DataFrame(values, index=index, columns=feature_ids[columns]),
            cells=pd.DataFrame(
                {
                    "is_control": read("is_control").astype(bool),
                    "sample": np.asarray(sample_names, dtype=object)[read("samples")],
                },
                index=index,
            ),
            neighbors=pd.DataFrame(cell_ids[read("neighbors")], index=index),
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
        labelled = read_stored_selection_indices(
            self.zw,
            status.input_ref("cell_selection"),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        values = read("values").astype(str)
        single = (read("n_targets") == 1) & ~read("is_control").astype(bool)
        codes = np.where(single, read("target_codes")[:, 0], -1).astype(np.int64)
        positions = np.searchsorted(cell_index, labelled)
        present = positions < len(cell_index)
        present[present] = cell_index[positions[present]] == labelled[present]
        cell_labels = np.full(len(cell_index), _UNLABELLED, dtype=object)
        cell_labels[positions[present]] = values[present]
        cell_codes = np.full(len(cell_index), -1, dtype=np.int64)
        cell_codes[positions[present]] = codes[present]
        return cell_labels, cell_codes, read("targets").astype(str)

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
        from ...assay.normalization import feature_value_blocks
        from ...metadata.selection import NormalizationSpec
        from ...perturbation.mixscape import (
            MIXSCAPE_STATUSES,
            MixscapeFit,
            classify_target,
            group_totals,
            rank_sum_p_values,
            sorted_log_values,
        )
        from ...perturbation.signature import (
            normalize_counts_by_feature,
            signature_of_values,
        )

        if not isinstance(signature, ArtifactRef):
            raise TypeError("signature must be an ArtifactRef")
        if signature.kind != PerturbationSignatureArguments.artifact_kind:
            raise ValueError(
                "signature must be a perturbation_signature artifact, not "
                f"{signature.kind!r}"
            )
        de_floor = integer_argument(min_de_genes, "min_de_genes", minimum=1)
        cell_floor = integer_argument(min_cells, "min_cells", minimum=1)
        iterations = integer_argument(max_iterations, "max_iterations", minimum=1)
        fold = float_argument(logfc_threshold, "logfc_threshold")
        fraction = float_argument(min_fraction, "min_fraction")
        cutoff = float_argument(p_value_cutoff, "p_value_cutoff")
        if fold < 0:
            raise ValueError("logfc_threshold must be at least 0")
        if not 0 <= fraction <= 1:
            raise ValueError("min_fraction must be from 0 to 1")
        if not 0 < cutoff <= 1:
            raise ValueError("p_value_cutoff must be above 0 and at most 1")
        if not isinstance(scale, bool):
            raise TypeError("scale must be a boolean")
        if not isinstance(perturbation_type, str):
            raise TypeError("perturbation_type must be a string")
        if not perturbation_type or " " in perturbation_type:
            raise ValueError(
                "perturbation_type must be one word, such as 'KO', not "
                f"{perturbation_type!r}"
            )
        if split_column is not None:
            if not isinstance(split_column, str) or not split_column:
                raise TypeError("split_column must be a non-empty string or None")
            if split_column not in self.cells.columns:
                raise ValueError(
                    f"split_column '{split_column}' not found in cell metadata"
                )
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        source = require_complete_artifact(self.zw, signature)
        stored = artifact_group(self.zw, signature)
        assay = self._get_assay(str(signature.assay))
        scale_factor = float((source.parameters or {})["scale_factor"])
        cell_index = read_stored_selection_indices(
            self.zw,
            source.input_ref("cell_selection"),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = len(cell_index)
        cell_labels, target_codes, target_names = self._mixscape_labels(
            source.input_ref("labels"), cell_index
        )
        is_control = np.asarray(
            as_zarr_array(stored["is_control"], name="is_control")[:], dtype=bool
        )
        columns = () if split_column is None else (split_column,)
        splits, split_labels, assignments = sample_codes(
            columns,
            [self._guide_sample_labels(column, cell_index) for column in columns],
            n_cells,
            reason="Each split is classified apart, so every cell needs one",
        )
        split_names = [
            None if split_column is None else labels_by_column[split_column]
            for labels_by_column in split_labels
        ]
        arguments = MixscapeArguments(
            signature=signature,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            split_assignments_fingerprint=assignments,
            split_columns=columns,
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
        record = arguments.to_record()
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
        width = min(
            n_features,
            max(self.memoryBytes - resident, 0) // per_feature,
            max(1, _MIXSCAPE_BAND_BYTES // (8 * max(n_members, 1))),
        )
        if width < 1:
            raise MemoryError(
                f"The values of one feature over {n_members} cells hold about "
                f"{resident + per_feature} bytes with the cells' classes and "
                f"neighbors, more than the memory budget of {self.memoryBytes} "
                "bytes. Raise mem_budget."
            )
        self._require_writable(arguments.operation)
        member_totals = assay._cell_count_totals(cell_index[members])
        position_of = np.full(n_cells, -1, dtype=np.int64)
        position_of[members] = np.arange(n_members)
        # Every neighbor is a control cell, so a member.
        member_neighbors = position_of[np.asarray(stored_neighbors[:])[members]]
        signature_values = as_zarr_array(stored["values"], name="values")
        signature_features = np.asarray(
            as_zarr_array(stored["features"], name="features")[:], dtype=np.int64
        )
        signature_column = pd.Index(signature_features)
        in_signature = np.zeros(n_features, dtype=bool)
        in_signature[signature_features] = True
        normalization = NormalizationSpec(source="raw")

        def normalized(
            features: np.ndarray, cells: np.ndarray, held: int
        ) -> np.ndarray:
            """The normalized values of features over members, features x cells."""
            out = np.empty((len(features), len(cells)))
            _, blocks = feature_value_blocks(
                assay,
                features,
                cell_index[members[cells]],
                normalization,
                nthreads=self.nthreads,
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
        with _numba_threads(self.nthreads):
            for start in range(0, n_features if testable else 0, width):
                band = np.arange(start, min(start + width, n_features))
                values = normalized(
                    band, every_member, resident + per_feature * len(band)
                )
                sums, detected = group_totals(values, member_groups, n_groups)
                tested: dict[int, np.ndarray] = {}
                for row in testable:
                    pair = [n_splits + row, int(row_splits[row])]
                    counts = sizes[pair].astype(np.float64)
                    pct = np.round(detected[pair] / counts[:, None], 3)
                    change = np.log2((sums[pair[0]] + 1) / counts[0]) - np.log2(
                        (sums[pair[1]] + 1) / counts[1]
                    )
                    genes = np.flatnonzero(
                        (pct.max(axis=0) >= fraction) & (np.abs(change) >= fold)
                    )
                    if len(genes):
                        tested[row] = genes
                for split in range(n_splits):
                    split_rows = [row for row in tested if row_splits[row] == split]
                    if not split_rows:
                        continue
                    union = np.unique(
                        np.concatenate([tested[row] for row in split_rows])
                    )
                    slot_of = np.full(len(band), -1, dtype=np.int64)
                    slot_of[union] = np.arange(len(union))
                    sorted_controls, control_ties = sorted_log_values(
                        values, union, group_members(split)
                    )
                    for row in split_rows:
                        genes = tested[row]
                        p_values = rank_sum_p_values(
                            values,
                            genes,
                            group_members(n_splits + row),
                            slot_of[genes],
                            sorted_controls,
                            control_ties,
                        )
                        # Bonferroni over every feature, as FindMarkers.
                        de_parts[row].append(
                            band[genes[p_values * n_features < cutoff]]
                        )
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
            if resident + batch_bytes(*grown) > self.memoryBytes:
                raise MemoryError(
                    f"The signatures of the {n_target} cells of "
                    f"{target_names[row_targets[row]]} and of the {n_controls} "
                    f"control cells in its {len(genes)} DE genes hold about "
                    f"{resident + batch_bytes(*grown)} bytes, more than the memory "
                    f"budget of {self.memoryBytes} bytes. Raise mem_budget."
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
        with _numba_threads(self.nthreads):
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
                        max(1, _MIXSCAPE_BAND_BYTES // (8 * len(cells))),
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

        classes = np.where(is_control, "control", cell_labels).astype(object)
        global_class = classes.copy()
        posterior = np.full(n_cells, np.nan)
        score = np.full(n_cells, np.nan)
        status = np.zeros(n_rows, dtype=np.int8)
        row_values: dict[str, list[Any]] = {
            name: []
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
            )
        }
        for row in range(n_rows):
            genes = de_lists[row]
            target_cells = members[group_members(n_splits + row)]
            fit = fits[row]
            if fit is None:
                fit = classify_target(
                    np.empty((len(target_cells), 0)),
                    np.empty((int(sizes[row_splits[row]]), 0)),
                    n_de_genes=len(genes),
                    min_de_genes=de_floor,
                    min_cells=cell_floor,
                    max_iterations=iterations,
                    scale=scale,
                )
            name = str(target_names[row_targets[row]])
            classes[target_cells] = np.where(
                fit.ko, f"{name} {perturbation_type}", f"{name} NP"
            )
            global_class[target_cells] = np.where(fit.ko, perturbation_type, "NP")
            posterior[target_cells] = fit.posterior
            score[target_cells] = fit.score
            status[row] = fit.status
            for key, value in (
                ("n_cells", len(target_cells)),
                ("n_ko", int(np.count_nonzero(fit.ko))),
                ("n_de_genes", len(genes)),
                ("iterations", fit.iterations),
                ("converged", fit.converged),
                ("control_mean", fit.control_mean),
                ("control_sd", fit.control_sd),
                ("perturbed_mean", fit.perturbed_mean),
                ("perturbed_sd", fit.perturbed_sd),
                ("perturbed_weight", fit.perturbed_weight),
            ):
                row_values[key].append(value)
        de_offsets = np.concatenate(
            [[0], np.cumsum([len(genes) for genes in de_lists])]
        ).astype(np.int64)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": classes.astype(str),
                "global_class": global_class.astype(str),
                "posterior": posterior,
                "score": score,
                "splits": splits.astype(np.int32),
                "targets": target_names[row_targets].astype(str),
                "row_splits": row_splits.astype(np.int32),
                "status": status,
                "n_cells": np.asarray(row_values["n_cells"], dtype=np.int64),
                "n_ko": np.asarray(row_values["n_ko"], dtype=np.int64),
                "n_de_genes": np.asarray(row_values["n_de_genes"], dtype=np.int64),
                "iterations": np.asarray(row_values["iterations"], dtype=np.int64),
                "converged": np.asarray(row_values["converged"], dtype=bool),
                **{
                    key: np.asarray(row_values[key], dtype=np.float64)
                    for key in (
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

        if not isinstance(classes, ArtifactRef):
            raise TypeError("classes must be an ArtifactRef")
        if classes.kind != MixscapeArguments.artifact_kind:
            raise ValueError(
                f"classes must be a mixscape_classes artifact, not {classes.kind!r}"
            )
        status = require_complete_artifact(self.zw, classes)
        group = artifact_group(self.zw, classes)
        assay = self._get_assay(str(classes.assay))

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        cell_index = read_stored_selection_indices(
            self.zw,
            status.input_ref("cell_selection"),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        index = pd.Index(
            np.asarray(
                read_metadata_rows_chunkwise(self.cells, "ids", cell_index)
            ).astype(str),
            name="ids",
        )
        split_names = np.asarray(_stored_samples(group, "split_names"), dtype=object)
        cells = pd.DataFrame(
            {
                "mixscape_class": np.asarray(
                    self._artifacts_load_values(classes).values
                ).astype(str),
                "global_class": np.asarray(
                    self._artifacts_load_values(classes, value="global_class").values
                ).astype(str),
                "posterior": read("posterior"),
                "score": read("score"),
                "split": split_names[read("splits")],
            },
            index=index,
        )
        row_splits = split_names[read("row_splits")]
        targets = read("targets").astype(str)
        table: dict[str, Any] = {
            "target": targets,
            "split": row_splits,
            "status": np.asarray(MIXSCAPE_STATUSES, dtype=object)[read("status")],
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
            table[name] = read(name)
        offsets = read("de_offsets")
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
                    "feature": feature_ids[read("de_features")],
                }
            ),
            perturbation_type=str(parameters["perturbation_type"]),
            split_column=split_columns[0] if split_columns else None,
        )
