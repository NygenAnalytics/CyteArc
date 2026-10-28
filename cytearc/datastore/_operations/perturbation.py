import hashlib
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any


import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    GuideCallArguments,
    GuideCallImportArguments,
    GuideThresholdArguments,
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
