import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    GuideCallArguments,
    GuideCallImportArguments,
    GuideThresholdArguments,
    PerturbationLabelArguments,
)
from ...metadata.artifacts import write_cell_data_artifact
from ...metadata.rows import read_array_rows_chunkwise, read_metadata_rows_chunkwise
from ...metadata.selection import (
    cell_selection_indices,
    cell_value_array,
    resolve_cell_aligned_artifact,
)
from ...perturbation.calling import threshold_calls, top_guide_calls
from ...perturbation.guide_calls import (
    GUIDE_ASSAY_TYPE,
    GuideCallArrays,
    GuideCalls,
    GuideCountSummary,
    guide_call_arrays,
    guide_calls_frames,
    import_settings,
    summarize_guide_counts,
    validated_guide_ids,
)
from ...perturbation.guide_thresholds import (
    FITTED_METHODS,
    METHOD_PARAMETERS,
    GuideThresholds,
    call_settings,
    call_thresholds,
    guide_threshold_frames,
)
from ...perturbation.labels import (
    PerturbationLabels,
    guide_targets,
    label_settings,
    perturbation_label_arrays,
    perturbation_label_frames,
)
from ...perturbation.histograms import (
    GuideCountHistograms,
    guide_count_histograms,
    histogram_bytes,
)
from ...perturbation.mixtures import (
    FIT_STATUSES,
    GUIDE_FIT_FITTED,
    GUIDE_FIT_NO_CELLS,
    GUIDE_FIT_NO_FIT,
    GUIDE_FIT_NO_POSITIVE,
    GUIDE_FIT_TOO_SPARSE,
    MAX_ITERATIONS,
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
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.numba import numba_threads
from ...utils.warnings import warn
from .perturbation_store import (
    _cell_samples,
    _cell_values,
    _plan_cells,
    _require_budget,
    _require_cell_column,
    _require_ref,
    _require_status_names,
    _stored_samples,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _PerturbationGuideOperationsBase
else:
    _PerturbationGuideOperationsBase = object

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
    GUIDE_FIT_NO_CELLS: "have no cell with a UMI of the guide",
    GUIDE_FIT_TOO_SPARSE: "have too few cells or counts to fit",
    GUIDE_FIT_NO_FIT: "have no admissible fit",
    GUIDE_FIT_NO_POSITIVE: "have no guide population above the background",
}
# Above this share of called cells with several guides, a gaussian fit over
# every cell has likely set thresholds among the ambient counts.
_GAUSSIAN_MULTIPLE_SHARE = 0.5


def _call_file_fingerprint(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a call file's bytes, as hexadecimal text."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _unmatched_barcodes_warning(attributes: Mapping[str, Any]) -> str | None:
    """Name the call file's barcodes that match no selected cell, or return None.

    Imported calls warn with this message, and so do saved calls when an
    import reuses them.

    Args:
        attributes: The attributes of the imported calls:
            ``n_file_barcodes``, ``n_unmatched_barcodes`` and
            ``unmatched_barcode_examples``.
    """
    n_unmatched = int(attributes["n_unmatched_barcodes"])
    if not n_unmatched:
        return None
    return (
        f"{n_unmatched} of {int(attributes['n_file_barcodes'])} barcodes of the "
        "call file match no selected cell, for example "
        + ", ".join(
            repr(str(value)) for value in attributes["unmatched_barcode_examples"]
        )
        + ". Check barcode_prefix and cell_selection."
    )


def _unfitted_warning(
    status: np.ndarray, samples: list[str | None], guide_ids: np.ndarray
) -> str | None:
    """Describe the fits that call no cell, or return None when every fit calls."""
    unfitted = status != GUIDE_FIT_FITTED
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


class _PerturbationGuideOperationsMixin(_PerturbationGuideOperationsBase):
    def _guide_assay(
        self, from_assay: str | None, *, argument: str = "from_assay"
    ) -> Any:
        """Return the guide assay after checking its declared type and counts.

        ``from_assay`` names the assay, or None takes the store's only assay
        declared ``CRISPR``, whatever its name. ``argument`` names the keyword
        that the messages ask for, such as the recipe's ``guide_assay``.

        Raises:
            TypeError: If ``from_assay`` is not a name or None, or the assay
                is not declared ``CRISPR`` or does not store unsigned counts.
            ValueError: If ``from_assay`` is None and the store holds no
                assay declared ``CRISPR``, or several.
        """
        from ...assay.classification import declared_assay_type

        if from_assay is None:
            declared_crispr = [
                name
                for name in self.assay_names
                if declared_assay_type(self._get_assay(name)) == GUIDE_ASSAY_TYPE
            ]
            if len(declared_crispr) != 1:
                found = (
                    "No assay of the store is"
                    if not declared_crispr
                    else f"{len(declared_crispr)} assays of the store, "
                    + ", ".join(repr(name) for name in declared_crispr)
                    + ", are"
                )
                raise ValueError(
                    f"{found} declared {GUIDE_ASSAY_TYPE!r}, and guide calls read "
                    f"one; name the guide assay with {argument}=..."
                )
            name = declared_crispr[0]
        elif not isinstance(from_assay, str):
            raise TypeError(f"{argument} must be an assay name or None")
        else:
            name = from_assay
        declared = (
            declared_assay_type(self._get_assay(name))
            if name in self.assay_names
            else None
        )
        if declared != GUIDE_ASSAY_TYPE:
            raise TypeError(
                f"Guide calls need an assay declared with type {GUIDE_ASSAY_TYPE!r}; "
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

    def _guide_feature_column(
        self, assay: Any, column: str | None, name: str
    ) -> np.ndarray | None:
        """Read a feature column of the guide assay, or return None without one.

        Args:
            assay: The guide assay.
            column: The feature column, such as each guide's target, or None.
            name: The argument that names the column, for the error message.

        Raises:
            ValueError: If ``column`` is not a feature column of the assay.
        """
        if column is None:
            return None
        if column not in assay.feats.columns:
            raise ValueError(
                f"{name} {column!r} is not a feature column of assay {assay.name!r}"
            )
        values: np.ndarray = np.asarray(assay.feats.fetch_all(column), dtype=object)
        return values

    def _guide_call_cells(
        self, from_assay: str | None, cell_selection: ArtifactRef, operation: str
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Return the guide assay, its guide IDs and the cells that calls label.

        Raises:
            ValueError: If ``cell_selection`` selects no cell, or a guide ID
                is empty, repeated, or a reserved label.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
        """
        assay = self._guide_assay(from_assay)
        guide_ids = validated_guide_ids(assay.feats.fetch_all("ids"))
        cell_index = cell_selection_indices(self.zw, cell_selection)
        if not len(cell_index):
            raise ValueError("cell_selection selects no cell")
        self._require_measured_cells(assay.name, cell_selection, operation=operation)
        return assay, guide_ids, cell_index

    def _plan_guide_calls(
        self,
        arguments: GuideCallArguments | GuideCallImportArguments,
        assay: str,
        n_cells: int,
        n_guides: int,
        *,
        required_attributes: tuple[str, ...] = (),
    ) -> PlannedArtifact:
        """Plan a ``guide_calls`` artifact, imported or called.

        Saved calls are reused only when they name ``n_guides`` guides.
        """

        def has_guides(_ref: ArtifactRef, group: zarr.Group) -> bool:
            guides = group.get("guides")
            return isinstance(guides, zarr.Array) and guides.shape == (n_guides,)

        return _plan_cells(
            self,
            arguments,
            assay=assay,
            cell_selection=arguments.cell_selection,
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
            invalidate_cache=arguments.invalidate_cache,
            required_attributes=required_attributes,
            reuse_validator=has_guides,
        )

    def _write_guide_calls(
        self,
        planned: PlannedArtifact,
        arrays: GuideCallArrays,
        summary: GuideCountSummary,
        guide_ids: np.ndarray,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Save each cell's calls and its top guide, top UMIs and guide UMIs.

        A cell without guide UMIs has an empty top guide.
        """
        top_guide = np.full(len(arrays.values), "", dtype=object)
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
            attributes=attributes,
        )

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
        ``unmatched_barcode_examples`` attributes, and named in a warning,
        which a reused import gives again.

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
            from_assay: The guide assay, declared with type ``CRISPR``, or
                None for the store's only assay declared ``CRISPR``, whatever
                its name.
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
            ValueError: Without ``from_assay``, if the store holds no assay
                declared ``CRISPR``, or several.
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

        import_settings(path, barcode_prefix)
        _require_ref(cell_selection, "cell_selection")
        bool_argument(invalidate_cache, "invalidate_cache")
        assay, guide_ids, cell_index = self._guide_call_cells(
            from_assay, cell_selection, GuideCallImportArguments.operation
        )
        n_cells = len(cell_index)
        arguments = GuideCallImportArguments(
            cell_selection=cell_selection,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            guide_ids_fingerprint=fingerprint_strings(guide_ids.astype(str)),
            call_file_fingerprint=_call_file_fingerprint(path),
            barcode_prefix=barcode_prefix,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_guide_calls(
            arguments,
            assay.name,
            n_cells,
            len(guide_ids),
            required_attributes=(
                "n_file_barcodes",
                "n_unmatched_barcodes",
                "unmatched_barcode_examples",
            ),
        )
        if planned.reused:
            # Saved calls warn as they did when they were imported.
            message = _unmatched_barcodes_warning(
                artifact_group(self.zw, planned.ref).attrs
            )
            if message is not None:
                warn(message)
            return planned.ref

        file_calls = read_protospacer_calls(path)
        n_calls = len(file_calls.features)
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
        resident = _BYTES_PER_CELL * n_cells + _BYTES_PER_CALL * n_calls
        _require_budget(
            resident,
            self.memoryBytes,
            f"Importing {n_calls} guide calls for {n_cells} cells holds about",
            remedy="Raise mem_budget, or pass a smaller cell_selection.",
        )
        self._require_writable(arguments.operation)
        # Only the selected cells' IDs are read, a sample's in a merged store.
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index)
        ).astype(str)
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
        summary = summarize_guide_counts(blocks, n_cells, len(guide_ids))
        # The attributes also hold what the warning names, so that saved calls
        # warn again as they did.
        attributes = {
            "n_file_barcodes": int(len(file_calls.barcodes)),
            "n_unmatched_barcodes": int(len(unmatched)),
            "unmatched_barcode_examples": [str(value) for value in examples],
        }
        self._write_guide_calls(planned, arrays, summary, guide_ids, attributes)
        message = _unmatched_barcodes_warning(attributes)
        if message is not None:
            warn(message)
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
        _require_ref(calls, "calls", GuideCallImportArguments.artifact_kind)
        status = require_complete_artifact(self.zw, calls)
        group = artifact_group(self.zw, calls)
        guide_ids = read_array(group, "guides").astype(str)
        cell_ids, arrays = self._cell_arrays(calls, _GUIDE_CALL_ARRAYS)
        cells, call_table = guide_calls_frames(cell_ids, **arrays, guide_ids=guide_ids)
        return GuideCalls(
            cells=cells,
            calls=call_table,
            guides=pd.Index(guide_ids, name="guide"),
            source=str(status.operation),
        )

    def _cell_arrays(
        self, ref: ArtifactRef, names: Sequence[str]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read per-cell arrays of a cell-aligned artifact and the cells' IDs.

        One check of the memory budget covers every array and the IDs, and
        the IDs are read once.

        Raises:
            ValueError: If a name is not a per-cell array of the artifact's
                kind, or an array does not hold one row per cell.
            MemoryError: If the arrays and the IDs do not fit the budget.
        """
        for name in names[1:]:
            cell_value_array(ref.kind, name)
        group = artifact_group(self.zw, ref)
        others = {name: as_zarr_array(group[name], name=name) for name in names[1:]}
        resolved = resolve_cell_aligned_artifact(
            self.zw,
            ref,
            value_name=names[0],
            ndim=None,
            max_bytes=self.memoryBytes,
            # The other arrays and the IDs are read at the same rows.
            caller_reads=(self.cells._get_array("ids"), *others.values()),
        )
        n_cells = len(resolved.cell_idx)
        rows = np.arange(n_cells, dtype=np.int64)
        arrays = {names[0]: np.asarray(resolved.values)}
        for name, array in others.items():
            if tuple(array.shape[:1]) != (n_cells,):
                raise ValueError(
                    f"The {ref.kind} artifact's {name!r} array must hold one row "
                    f"per cell of its cell selection, {n_cells} rows, not "
                    f"{array.shape[0] if array.shape else 0}"
                )
            arrays[name] = np.asarray(read_array_rows_chunkwise(array, rows))
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", resolved.cell_idx)
        )
        return cell_ids, arrays

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
        operation: str,
    ) -> tuple[np.ndarray, np.ndarray, GuideCountHistograms]:
        """Count the UMIs of each guide, fit it in every sample, and save the fits.

        The store must be writable for ``operation`` once the memory budget
        is checked. Returns the fitted thresholds, the statuses and the
        histograms.
        """
        n_cells = len(cell_index)
        n_samples = len(samples)
        n_guides = len(guide_ids)
        # Each cell's position and sample code stay beside the histograms.
        held = 16 * n_cells
        least = held + histogram_bytes(n_samples, n_guides, n_cells, n_distinct=0)
        _require_budget(
            least,
            self.memoryBytes,
            f"Fitting guide thresholds for {n_cells} cells in {n_samples} samples "
            f"and {n_guides} guides holds at least",
            remedy="Raise mem_budget, or pass a smaller cell_selection or fewer "
            "samples.",
        )
        self._require_writable(operation)
        # The histograms grow with the distinct counts that the cells hold,
        # which only reading the counts tells. They may hold half of the
        # memory that the cells' arrays leave, never less than their least,
        # and raise MemoryError before they need more; the counts stream
        # within the rest.
        histogram_memory = max(least - held, (self.memoryBytes - held) // 2)
        blocks = assay.rawData[cell_index, :]._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Counting guide UMIs",
            prefetch=None,
            row_mask=None,
            resident_bytes=held + histogram_memory,
            memory_bytes=self.memoryBytes,
        )
        histograms = guide_count_histograms(
            blocks,
            codes,
            n_samples=n_samples,
            n_guides=n_guides,
            memory_bytes=histogram_memory,
        )
        with numba_threads(self.nthreads):
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
        _require_budget(
            needed, self.memoryBytes, "Reading the saved guide fits holds about"
        )

        threshold = read_array(group, "threshold").astype(np.int64)
        histograms = GuideCountHistograms(
            n_cells=read_array(group, "n_cells").astype(np.int64),
            n_guides=threshold.shape[1],
            offsets=read_array(group, "histogram_offsets").astype(np.int64),
            counts=read_array(group, "histogram_counts").astype(np.int64),
            cells=read_array(group, "histogram_cells").astype(np.int64),
        )
        return (
            threshold,
            read_array(group, "status"),
            histograms,
            _stored_samples(group),
        )

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
        memory budget: once to count each guide's UMIs and once to call. The
        histograms grow with the distinct UMI counts of each guide in each
        sample, many more in a screen whose cells carry many guides, and may
        hold half of the memory budget.

        The calls are saved as a ``guide_calls`` artifact, as
        :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls`
        saves Cell Ranger's: ``values`` labels each cell with its one called
        guide, ``"Negative"`` or ``"Multiple"``, beside each cell's number of
        calls, top guide, top UMIs, share and total guide UMIs, and its called
        guides with their UMIs.

        Args:
            cell_selection: The cells to call.
            from_assay: The guide assay, declared with type ``CRISPR``, or
                None for the store's only assay declared ``CRISPR``, whatever
                its name.
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
            ValueError: Without ``from_assay``, if the store holds no assay
                declared ``CRISPR``, or several.
            ValueError: If an argument is out of range or does not apply to
                the method, a sample column is unknown or lacks a label for a
                selected cell, or a guide ID is empty, repeated, or reserved.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            PermissionError: If the store is read-only and the calls are not
                saved yet.
            MemoryError: If the histograms or the calls do not fit the memory
                budget. The histograms raise it while the counts are read,
                before they would hold more than their half.
        """
        _require_ref(cell_selection, "cell_selection")
        resolved_min_umi, fraction = call_settings(
            method, min_umi, fraction_threshold, sample_column
        )
        fitted = method in FITTED_METHODS
        _require_cell_column(self, sample_column, "sample_column")
        bool_argument(invalidate_cache, "invalidate_cache")
        assay, guide_ids, cell_index = self._guide_call_cells(
            from_assay, cell_selection, GuideCallArguments.operation
        )
        n_cells = len(cell_index)
        cell_samples = _cell_samples(
            self,
            sample_column,
            cell_index,
            reason="Guide thresholds are fitted within each sample, so every "
            "selected cell needs one",
        )
        codes, samples = cell_samples.codes, cell_samples.names
        dataset_fingerprint = self._ensure_dataset_fingerprint(assay.name)
        guide_ids_fingerprint = fingerprint_strings(guide_ids.astype(str))
        n_guide_ids = len(guide_ids)
        thresholds = (
            self._plan_guide_thresholds(
                assay=assay.name,
                cell_selection=cell_selection,
                dataset_fingerprint=dataset_fingerprint,
                guide_ids_fingerprint=guide_ids_fingerprint,
                sample_assignments_fingerprint=cell_samples.assignments,
                method=method,
                sample_columns=cell_samples.columns,
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
        planned = self._plan_guide_calls(arguments, assay.name, n_cells, n_guide_ids)
        status: np.ndarray | None = None
        if planned.reused:
            if thresholds is not None:
                fits = artifact_group(self.zw, thresholds.ref)
                status, samples = read_array(fits, "status"), _stored_samples(fits)
            stored = artifact_group(self.zw, planned.ref)
            self._warn_about_guide_calls(
                method,
                status,
                samples,
                guide_ids,
                read_array(stored, "n_guides"),
            )
            return planned.ref
        if thresholds is None:
            table = np.full(
                (1, n_guide_ids),
                resolved_min_umi if method == "fixed" else 0,
                dtype=np.int64,
            )
            # A fixed floor calls at most each cell's detected guides; only
            # the selected cells' counts are read.
            n_calls = (
                int(
                    np.asarray(
                        read_metadata_rows_chunkwise(
                            self.cells, f"{assay.name}_nFeatures", cell_index
                        )
                    ).sum()
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
                    operation=arguments.operation,
                )
            table = call_thresholds(threshold, resolved_min_umi)
            n_calls = histograms.n_calls(table)
            # The histograms are saved; the second pass does not need them.
            del histograms
        resident = _BYTES_PER_CELL * n_cells + _BYTES_PER_CALL * n_calls
        _require_budget(
            resident,
            self.memoryBytes,
            f"Calling guides makes {n_calls} calls for {n_cells} cells and holds about",
            remedy="Raise mem_budget, or pass a smaller cell_selection.",
        )
        self._require_writable(arguments.operation)
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
        self._write_guide_calls(planned, arrays, summary, guide_ids)
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
        _require_ref(thresholds, "thresholds")
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
        guide_ids = read_array(group, "guides").astype(str)
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
        offsets = read_array(group, "histogram_offsets").astype(np.int64)
        samples = _stored_samples(group)
        n_samples = len(samples)
        entries = (
            np.arange(n_samples)[:, None] * len(guide_ids) + positions[None, :]
        ).reshape(-1)
        sizes = offsets[entries + 1] - offsets[entries]
        needed = 40 * int(sizes.sum()) + 160 * len(entries)
        _require_budget(
            needed,
            self.memoryBytes,
            f"Reading the thresholds of {len(positions)} guides in {n_samples} "
            "samples holds about",
            remedy="Pass guides to read fewer.",
        )
        if guides is None:
            counts = read_array(group, "histogram_counts")
            cells = read_array(group, "histogram_cells")
        else:
            counts_array = as_zarr_array(
                group["histogram_counts"], name="histogram_counts"
            )
            cells_array = as_zarr_array(
                group["histogram_cells"], name="histogram_cells"
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
        # The frames decode the status codes by the names of FIT_STATUSES,
        # which must be the stored ones.
        _require_status_names(group, FIT_STATUSES)
        table, histograms = guide_threshold_frames(
            samples=samples,
            guides=guide_ids[positions],
            n_cells=read_array(group, "n_cells"),
            status=read_array(group, "status")[:, positions],
            threshold=read_array(group, "threshold")[:, positions],
            n_umi_cells=read_array(group, "n_umi_cells")[:, positions],
            max_umi=read_array(group, "max_umi")[:, positions],
            iterations=read_array(group, "iterations")[:, positions],
            parameters={
                name: read_array(group, name)[:, positions]
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
        _require_ref(calls, "calls", GuideCallImportArguments.artifact_kind)
        control_names = label_settings(controls, target_column, targets, policy)
        bool_argument(invalidate_cache, "invalidate_cache")
        status = require_complete_artifact(self.zw, calls)
        assay = self._get_assay(str(calls.assay))
        group = artifact_group(self.zw, calls)
        guide_ids = read_array(group, "guides").astype(str)
        column = self._guide_feature_column(assay, target_column, "target_column")
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

        planned = _plan_cells(
            self,
            arguments,
            assay=assay.name,
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
        _require_budget(
            needed, self.memoryBytes, f"Labelling {n_cells} cells holds about"
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
        guide_label = _cell_values(self, calls)
        self._require_writable(arguments.operation)
        arrays = perturbation_label_arrays(guide_codes, resolved, policy=policy)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": arrays.values,
                "guide_label": guide_label,
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
        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        status = require_complete_artifact(self.zw, labels)
        group = artifact_group(self.zw, labels)
        calls = artifact_group(self.zw, status.input_ref("guide_calls"))

        cell_ids, loaded = self._cell_arrays(
            labels, ("values", "guide_label", "is_control", "n_targets", "target_codes")
        )
        cells, targets, guides = perturbation_label_frames(
            cell_ids,
            **loaded,
            target_names=read_array(group, "targets").astype(str),
            target_is_control=read_array(group, "target_is_control"),
            guide_ids=read_array(calls, "guides").astype(str),
            guide_target_codes=read_array(group, "guide_target_codes"),
        )
        parameters = status.parameters or {}
        return PerturbationLabels(
            cells=cells,
            targets=targets,
            guides=guides,
            controls=tuple(str(name) for name in parameters["controls"]),
            policy=str(parameters["policy"]),
        )
