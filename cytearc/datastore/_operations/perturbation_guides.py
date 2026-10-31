import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...merge.row_plan import MERGED_ID_SEPARATOR
from ...metadata.arguments import (
    GuideCallArguments,
    GuideCallImportArguments,
    GuideThresholdArguments,
    PerturbationLabelArguments,
)
from ...metadata.artifacts import write_cell_data_artifact
from ...metadata.rows import (
    array_row_selection_parts,
    iter_metadata_column_blocks,
    read_array_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import (
    cell_selection_indices,
    cell_value_array,
    resolve_cell_aligned_artifact,
)
from ...perturbation.calling import threshold_calls, top_guide_calls
from ...perturbation.guide_calls import (
    GUIDE_ASSAY_TYPE,
    MULTIPLE,
    NEGATIVE,
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
    GuideTargets,
    PerturbationLabels,
    guide_targets,
    label_memory_bytes,
    label_settings,
    longest_label,
    perturbation_label_arrays,
    perturbation_label_frames,
)
from ...perturbation.histograms import (
    GuideCountHistograms,
    HistogramMemoryError,
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
    _column_read_bytes,
    _plan_cells,
    _read_chunkwise,
    _require_budget,
    _require_cell_column,
    _require_ref,
    _require_status_names,
    _stored_samples,
    _stream_bytes,
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
# artifact's attributes and in the messages about them.
_UNMATCHED_EXAMPLES = 5
# The bytes of the Python objects that a call holds beside its arrays: the
# guide IDs, the plans of its artifacts, and the store's metadata it opens.
_CALL_FIXED_BYTES = 1 << 18
# The bytes that reading a cell selection holds: a stored chunk, and for each
# cell of the store its flag and position.
_SELECTION_READ_BYTES = 1 << 18
_SELECTION_BYTES_PER_CELL = 32
# The bytes of each guide ID as a Python text, beside its characters.
_GUIDE_ID_BYTES = 64
# The bytes of each selected cell's sample label while the labels are read
# and numbered, beside the label's stored bytes.
_SAMPLE_BYTES_PER_CELL = 160
# The bytes that each selected cell holds while its calls are made and saved,
# beside its padded calls and the text of its labels: its position and sample
# code, its top guide, top UMIs and total UMIs, its number of calls and their
# count while they are placed, its label and top guide as Python objects, its
# top fraction, and the masks and gathers that label it.
_BYTES_PER_CELL = 120
# The bytes of each call: its cell, guide and UMIs, and their copies when the
# calls of the blocks are joined.
_BYTES_PER_CALL = 40
# The bytes that saving each label array holds per character of its width
# and cell: NumPy's two fixed-width text copies, at 4 bytes per character.
_TEXT_BYTES_PER_CHARACTER = 8
# Zarr encodes an artifact array a chunk of this many cells at a time.
_WRITE_CHUNK_CELLS = 100_000
# The bytes that summarising the guide counts makes of each row of a block:
# its total, its top guide and its top UMIs, and their masked copies.
_SUMMARY_BYTES_PER_ROW = 48
# The bytes that each selected cell holds while the counts stream to
# summarise its guides: its position, number of calls, label and summary.
_STREAMED_BYTES_PER_CELL = 40
# The bytes that each cell holds while the counts stream to call its guides:
# its position, sample code and summary; and the bytes of each call made: its
# cell, guide and UMIs.
_STREAMED_CALL_BYTES_PER_CELL = 40
_STREAMED_BYTES_PER_CALL = 20
# The bytes of each selected cell's ID beside its fixed-width text, while the
# file's barcodes are matched to the cells: its Python text and the index
# that finds it.
_ID_BYTES_PER_CELL = 120
# The bytes of each barcode of the file while the barcodes are matched: its
# position among the cells and the lookup's copies.
_MATCH_BYTES_PER_BARCODE = 40
# The bytes of each barcode with its prefix as Python text, beside its
# characters, and its pointers.
_PREFIXED_BYTES_PER_BARCODE = 80
# The store's cell IDs searched for the barcodes that match no selected cell,
# a block at a time, and the bytes of each such barcode and of each searched
# ID beside its fixed-width text.
_SEARCH_BLOCK_IDS = 1 << 14
_UNMATCHED_BYTES_PER_BARCODE = 120
_SEARCHED_BYTES_PER_ID = 120
# The bytes of each call of the file while its cells are matched and its calls
# gathered for padding: its cell, whether it matched, and the gathered cell,
# guide, UMIs and uint32 UMIs.
_CALL_INPUT_BYTES_PER_CALL = 48


def _report_barcodes(attributes: Mapping[str, Any]) -> None:
    """Warn about the call file's barcodes that match no cell, and log those outside the selection."""
    message = _unmatched_barcodes_warning(attributes)
    if message is not None:
        warn(message)
    message = _unselected_barcodes_message(attributes)
    if message is not None:
        logger.info(message)


def _import_bytes(
    size: Any,
    *,
    n_cells: int,
    guide_ids: np.ndarray,
    id_array: Any,
    prefix_length: int,
    stream_least: int,
    reading: int,
) -> int:
    """Return the most bytes that importing Cell Ranger's calls holds.

    The import reads the selected cells' IDs, parses the call file, matches
    its barcodes to the cells and searches the store's other cells for the
    barcodes that match none, pads the calls, streams the counts to summarise
    each cell's guides, and saves the calls; this returns the most that any of
    these steps holds with what the earlier ones keep.

    Args:
        size: The :class:`~cytearc.readers.cellranger.ProtospacerCallsSize`
            of the call file.
        n_cells: The selected cells.
        guide_ids: The guide IDs of the assay.
        id_array: The store's array of cell IDs.
        prefix_length: The characters of the barcode prefix.
        stream_least: What the count stream holds with one-row blocks.
        reading: What the reads before the plan hold, from
            ``_guide_read_bytes``.
    """
    n_rows = size.n_rows
    n_calls = size.n_calls
    width = max(1, min(size.most_calls, len(guide_ids)))
    fixed, per_id = array_row_selection_parts(id_array)
    id_text = n_cells * np.dtype(id_array.dtype).itemsize
    id_width = max(1, np.dtype(id_array.dtype).itemsize // 4)
    # The selected cells' IDs as fixed-width text, and their read.
    ids = id_text + fixed + per_id * n_cells
    # The barcodes with their prefix, as Python text, when there is one.
    prefixed = (
        n_rows * (_PREFIXED_BYTES_PER_BARCODE + prefix_length) + size.n_bytes
        if prefix_length
        else 0
    )
    parsed = size.parsed_bytes() + 8 * n_calls + 8 * n_rows
    parsing = ids + size.parse_bytes()
    matching = (
        ids
        + parsed
        + prefixed
        + n_cells * (_ID_BYTES_PER_CELL + id_width)
        + _MATCH_BYTES_PER_BARCODE * n_rows
    )
    # A block of the store's cell IDs at a time, beside the barcodes that match
    # no selected cell.
    searching = (
        parsed
        + prefixed
        + _UNMATCHED_BYTES_PER_BARCODE * n_rows
        + fixed
        + _SEARCH_BLOCK_IDS * (_SEARCHED_BYTES_PER_ID + 2 * id_width * 4)
    )
    padding = (
        parsed
        + _CALL_INPUT_BYTES_PER_CALL * n_calls
        + _call_bytes(n_cells, 0, width, 0)
    )
    streaming = _STREAMED_BYTES_PER_CELL * n_cells + 8 * width * n_cells + stream_least
    saving = _call_bytes(n_cells, 0, width, _label_width(guide_ids))
    return int(
        max(
            reading,
            _CALL_FIXED_BYTES
            + 8 * n_cells
            + max(parsing, matching, searching, padding, streaming, saving),
        )
    )


def _calling_bytes(
    n_cells: int, n_calls: int, width: int, text_width: int, stream_least: int
) -> int:
    """Return the most bytes that ``call_guides`` holds from its calling pass on.

    The calling pass streams the counts beside each cell's position, sample
    code and summary and the calls made so far; the calls are then padded and
    saved, as :func:`_call_bytes` counts.

    Args:
        n_cells: The cells to call.
        n_calls: The calls, or their bound.
        width: The most calls of one cell, or its bound.
        text_width: The most characters of a label.
        stream_least: What the count stream holds with one-row blocks.
    """
    streaming = (
        _CALL_FIXED_BYTES
        + _STREAMED_CALL_BYTES_PER_CELL * n_cells
        + _STREAMED_BYTES_PER_CALL * n_calls
        + stream_least
    )
    return max(streaming, _call_bytes(n_cells, n_calls, width, text_width))


def _call_bytes(n_cells: int, n_calls: int, width: int, text_width: int) -> int:
    """Return the most bytes that making, padding and saving guide calls holds.

    The count stream is not counted, since it ends before the calls are
    padded.

    Args:
        n_cells: The cells that the calls label.
        n_calls: The calls, or their bound.
        width: The padded width, the most calls of one cell.
        text_width: The most characters of a label: a guide ID, ``"Negative"``
            or ``"Multiple"``.
    """
    chunk = min(n_cells, _WRITE_CHUNK_CELLS)
    return (
        _CALL_FIXED_BYTES
        + _BYTES_PER_CELL * n_cells
        + _BYTES_PER_CALL * n_calls
        # The padded guide positions and UMIs, and Zarr's encoding of a chunk
        # of them.
        + 8 * width * (n_cells + chunk)
        + _TEXT_BYTES_PER_CHARACTER * text_width * (n_cells + chunk)
    )


def _label_width(guide_ids: np.ndarray) -> int:
    """Return the most characters of a label: a guide ID, ``"Negative"`` or ``"Multiple"``."""
    return max(len(NEGATIVE), len(MULTIPLE), *(len(str(guide)) for guide in guide_ids))


def _require_call_file_cells(
    cell_ids: np.ndarray,
    barcode_prefix: str | None,
    *,
    cells: str,
    restrict: str,
    remedy: str,
) -> None:
    """Refuse cells that the barcodes of one call file cannot reach.

    Cell Ranger writes a call file for each sample, and the import matches
    each of its barcodes to the cell whose ID is ``barcode_prefix`` followed
    by the barcode. A cell that no barcode can reach would be labelled
    ``"Negative"``. With a prefix, every cell's ID must start with it;
    without one, the cells must not come from several samples of a merged
    store, which names each cell ``<sample>__<barcode>``.

    Args:
        cell_ids: The IDs of the cells to label, as text.
        barcode_prefix: The prefix of the cell IDs, or None.
        cells: How the messages name the cells, such as ``"the run's cells"``.
        restrict: How they ask for the cells of one sample, such as
            ``"restrict cells to"``.
        remedy: The other way out that they give.

    Raises:
        ValueError: If a cell's ID does not start with ``barcode_prefix``, or
            without it, the cells come from several samples.
    """
    if barcode_prefix:
        outside = ~np.strings.startswith(cell_ids, barcode_prefix)
        count = int(np.count_nonzero(outside))
        if count:
            verb = "does" if count == 1 else "do"
            raise ValueError(
                "The call file's barcodes reach only the cells whose IDs start "
                f"with barcode_prefix {barcode_prefix!r}, but {count} of {cells}, "
                f"such as {str(cell_ids[np.argmax(outside)])!r}, {verb} not; "
                f"{restrict} the cells of that sample, {remedy}"
            )
        return
    position = np.strings.find(cell_ids, MERGED_ID_SEPARATOR)
    merged = np.flatnonzero(position >= 0)
    if not len(merged):
        return
    first = str(cell_ids[merged[0]])[: int(position[merged[0]])]
    other = merged[
        ~np.strings.startswith(cell_ids[merged], first + MERGED_ID_SEPARATOR)
    ]
    if len(other):
        second = str(cell_ids[other[0]])[: int(position[other[0]])]
        raise ValueError(
            f"The call file holds the barcodes of one sample, but {cells} come "
            f"from several samples of a merged store, such as {first!r} and "
            f"{second!r}; {restrict} the cells of one sample and pass its "
            f"barcode_prefix, such as {first + MERGED_ID_SEPARATOR!r}, {remedy}"
        )


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
    """Name the call file's barcodes that match no cell of the store, or return None.

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
        "call file match no cell of the store, for example "
        + ", ".join(
            repr(str(value)) for value in attributes["unmatched_barcode_examples"]
        )
        + ". A cell matches a barcode when its ID is barcode_prefix followed by "
        "the barcode, so check barcode_prefix when the cell IDs carry a sample "
        "prefix."
    )


def _unselected_barcodes_message(attributes: Mapping[str, Any]) -> str | None:
    """Name the call file's barcodes of cells outside the selection, or return None.

    Imported calls log this message, and so do saved calls when an import
    reuses them. Such barcodes are expected when the selection leaves out
    cells, such as those that QC filtering removed.

    Args:
        attributes: The attributes of the imported calls:
            ``n_file_barcodes``, ``n_unselected_barcodes`` and
            ``unselected_barcode_examples``.
    """
    n_unselected = int(attributes["n_unselected_barcodes"])
    if not n_unselected:
        return None
    return (
        f"{n_unselected} of {int(attributes['n_file_barcodes'])} barcodes of the "
        "call file are cells of the store outside the selected cells, for "
        "example "
        + ", ".join(
            repr(str(value)) for value in attributes["unselected_barcode_examples"]
        )
        + "; the import labels the selected cells only"
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


def _labelling_bytes(
    assay: Any,
    target_column: str | None,
    codes: zarr.Array,
    call_labels: zarr.Array,
    targets: GuideTargets,
    guide_ids: np.ndarray,
    *,
    policy: str,
) -> int:
    """Return the most bytes that labelling the cells of guide calls holds.

    Labelling reads the guide assay's target column, then the calls' guide
    codes one chunk at a time, checks that each called guide has a target,
    labels the cells, releases the codes, reads the calls' labels and writes
    the labels' arrays one at a time.

    Args:
        assay: The guide assay.
        target_column: Its feature column of targets, or None.
        codes: The calls' ``guide_codes``.
        call_labels: The calls' labels, their ``values``.
        targets: The target of every guide.
        guide_ids: The guide IDs.
        policy: ``"single"`` or ``"combinations"``.
    """
    n_cells, width = (int(size) for size in codes.shape)
    label_length = longest_label(targets, width, policy=policy)
    # The labels, from the codes' entries to each cell's label and targets.
    labelled = label_memory_bytes(
        n_cells, width, label_length=label_length, joined=policy == "combinations"
    )
    # The int32 codes, beside the chunk that a read decodes or the flags that
    # check the guides' targets.
    held_codes = 4 * n_cells * width
    reading = held_codes + max(array_row_selection_parts(codes)[0], 3 * n_cells * width)
    # The calls' labels, each a guide ID, "Negative" or "Multiple", beside the
    # chunk that their read decodes, or what saving the longest text array or
    # a chunk of the target codes holds, as saving guide calls counts it.
    guide_length = _label_width(guide_ids)
    chunk = min(n_cells, _WRITE_CHUNK_CELLS)
    n_genes = int(np.count_nonzero(~targets.is_control))
    writing = max(
        array_row_selection_parts(call_labels)[0],
        _TEXT_BYTES_PER_CHARACTER * max(label_length, guide_length) * (n_cells + chunk),
        4 * max(1, min(width, n_genes)) * chunk,
    )
    return max(
        _column_read_bytes(assay.feats, target_column),
        reading,
        held_codes + labelled,
        labelled + 4 * guide_length * n_cells + writing,
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
            ValueError: If ``from_assay`` is not an assay of the store, or is
                None and the store holds no assay declared ``CRISPR``, or
                several.
        """
        from ...assay.classification import declared_assay_type

        declared_types = {
            name: declared_assay_type(self._get_assay(name))
            for name in self.assay_names
        }
        declared_crispr = [
            name
            for name, declared in declared_types.items()
            if declared == GUIDE_ASSAY_TYPE
        ]

        def declare(name: str) -> str:
            # An assay takes its declared type when the store is opened.
            return (
                "declare the guide assay's type when opening the store, such as "
                f"DataStore(..., assay_types={{{name!r}: {GUIDE_ASSAY_TYPE!r}}})"
            )

        if from_assay is None:
            if not declared_crispr:
                # A guide assay without a preset name opens as a generic Assay.
                generic = [
                    name
                    for name, declared in declared_types.items()
                    if declared == "Assay"
                ]
                raise ValueError(
                    f"No assay of the store is declared {GUIDE_ASSAY_TYPE!r}, and guide "
                    "calls read one; "
                    + declare((generic or list(declared_types))[0])
                    + ". The store's assays are "
                    + ", ".join(
                        f"{name!r}, declared {declared!r}"
                        for name, declared in declared_types.items()
                    )
                )
            if len(declared_crispr) > 1:
                raise ValueError(
                    f"{len(declared_crispr)} assays of the store, "
                    + ", ".join(repr(name) for name in declared_crispr)
                    + f", are declared {GUIDE_ASSAY_TYPE!r}, and guide calls read one; "
                    f"name the guide assay with {argument}=..."
                )
            name = declared_crispr[0]
        elif not isinstance(from_assay, str):
            raise TypeError(f"{argument} must be an assay name or None")
        elif from_assay not in declared_types:
            raise ValueError(
                f"{argument} {from_assay!r} is not an assay of the store, whose assays "
                "are " + ", ".join(repr(name) for name in declared_types)
            )
        else:
            name = from_assay
        declared = declared_types[name]
        if declared != GUIDE_ASSAY_TYPE:
            if len(declared_crispr) == 1:
                remedy = (
                    f"; the store's assay declared {GUIDE_ASSAY_TYPE!r} is "
                    f"{declared_crispr[0]!r}"
                )
            elif declared_crispr:
                remedy = (
                    f"; the store's assays declared {GUIDE_ASSAY_TYPE!r} are "
                    + ", ".join(repr(other) for other in declared_crispr)
                )
            else:
                remedy = "; " + declare(name)
            raise TypeError(
                f"Guide calls need an assay declared with type {GUIDE_ASSAY_TYPE!r}; "
                f"{name!r} is declared as {declared!r}" + remedy
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

    def _guide_read_bytes(
        self, assay: Any, n_cells: int, sample_column: str | None = None
    ) -> int:
        """Return the most bytes that a guide operation's reads before its plan hold.

        An operation reads the guide IDs, the selected cells and, with
        ``sample_column``, their samples before it can plan; Zarr decodes a
        whole stored chunk for any of its rows.
        """
        ids = assay.feats._get_array("ids")
        fixed, per_row = array_row_selection_parts(ids)
        n_guides = int(ids.shape[0])
        reads = [
            fixed
            + (per_row + _GUIDE_ID_BYTES + np.dtype(ids.dtype).itemsize) * n_guides,
            _SELECTION_READ_BYTES + _SELECTION_BYTES_PER_CELL * int(self.cells.N),
        ]
        if sample_column is not None:
            fixed, per_row = array_row_selection_parts(
                self.cells._get_array(sample_column)
            )
            reads.append(fixed + (per_row + _SAMPLE_BYTES_PER_CELL) * n_cells)
        return int(_CALL_FIXED_BYTES + 16 * n_cells + max(reads))

    def _store_cell_ids_among(self, ids: np.ndarray) -> np.ndarray:
        """Return which of ``ids``, distinct texts, are IDs of cells of the store.

        The store's cell IDs are read a block at a time, so the search holds
        one block beside ``ids``.
        """
        found = np.zeros(len(ids), dtype=bool)
        lookup = pd.Index(ids)
        for block in iter_metadata_column_blocks(
            self.cells, "ids", block_rows=_SEARCH_BLOCK_IDS
        ):
            positions = lookup.get_indexer(np.asarray(block).astype(str))
            found[positions[positions >= 0]] = True
        return found

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
        and a cell selection of that sample; a selection whose cells the file
        cannot reach is refused, since its other cells would be labelled
        ``"Negative"``. Barcodes that match no cell of the store are counted
        in the artifact's ``n_unmatched_barcodes`` and
        ``unmatched_barcode_examples`` attributes and named in a warning, and
        barcodes of cells outside the selection, such as cells that QC
        filtering removed, in ``n_unselected_barcodes`` and
        ``unselected_barcode_examples`` and in a log message; a reused import
        reports both again.

        The memory budget is checked before the call file is parsed, from its
        rows, ``|`` separators and bytes, counted in one streamed pass: it
        must hold the parsed file, the selected cells' IDs, the padded calls,
        as wide as the most calls of one row, and the text of their labels,
        and the refusal names a budget that runs the import.

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
                gzip-compressed, whatever its name.
            cell_selection: The cells that the calls label.
            from_assay: The guide assay, declared with type ``CRISPR``, or
                None for the store's only assay declared ``CRISPR``, whatever
                its name.
            barcode_prefix: Text that precedes each barcode in the cell IDs,
                such as ``"lane1__"`` in a merged store. None, or an empty
                text, adds nothing.
            invalidate_cache: Import again instead of reusing saved calls.

        Returns:
            A complete assay-scoped ``guide_calls`` artifact. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls`
            or plot it with ``ds.plots.guide_calls``.

        Raises:
            TypeError: If an argument has the wrong type, or the assay is not
                declared ``CRISPR`` or does not store unsigned integer counts.
            ValueError: If ``from_assay`` is not an assay of the store, or
                without it, if the store holds no assay declared ``CRISPR``,
                or several.
            ValueError: If the call file is malformed, a guide ID is empty,
                repeated, or a label reserved for cells without one guide, a
                called guide is not a feature of the assay, a guide count
                exceeds the uint32 maximum, or the file's barcodes cannot
                reach every selected cell: with ``barcode_prefix``, a cell
                whose ID does not start with it, and without one, cells from
                several samples of a merged store.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            PermissionError: If the store is read-only and the calls are not
                saved yet.
            MemoryError: If the import does not fit the memory budget; the
                message names a budget that runs it.
        """
        from ...readers.cellranger import protospacer_calls_size, read_protospacer_calls

        import_settings(path, barcode_prefix)
        # An empty prefix adds nothing, as None does, and is recorded as None
        # so that the two imports reuse each other.
        barcode_prefix = barcode_prefix or None
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
                "n_unselected_barcodes",
                "unselected_barcode_examples",
            ),
        )
        if planned.reused:
            # Saved calls report their barcodes as they did when imported.
            _report_barcodes(artifact_group(self.zw, planned.ref).attrs)
            return planned.ref

        counts = assay.rawData[cell_index, :]
        needed = _import_bytes(
            protospacer_calls_size(path),
            n_cells=n_cells,
            guide_ids=guide_ids,
            id_array=self.cells._get_array("ids"),
            prefix_length=len(barcode_prefix or ""),
            stream_least=_stream_bytes(counts, _SUMMARY_BYTES_PER_ROW)[0],
            reading=self._guide_read_bytes(assay, n_cells),
        )
        _require_budget(
            needed,
            self.memoryBytes,
            f"Importing the guide calls of {n_cells} cells holds about",
            beside=" with the parsed call file and one row of counts read at a time",
            remedy=f"Raise mem_budget to at least {needed} bytes, or pass a smaller "
            "cell_selection.",
        )
        # Only the selected cells' IDs are read, a sample's in a merged store.
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index)
        ).astype(str)
        _require_call_file_cells(
            cell_ids,
            barcode_prefix,
            cells="the cells of cell_selection",
            restrict="select only",
            remedy="or call the guides from their counts with perturbation.call_guides",
        )
        file_calls = read_protospacer_calls(path)
        guide_positions = pd.Index(guide_ids).get_indexer(file_calls.features)
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
        self._require_writable(arguments.operation)
        file_ids = (
            file_calls.barcodes
            if barcode_prefix is None
            else np.asarray(
                [barcode_prefix + barcode for barcode in file_calls.barcodes],
                dtype=object,
            )
        )
        row_cells = pd.Index(cell_ids).get_indexer(file_ids)
        del cell_ids
        unmatched = np.flatnonzero(row_cells < 0)
        # A barcode that matches no selected cell may be a cell of the store
        # outside the selection, such as one that QC filtering removed.
        unselected = (
            self._store_cell_ids_among(file_ids[unmatched])
            if len(unmatched)
            else np.zeros(0, dtype=bool)
        )
        del file_ids
        # The attributes hold what the messages name, so that saved calls
        # report their barcodes again as they did.
        attributes: dict[str, Any] = {"n_file_barcodes": int(len(file_calls.barcodes))}
        for kind, rows in (
            ("unmatched", unmatched[~unselected]),
            ("unselected", unmatched[unselected]),
        ):
            attributes[f"n_{kind}_barcodes"] = int(len(rows))
            attributes[f"{kind}_barcode_examples"] = [
                str(value) for value in file_calls.barcodes[rows[:_UNMATCHED_EXAMPLES]]
            ]
        call_cells = row_cells[file_calls.rows]
        matched_calls = call_cells >= 0
        arrays = guide_call_arrays(
            call_cells[matched_calls],
            guide_positions[matched_calls],
            file_calls.umis[matched_calls],
            n_cells=n_cells,
            guide_ids=guide_ids,
        )
        del file_calls, guide_positions, row_cells, call_cells, matched_calls
        # The cells' positions, calls and labels stay beside the summaries
        # while the counts stream.
        resident = (
            _CALL_FIXED_BYTES
            + _STREAMED_BYTES_PER_CELL * n_cells
            + arrays.guide_codes.nbytes
            + arrays.guide_umis.nbytes
        )
        blocks = counts._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Summarising guide counts",
            prefetch=None,
            row_mask=None,
            resident_bytes=resident,
            memory_bytes=self.memoryBytes,
            scratch_bytes_per_row=_SUMMARY_BYTES_PER_ROW,
        )
        summary = summarize_guide_counts(blocks, n_cells, len(guide_ids))
        self._write_guide_calls(planned, arrays, summary, guide_ids, attributes)
        _report_barcodes(attributes)
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
        counts: Any,
        codes: np.ndarray,
        samples: list[str | None],
        method: str,
        guide_ids: np.ndarray,
        operation: str,
        calling_least: int,
    ) -> tuple[np.ndarray, np.ndarray, GuideCountHistograms]:
        """Count the UMIs of each guide, fit it in every sample, and save the fits.

        ``counts`` are the selected cells' guide counts, as
        ``assay.rawData[cell_index, :]``. The budget must hold the least that
        the fits hold and ``calling_least``, the least that the calls made
        from them hold, so that a budget too small for either is refused
        before the counts are read. The store must then be writable for
        ``operation``. Returns the fitted thresholds, the statuses and the
        histograms.
        """
        n_cells = len(codes)
        n_samples = len(samples)
        n_guides = len(guide_ids)
        # Each cell's position and sample code stay beside the histograms,
        # with the Python objects of the call.
        held = 16 * n_cells + _CALL_FIXED_BYTES
        histogram_least = histogram_bytes(n_samples, n_guides, n_cells, n_distinct=0)
        stream_least, stream_full = _stream_bytes(counts)
        least = max(held + histogram_least + stream_least, calling_least)
        _require_budget(
            least,
            self.memoryBytes,
            f"Fitting guide thresholds for {n_cells} cells in {n_samples} samples "
            f"and {n_guides} guides, and calling their guides, holds at least",
            beside=", with one row of counts read at a time",
            remedy=f"Raise mem_budget to at least {least} bytes, or pass a smaller "
            "cell_selection or fewer samples.",
        )
        self._require_writable(operation)
        # The histograms grow with the distinct counts that the cells hold,
        # which only reading the counts tells. The counts stream in full
        # blocks when the budget holds them beside the histograms' least, and
        # in smaller ones otherwise; the histograms take the rest, and raise
        # MemoryError before they need more.
        stream_memory = min(stream_full, self.memoryBytes - held - histogram_least)
        histogram_memory = self.memoryBytes - held - stream_memory
        blocks = counts._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Counting guide UMIs",
            prefetch=None,
            row_mask=None,
            resident_bytes=held + histogram_memory,
            memory_bytes=self.memoryBytes,
        )
        try:
            histograms = guide_count_histograms(
                blocks,
                codes,
                n_samples=n_samples,
                n_guides=n_guides,
                memory_bytes=histogram_memory,
            )
        except HistogramMemoryError as error:
            # Distinct counts grow about as fast as the rows read, or slower,
            # so while rows remain the histograms of every row are estimated
            # from those read; with the budget named, the counts stream in
            # full blocks and the histograms get at least what they needed
            # where they stopped.
            estimate = error.needed
            remain = ""
            if error.counts_remain:
                estimate = max(
                    error.needed,
                    -(-error.needed * error.n_rows // max(error.rows_read, 1)),
                )
                remain = (
                    f"; {error.n_rows - error.rows_read} of {error.n_rows} rows of "
                    "counts remain to be read, and at the rate of the rows read the "
                    f"histograms of every row need about {estimate} bytes"
                )
            raise MemoryError(
                f"The guide count histograms of {n_cells} cells need about "
                f"{error.needed} bytes for {error.n_distinct} distinct counts, more "
                f"than the {histogram_memory} bytes that the memory budget of "
                f"{self.memoryBytes} bytes leaves them beside the streamed counts"
                f"{remain}. Raise mem_budget to at least "
                f"{held + stream_full + estimate} bytes."
            ) from error
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
        self, thresholds: ArtifactRef, *, n_cells: int, calling_least: int
    ) -> tuple[np.ndarray, np.ndarray, GuideCountHistograms, list[str | None]]:
        """Read the thresholds, statuses, histograms and samples of saved fits.

        The budget must hold the read beside each of the ``n_cells`` cells'
        position and sample code, and ``calling_least``, the least that the
        calls made from the fits hold.
        """
        group = artifact_group(self.zw, thresholds)

        def array(name: str) -> zarr.Array:
            return as_zarr_array(group[name], name=name)

        # The thresholds, statuses and offsets as int64; each count as stored,
        # as int64, and its cells as int64; and the decode of a stored chunk.
        decoded, _ = array_row_selection_parts(array("histogram_counts"))
        needed = max(
            calling_least,
            _CALL_FIXED_BYTES
            + 16 * n_cells
            + decoded
            + 24 * int(np.prod(array("threshold").shape))
            + 20 * int(array("histogram_counts").shape[0]),
        )
        _require_budget(
            needed,
            self.memoryBytes,
            "Reading the saved guide fits and calling guides from them holds at least",
            remedy=f"Raise mem_budget to at least {needed} bytes.",
        )

        threshold = read_array(group, "threshold").astype(np.int64)
        histograms = GuideCountHistograms(
            n_cells=read_array(group, "n_cells").astype(np.int64),
            n_guides=threshold.shape[1],
            offsets=read_array(group, "histogram_offsets").astype(np.int64),
            counts=read_array(group, "histogram_counts").astype(np.int64),
            cells=read_array(group, "histogram_cells").astype(np.int64, copy=False),
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
        - ``"top_guide"`` calls each cell's top guide when the cell holds
          ``min_umi`` guide UMIs (default 5) and the top guide's share of
          them exceeds ``fraction_threshold`` (default 0.8), as the maximum
          rule of the SCEPTRE book does. A cell gets at most one call; a cell
          with fewer guide UMIs, or whose top guide holds a smaller share, is
          labelled ``"Negative"``.

        The fitted methods fit each guide in each sample of ``sample_column``
        separately and save the fits, the thresholds and each guide's
        histogram of counts as a ``guide_thresholds`` artifact, which the calls
        record as an input. ``min_umi`` raises each threshold and belongs to
        the calls, so calling again with another floor reuses the fits and
        reads the counts once. The counts are read in row blocks within the
        memory budget: once to count each guide's UMIs and once to call. The
        histograms grow with the distinct UMI counts of each guide in each
        sample, many more in a screen whose cells carry many guides, and take
        what the counts leave of the budget once they stream in full blocks.
        The calls are planned before the calling pass reads a count: each
        cell's detected guides bound its calls under ``fixed`` and the
        fitted methods, whose fits give the number of calls, and
        ``top_guide`` makes at most one call per cell. A refusal names a
        budget that gets past the point where it stopped, and says when more
        counts remained to be read.

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
            min_umi: The UMI floor: the fewest UMIs of a called guide for
                the fitted methods and ``"fixed"``, and the fewest guide UMIs
                of a called cell for ``"top_guide"``. None takes the method's
                default; ``"fixed"`` needs a value.
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
            ValueError: If ``from_assay`` is not an assay of the store, or
                without it, if the store holds no assay declared ``CRISPR``,
                or several.
            ValueError: If an argument is out of range or does not apply to
                the method, a sample column is unknown or lacks a label for a
                selected cell, or a guide ID is empty, repeated, or reserved.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            PermissionError: If the store is read-only and the calls are not
                saved yet.
            MemoryError: If the histograms or the calls do not fit the memory
                budget. The histograms raise it while the counts are read,
                before they would hold more than the counts leave them.
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
        counts = assay.rawData[cell_index, :]
        stream_least = _stream_bytes(counts)[0]
        label_width = _label_width(guide_ids)
        # The reads before the plan: the guide IDs, the cells and their samples.
        reading = self._guide_read_bytes(assay, n_cells, sample_column)
        # A threshold of at least 1 UMI calls at most a cell's detected guides,
        # which bound its calls; top_guide calls at most one guide per cell.
        if method == "top_guide":
            detected_calls, detected_width = n_cells, 1
        else:
            detected_array = self.cells._get_array(f"{assay.name}_nFeatures")
            fixed, per_row = array_row_selection_parts(detected_array)
            reading = max(
                reading,
                _CALL_FIXED_BYTES + 16 * n_cells + fixed + per_row * n_cells,
            )
            detected = np.asarray(
                read_metadata_rows_chunkwise(
                    self.cells, f"{assay.name}_nFeatures", cell_index
                )
            )
            detected_calls = int(detected.sum())
            detected_width = max(1, int(detected.max(initial=0)))
            del detected
        if thresholds is None:
            table = np.full(
                (1, n_guide_ids),
                resolved_min_umi if method == "fixed" else 0,
                dtype=np.int64,
            )
            n_calls, width = detected_calls, detected_width
        else:
            # The calls need at least their cells' arrays and the reads above.
            calling_least = max(
                reading, _calling_bytes(n_cells, 0, 1, label_width, stream_least)
            )
            if thresholds.reused:
                threshold, status, histograms, samples = self._stored_guide_fits(
                    thresholds.ref, n_cells=n_cells, calling_least=calling_least
                )
            else:
                threshold, status, histograms = self._fit_guide_thresholds(
                    assay,
                    thresholds,
                    counts=counts,
                    codes=codes,
                    samples=samples,
                    method=method,
                    guide_ids=guide_ids,
                    operation=arguments.operation,
                    calling_least=calling_least,
                )
            table = call_thresholds(threshold, resolved_min_umi)
            n_calls = histograms.n_calls(table)
            # The histograms are saved; the second pass does not need them.
            del histograms
            # Only the guides that some sample calls can be called.
            callable_guides = int(np.count_nonzero((table > 0).any(axis=0)))
            width = max(1, min(detected_width, callable_guides))
        needed = max(
            reading,
            _calling_bytes(n_cells, n_calls, width, label_width, stream_least),
        )
        _require_budget(
            needed,
            self.memoryBytes,
            f"Calling guides makes up to {n_calls} calls for {n_cells} cells, up to "
            f"{width} in one cell, and holds about",
            beside=", with one row of counts read at a time",
            remedy=f"Raise mem_budget to at least {needed} bytes, or pass a smaller "
            "cell_selection.",
        )
        self._require_writable(arguments.operation)
        blocks = counts._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Calling guides",
            prefetch=None,
            row_mask=None,
            resident_bytes=_CALL_FIXED_BYTES
            + _STREAMED_CALL_BYTES_PER_CELL * n_cells
            + _STREAMED_BYTES_PER_CALL * n_calls,
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
        del calls
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
                    f"{short} cells hold guide UMIs but no call: they hold fewer "
                    f"than {resolved_min_umi} guide UMIs, or their top guide holds "
                    f"at most {fraction:g} of them"
                )
        return planned.ref

    @validation_scoped
    def _perturbation_load_guide_thresholds(
        self,
        thresholds: ArtifactRef,
        *,
        guides: Iterable[str] | None = None,
    ) -> GuideThresholds:
        """Read the fitted thresholds of guide calls, with each guide's histogram.

        Args:
            thresholds: Guide calls of
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.call_guides`
                with ``method="poisson_gaussian"`` or ``"gaussian"``, or their
                ``guide_thresholds`` artifact. Read through the calls, the
                table also holds the smallest called count once ``min_umi``
                applies.
            guides: The IDs of the guides to read, such as a list, a NumPy
                array or the ``guides`` index of a loaded record; None reads
                every guide.

        Returns:
            One row per sample and guide with its status, threshold and fitted
            parameters, and one row per sample, guide and UMI count with its
            cells.

        Raises:
            TypeError: If ``thresholds`` is not an ArtifactRef, or ``guides``
                is text or not a collection of guide IDs.
            ValueError: If ``thresholds`` holds no fitted thresholds, or a
                guide is not a guide of the assay.
            MemoryError: If the tables do not fit the memory budget.
        """
        _require_ref(thresholds, "thresholds")
        if guides is not None:
            # Text is iterable too, but names one guide by its characters.
            if isinstance(guides, str | bytes) or not isinstance(guides, Iterable):
                raise TypeError("guides must be a sequence of guide IDs or None")
            guides = list(guides)
            if not all(isinstance(guide, str) for guide in guides):
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
                in the assay or gives an empty target, a target that is not a
                control is a reserved label or, with ``"combinations"``, holds
                ``"+"``, a control matches no target, a called guide has no
                target, or ``policy`` is unknown.
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
        call_labels = as_zarr_array(group["values"], name="values")
        needed = _labelling_bytes(
            assay, target_column, codes, call_labels, resolved, guide_ids, policy=policy
        )
        _require_budget(
            needed,
            self.memoryBytes,
            f"Labelling {n_cells} cells holds about",
            remedy=f"Raise mem_budget to at least {needed} bytes.",
        )
        guide_codes = _read_chunkwise(codes)
        # Each called guide must have a target; a padded entry reads the last
        # guide's flag, which the call flags drop.
        called = guide_codes >= 0
        lacking = (resolved.guide_codes < 0)[guide_codes]
        lacking &= called
        del called
        untargeted = np.unique(guide_codes[lacking])
        del lacking
        if len(untargeted):
            raise ValueError(
                f"{len(untargeted)} called guides have no target, for example "
                + ", ".join(repr(str(guide)) for guide in guide_ids[untargeted[:5]])
                + "; pass targets or choose a target_column that names them"
            )
        self._require_writable(arguments.operation)
        arrays = perturbation_label_arrays(guide_codes, resolved, policy=policy)
        del guide_codes
        # The labels of the calls, which hold one label per cell of the calls.
        guide_label = _read_chunkwise(call_labels)
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
