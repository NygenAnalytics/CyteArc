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
from ...metadata.artifacts import cell_data_write_bytes, write_cell_data_artifact
from ...metadata.rows import read_rows, read_text_rows, read_whole
from ...metadata.selection import (
    CellSelectionRead,
    cell_value_array,
    plan_cell_selection_read,
    read_cell_selection,
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
    PerturbationLabels,
    guide_targets,
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
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.diagnostics import DiagnosticRule, warn_diagnostics
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.memory_budget import (
    MemoryLedger,
    fixed_text_bytes,
    hash_table_bytes,
    memory_refusal,
    object_array_bytes,
    python_text_bytes,
)
from ...utils.numba import numba_threads
from ...utils.warnings import warn
from .cell_barcodes import (
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    join_bytes,
)
from .cell_barcodes import barcode_pattern as checked_barcode_pattern
from .perturbation_store import (
    _cell_samples,
    _count_stream,
    _plan_cells,
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
# The bytes of the Python objects that a guide operation holds beside its
# arrays: the plans of its artifacts and the store's metadata it opens.
_CALL_FIXED_BYTES = 1 << 18
# Checking the guide IDs holds, beside them as Python text, a list and an
# array of their references and four copies at their fixed width: their text
# as an array, and the copies and distinct values of the sort that finds a
# repeated ID.
_GUIDE_CHECK_COPIES = 4
_GUIDE_CHECK_BYTES_PER_GUIDE = 16
# Checking the cells' IDs against the call file's options holds, for each
# cell, whether it is outside the options and where its ID holds the
# separator of a merged store's samples, beside a copy of the IDs that hold
# it.
_CELL_CHECK_BYTES_PER_CELL = 18
# Summarising a cell's guide counts holds its top guide, its top UMIs and its
# total UMIs.
_SUMMARY_BYTES_PER_CELL = 4 + 4 + 8
# The bytes that summarising the guide counts makes of each row of a block:
# its total, its top guide and its top UMIs, and their masked copies.
_SUMMARY_BYTES_PER_ROW = 48
# Each call holds its cell, its guide and its UMIs.
_BYTES_PER_CALL = 8 + 8 + 4
# Matching each call of the file to its cell holds, beside its cell and
# whether it matched, its cell, guide and UMIs gathered for padding.
_MATCHED_BYTES_PER_CALL = 8 + 1
# Padding the calls holds, for each cell, its number of calls as counted and
# the labels' choice among the guides, and for each guide the cell that
# last called it and its label's reference.
_PADDING_BYTES_PER_CELL = 16
_PADDING_BYTES_PER_GUIDE = 16
# Each row of a table of calls or targets holds, beside its cell's ID, its
# cell and slot as found, ordered and gathered, its value's reference and its
# UMIs, and their copies in the table.
_FRAME_BYTES_PER_ENTRY = 128
# Reading saved fits as tables holds, for each count of a histogram, its
# entry, UMIs and cells as gathered, ordered and copied into a row with its
# sample's and guide's references, and for each fit its row, beside the
# arrays read.
_TABLE_BYTES_PER_COUNT = 128
_TABLE_BYTES_PER_FIT = 160
# Fitting a guide in a sample holds its cells with UMIs, its most UMIs, the
# fit's parameters, status and threshold as computed and as kept; the fits
# keep the status, the threshold, the parameters and the iterations.
_FIT_BYTES_PER_ENTRY = 160
_FITTED_BYTES_PER_ENTRY = 80
# Saving the calls holds, for each cell, its top guide's reference, whether
# it has one, the gather of its ID, and its top fraction with its mask.
_SAVING_BYTES_PER_CELL = 8 + 1 + 8 + 9


def _report_barcodes(
    arguments: GuideCallImportArguments,
    diagnostics: Mapping[str, Any],
    call_file: BarcodeFile,
) -> None:
    """Warn about the call file's barcodes that match no cell, and log those outside the selection.

    Imported calls report them from the diagnostics they saved, and so do
    saved calls when an import reuses them.
    """
    warn_diagnostics(
        arguments,
        diagnostics,
        (
            DiagnosticRule(
                "unmatchedBarcodes",
                applies=lambda numbers: int(numbers["unmatchedBarcodes"]) > 0,
                message=lambda numbers: _unmatched_barcodes_warning(numbers, call_file),
            ),
        ),
    )
    message = _unselected_barcodes_message(diagnostics)
    if message is not None:
        logger.info(message)


def _padded_bytes(n_cells: int, width: int) -> int:
    """Return the bytes of the per-cell arrays that :func:`guide_call_arrays` makes for calls ``width`` wide.

    Each cell's label reference and number of calls, and its padded guide
    positions and UMIs.
    """
    return n_cells * (8 + 4 + 8 * width)


def _padding_bytes(n_cells: int, width: int, n_guides: int) -> int:
    """Return the most bytes that :func:`guide_call_arrays` holds beside the calls that it pads, its result included."""
    return (
        _padded_bytes(n_cells, width)
        + _PADDING_BYTES_PER_CELL * n_cells
        + _PADDING_BYTES_PER_GUIDE * n_guides
    )


def _guide_call_write_bytes(n_cells: int, width: int, guide_ids: np.ndarray) -> int:
    """Return the most bytes that saving guide calls ``width`` wide holds beside the padded calls and the summary.

    Saving names each cell's top guide and its top fraction, and writes the
    artifact's arrays as :func:`cell_data_write_bytes` states, each label at
    the width of the longest: a guide ID, ``"Negative"`` or ``"Multiple"``.
    """
    id_width = max(1, *(len(str(guide)) for guide in guide_ids))
    return _SAVING_BYTES_PER_CELL * n_cells + cell_data_write_bytes(
        {
            "values": ((n_cells,), f"U{_label_width(guide_ids)}"),
            "n_guides": ((n_cells,), np.int32),
            "top_guide": ((n_cells,), f"U{id_width}"),
            "top_umi": ((n_cells,), np.uint32),
            "top_fraction": ((n_cells,), np.float64),
            "total_umi": ((n_cells,), np.uint64),
            "guide_codes": ((n_cells, width), np.int32),
            "guide_umis": ((n_cells, width), np.uint32),
            "guides": ((len(guide_ids),), f"U{id_width}"),
        }
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
    barcode_suffix: str | None = None,
) -> None:
    """Refuse cells that the barcodes of one call file cannot reach.

    Cell Ranger writes a call file for each sample, and the import matches
    each of its barcodes to the cell whose ID is ``barcode_prefix`` followed
    by the barcode and ``barcode_suffix``. A cell that no barcode can reach
    would be labelled ``"Negative"``. With a prefix, every cell's ID must
    start with it, and with a suffix end with it; without a prefix, the
    cells must not come from several samples of a merged store, which names
    each cell ``<sample>__<barcode>``.

    Args:
        cell_ids: The IDs of the cells to label, as text.
        barcode_prefix: The prefix of the cell IDs, or None.
        cells: How the messages name the cells, such as ``"the run's cells"``.
        restrict: How they ask for the cells of one sample, such as
            ``"restrict cells to"``.
        remedy: The other way out that they give.
        barcode_suffix: The suffix of the cell IDs, or None.

    Raises:
        ValueError: If a cell's ID does not start with ``barcode_prefix`` or
            end with ``barcode_suffix``, or without a prefix, the cells come
            from several samples.
    """
    for option, value, end, find in (
        ("barcode_prefix", barcode_prefix, "start", np.strings.startswith),
        ("barcode_suffix", barcode_suffix, "end", np.strings.endswith),
    ):
        if not value:
            continue
        outside = ~find(cell_ids, value)
        count = int(np.count_nonzero(outside))
        if count:
            verb = "does" if count == 1 else "do"
            raise ValueError(
                f"The call file's barcodes reach only the cells whose IDs {end} "
                f"with {option} {value!r}, but {count} of {cells}, such as "
                f"{str(cell_ids[np.argmax(outside)])!r}, {verb} not; {restrict} "
                f"the cells of that sample, {remedy}"
            )
    if barcode_prefix:
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


def _barcode_ids(barcodes: np.ndarray, barcode_prefix: str | None) -> np.ndarray:
    """Return the ID of the cell that each barcode of a call file names.

    A cell's ID is ``barcode_prefix`` followed by the barcode; None, or an
    empty text, adds nothing.
    """
    if not barcode_prefix:
        return barcodes
    return np.asarray([barcode_prefix + barcode for barcode in barcodes], dtype=object)


def _refuse_shared_cells(
    row_cells: np.ndarray, barcodes: np.ndarray, cell_ids: np.ndarray
) -> None:
    """Refuse a call file whose barcodes name one cell twice.

    Cell Ranger lists each barcode once, and a prefix and a suffix keep
    distinct barcodes distinct, so only a pattern that keeps the same part of
    two barcodes can name a cell twice.
    """
    matched = np.flatnonzero(row_cells >= 0)
    order = matched[np.argsort(row_cells[matched], kind="stable")]
    repeated = np.flatnonzero(row_cells[order][1:] == row_cells[order][:-1])
    if len(repeated):
        first, second = order[repeated[0]], order[repeated[0] + 1]
        raise ValueError(
            f"Barcodes {str(barcodes[first])!r} and {str(barcodes[second])!r} of "
            f"the call file name one cell, {str(cell_ids[row_cells[first]])!r}; "
            "check barcode_pattern"
        )


def _called_guide_positions(
    features: np.ndarray, guide_ids: np.ndarray, assay: Any
) -> np.ndarray:
    """Return the position of each called guide among the guide assay's guides.

    Args:
        features: The feature ID of each call of a call file.
        guide_ids: The guide IDs of the assay.
        assay: The guide assay, whose names the message compares with the
            calls.

    Raises:
        ValueError: If a called guide is not a feature of the assay; the
            message says so when the calls name the guides by name.
    """
    positions: np.ndarray = pd.Index(guide_ids).get_indexer(features)
    unknown = np.flatnonzero(positions < 0)
    if len(unknown):
        names = set(np.asarray(assay.feats.fetch_all("names")).astype(str))
        examples = pd.unique(features[unknown])[:5].tolist()
        hint = (
            "; they match guide names, so the file names guides by name "
            "while the assay's IDs differ"
            if all(example in names for example in examples)
            else ""
        )
        raise ValueError(
            f"{len(pd.unique(features[unknown]))} called guides are not features "
            f"of assay {assay.name!r}, for example "
            + ", ".join(repr(example) for example in examples)
            + hint
        )
    return positions


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


def _unmatched_barcodes_warning(
    diagnostics: Mapping[str, Any], call_file: BarcodeFile
) -> str:
    """Name the call file's barcodes that match no cell of the store.

    Imported calls warn with this message, and so do saved calls when an
    import reuses them.

    Args:
        diagnostics: The diagnostics of the imported calls: ``fileBarcodes``,
            ``unmatchedBarcodes`` and ``unmatchedBarcodeExamples``.
        call_file: The call file, whose options the message names.
    """
    if call_file.barcode_suffix is None and call_file.barcode_pattern is None:
        rule = (
            "A cell matches a barcode when its ID is barcode_prefix followed by "
            "the barcode, so check barcode_prefix when the cell IDs carry a "
            "sample prefix."
        )
    else:
        rule = (
            "A cell matches a barcode when its ID is barcode_prefix, the barcode "
            "or the part of it that barcode_pattern keeps, and barcode_suffix, so "
            "check these options against the cell IDs."
        )
    return (
        f"{int(diagnostics['unmatchedBarcodes'])} of "
        f"{int(diagnostics['fileBarcodes'])} barcodes of the call file match no "
        "cell of the store, for example "
        + ", ".join(
            repr(str(value)) for value in diagnostics["unmatchedBarcodeExamples"]
        )
        + ". "
        + rule
    )


def _unselected_barcodes_message(diagnostics: Mapping[str, Any]) -> str | None:
    """Name the call file's barcodes of cells outside the selection, or return None.

    Imported calls log this message, and so do saved calls when an import
    reuses them. Such barcodes are expected when the selection leaves out
    cells, such as those that QC filtering removed.

    Args:
        diagnostics: The diagnostics of the imported calls: ``fileBarcodes``,
            ``unselectedBarcodes`` and ``unselectedBarcodeExamples``.
    """
    n_unselected = int(diagnostics["unselectedBarcodes"])
    if not n_unselected:
        return None
    return (
        f"{n_unselected} of {int(diagnostics['fileBarcodes'])} barcodes of the "
        "call file are cells of the store outside the selected cells, for "
        "example "
        + ", ".join(
            repr(str(value)) for value in diagnostics["unselectedBarcodeExamples"]
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


def _strings_bytes(values: np.ndarray) -> int:
    """Return the bytes of fixed-width text ``values`` as Python strings and their references.

    Each string is counted at the stored width, which bounds its characters,
    and at the kind of the widest character stored.
    """
    n_values = len(values)
    width = max(1, values.dtype.itemsize // 4)
    widest = int(values.view(np.uint32).max()) if values.size else 0
    return object_array_bytes(n_values) + python_text_bytes(
        n_values, n_values * width, widest=widest
    )


def _frame_bytes(cell_ids: np.ndarray, arrays: Mapping[str, np.ndarray]) -> int:
    """Return the most bytes that the tables of stored per-cell ``arrays`` hold beside them.

    Each table holds the cells' IDs and each text array as Python strings,
    as pandas makes them, and a copy of each array of numbers; each entry of
    a padded two-dimensional array that holds a value becomes a row of its
    own, which holds its cell's ID, as fixed-width text and as a Python
    string, and the entry's values, their gathers and their copies.
    """
    total = _strings_bytes(cell_ids)
    id_width = max(1, cell_ids.dtype.itemsize // 4)
    for values in arrays.values():
        if values.ndim == 1 and values.dtype.kind == "U":
            total += object_array_bytes(len(values)) + _strings_bytes(values)
        elif values.ndim == 1:
            total += 2 * values.nbytes
        else:
            entries = int(np.count_nonzero(values >= 0))
            total += (
                entries * _FRAME_BYTES_PER_ENTRY
                + fixed_text_bytes(entries, id_width)
                + _strings_bytes(cell_ids[:1]) * entries
            )
    return total


def _read_guide_ids(ledger: MemoryLedger, array: Any) -> np.ndarray:
    """Read a stored array of guide IDs as Python text and check them, keeping them charged.

    The IDs are read through ``read_text_rows``; checking them holds what
    :func:`~cytearc.perturbation.guide_calls.validated_guide_ids` copies of
    them.

    Raises:
        MemoryBudgetError: If the read or the check would pass the budget.
        ValueError: If there is no guide, or an ID is empty, repeated, or
            reserved.
    """
    n_guides = int(array.shape[0])
    ids = read_text_rows(
        ledger, array, np.arange(n_guides, dtype=np.int64), item="the guide IDs"
    )
    width = max((len(value) for value in ids), default=1)
    ledger.peak(
        "checking the guide IDs",
        _GUIDE_CHECK_BYTES_PER_GUIDE * n_guides
        + _GUIDE_CHECK_COPIES * fixed_text_bytes(n_guides, max(1, width)),
    )
    return validated_guide_ids(ids)


def _check_cell_ids(
    ledger: MemoryLedger, cell_ids: np.ndarray, call_file: BarcodeFile, **words: str
) -> None:
    """Refuse cells that the call file's barcodes cannot reach, after charging the check.

    ``cell_ids`` are the cells' IDs as fixed-width text, and ``words`` the
    messages' words of :func:`_require_call_file_cells`.
    """
    ledger.peak(
        "checking the cells' IDs against the call file",
        _CELL_CHECK_BYTES_PER_CELL * len(cell_ids) + cell_ids.nbytes,
    )
    _require_call_file_cells(
        cell_ids,
        call_file.barcode_prefix,
        barcode_suffix=call_file.barcode_suffix,
        **words,
    )


def _parse_call_file(
    ledger: MemoryLedger, path: str | os.PathLike[str], size: Any
) -> Any:
    """Parse Cell Ranger's call file within the ledger, and keep what it parsed charged.

    ``size`` is the file's :class:`~cytearc.readers.cellranger.ProtospacerCallsSize`,
    whose bounds the parse and its result are charged by.
    """
    from ...readers.cellranger import read_protospacer_calls

    ledger.peak("parsing the call file", size.parse_bytes())
    calls = read_protospacer_calls(path)
    ledger.hold("the parsed call file", size.parsed_bytes())
    return calls


def _guide_positions(
    ledger: MemoryLedger, features: np.ndarray, guide_ids: np.ndarray, assay: Any
) -> np.ndarray:
    """Find each call's guide among the assay's guides, keeping the positions charged.

    The lookup is pandas' hash table of the guide IDs, beside each call's
    position.
    """
    ledger.peak(
        "finding the called guides",
        hash_table_bytes(len(guide_ids)) + 8 * len(features),
    )
    positions = _called_guide_positions(features, guide_ids, assay)
    ledger.hold("the called guides' positions", positions.nbytes)
    return positions


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
        self, assay: Any, column: str | None, name: str, *, ledger: MemoryLedger
    ) -> np.ndarray | None:
        """Read a feature column of the guide assay as Python objects, or return None without one.

        The column is read through ``ledger``, which keeps it charged.

        Args:
            assay: The guide assay.
            column: The feature column, such as each guide's target, or None.
            name: The argument that names the column, for the error message.
            ledger: The call's memory ledger.

        Raises:
            ValueError: If ``column`` is not a feature column of the assay.
            MemoryBudgetError: If the column does not fit the memory budget.
        """
        if column is None:
            return None
        if column not in assay.feats.columns:
            raise ValueError(
                f"{name} {column!r} is not a feature column of assay {assay.name!r}"
            )
        array = assay.feats._get_array(column)
        rows = np.arange(int(array.shape[0]), dtype=np.int64)
        item = f"the guides' {column!r}"
        if np.dtype(array.dtype).kind == "U":
            text: np.ndarray = read_text_rows(ledger, array, rows, item=item)
            return text
        values = read_rows(ledger, array, rows, item=item)
        # The values as Python objects beside their references.
        ledger.hold(item, object_array_bytes(len(values)) + 32 * len(values))
        stored = values.nbytes
        objects: np.ndarray = values.astype(object)
        del values
        ledger.release(stored)
        return objects

    def _guide_call_cells(
        self,
        from_assay: str | None,
        cell_selection: ArtifactRef,
        operation: str,
        *,
        doing: str,
    ) -> tuple[Any, MemoryLedger, np.ndarray, CellSelectionRead]:
        """Return the guide assay, the call's ledger, its guide IDs and the planned read of the cells that calls label.

        The ledger refuses the store's budget as ``doing``, a text of
        ``n_cells``, the selected cells, such as ``"Calling guides for
        {n_cells} cells"``, and first holds the call's Python objects and
        then the guide IDs, which it reads and checks.

        Raises:
            ValueError: If ``cell_selection`` selects no cell, or a guide ID
                is empty, repeated, or a reserved label.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            MemoryBudgetError: If the guide IDs do not fit the memory budget.
        """
        assay = self._guide_assay(from_assay)
        selection = plan_cell_selection_read(self.zw, cell_selection)
        ledger = MemoryLedger(
            self.memoryBytes,
            doing.format(n_cells=selection.cells),
            remedy="pass a smaller cell_selection",
            codec_threads=self.nthreads,
        )
        ledger.hold(
            "the call's plans and the store's metadata it opens", _CALL_FIXED_BYTES
        )
        guide_ids = _read_guide_ids(ledger, assay.feats._get_array("ids"))
        if not selection.cells:
            raise ValueError("cell_selection selects no cell")
        self._require_measured_cells(assay.name, cell_selection, operation=operation)
        return assay, ledger, guide_ids, selection

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
        ledger: MemoryLedger,
        planned: PlannedArtifact,
        arrays: GuideCallArrays,
        summary: GuideCountSummary,
        guide_ids: np.ndarray,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        """Save each cell's calls and its top guide, top UMIs and guide UMIs.

        A cell without guide UMIs has an empty top guide. Imported calls save
        the join's ``diagnostics``; called ones save none. What saving holds
        beside the calls and the summary, as :func:`_guide_call_write_bytes`
        states it, is charged to ``ledger`` first.
        """
        n_cells, width = arrays.guide_codes.shape
        ledger.peak(
            "saving the guide calls", _guide_call_write_bytes(n_cells, width, guide_ids)
        )
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
            diagnostics=diagnostics,
        )

    def _pad_calls(
        self,
        ledger: MemoryLedger,
        cells: np.ndarray,
        codes: np.ndarray,
        umis: np.ndarray,
        *,
        n_cells: int,
        width: int,
        guide_ids: np.ndarray,
    ) -> GuideCallArrays:
        """Pad calls into the per-cell arrays of a ``guide_calls`` artifact, keeping them charged.

        ``width`` bounds the most calls of one cell, which padding is
        charged by before the arrays are made.
        """
        ledger.peak(
            f"padding the calls of {n_cells} cells, up to {width} in one cell",
            _padding_bytes(n_cells, width, len(guide_ids)),
        )
        arrays = guide_call_arrays(
            cells, codes, umis, n_cells=n_cells, guide_ids=guide_ids
        )
        ledger.hold(
            "the padded calls",
            _padded_bytes(n_cells, int(arrays.guide_codes.shape[1])),
        )
        return arrays

    @validation_scoped
    def _perturbation_import_guide_calls(
        self,
        path: str | os.PathLike[str],
        *,
        cell_selection: ArtifactRef,
        from_assay: str | None = None,
        barcode_prefix: str | None = None,
        barcode_suffix: str | None = None,
        barcode_pattern: str | None = None,
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

        A cell's ID is ``barcode_prefix``, the barcode and ``barcode_suffix``,
        matched exactly; ``barcode_pattern`` keeps the part of each barcode
        that its ``barcode`` group matches first. The barcodes join the cells
        through the join that every importer of barcodes shares. A merged
        store prefixes each sample's barcodes with its name and two
        underscores, so import each sample's file with its own prefix and a
        cell selection of that sample; a selection whose cells the file cannot
        reach is refused, since its other cells would be labelled
        ``"Negative"``. The artifact's diagnostics, which
        ``ds.artifacts.inspect(calls).diagnostics`` returns, count the file's
        barcodes by the cell ID they name in ``fileBarcodes``, those that
        match no cell of the store in ``unmatchedBarcodes``, which a warning
        names, and those of cells outside the selection, such as cells that
        QC filtering removed, in ``unselectedBarcodes``, which a log message
        names; ``unmatchedBarcodeExamples`` and ``unselectedBarcodeExamples``
        show up to five of each beside the cell ID it derives. A reused import
        reports both again. Two barcodes that name one cell raise
        ``ValueError``.

        Each step is charged to one memory ledger before it allocates, from
        the file's rows, ``|`` separators and bytes, counted in one streamed
        pass before the file is parsed: the parsed file, the selected cells'
        IDs, the cell IDs that the prefix, suffix and pattern derive and the
        join's matching of them, the padded calls, as wide as the most calls
        of one row, the guide counts that stream to summarise each cell, and
        the text of the labels. A refusal names the step and the budget that
        passes it, before anything is written.

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
            barcode_suffix: Text that follows each barcode in the cell IDs,
                such as ``"-1"``. None, or an empty text, adds nothing.
            barcode_pattern: A regular expression whose named group
                ``barcode`` is the part of each barcode that names the cell,
                found as ``re.search`` finds it, such as
                ``r"^(?P<barcode>[ACGT]{16})"``; a barcode it does not match
                names no cell. None keeps the whole barcode.
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
                exceeds the uint32 maximum, ``barcode_pattern`` has no
                ``barcode`` group, two barcodes name one cell, or the file's
                barcodes cannot reach every selected cell: with
                ``barcode_prefix``, a cell whose ID does not start with it,
                with ``barcode_suffix``, a cell whose ID does not end with it,
                and without a prefix, cells from several samples of a merged
                store.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            PermissionError: If the store is read-only and the calls are not
                saved yet.
            MemoryBudgetError: If a step of the import does not fit the
                memory budget; the refusal names the step and the budget that
                passes it.
        """
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        from ...readers.cellranger import PROTOSPACER_SCAN_BYTES, protospacer_calls_size

        import_settings(path, barcode_prefix)
        if barcode_suffix is not None and not isinstance(barcode_suffix, str):
            raise TypeError("barcode_suffix must be text or None")
        barcode_pattern = checked_barcode_pattern(barcode_pattern)
        _require_ref(cell_selection, "cell_selection")
        assay, ledger, guide_ids, selection = self._guide_call_cells(
            from_assay,
            cell_selection,
            GuideCallImportArguments.operation,
            doing="Importing the guide calls of {n_cells} cells",
        )
        n_cells = selection.cells
        # An empty prefix or suffix adds nothing, as None does, and the file
        # records it as None so that the two imports reuse each other.
        (call_file,) = barcode_files(
            path,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            what="call file",
        )
        arguments = GuideCallImportArguments(
            cell_selection=cell_selection,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            guide_ids_fingerprint=fingerprint_strings(guide_ids.astype(str)),
            call_file_fingerprint=call_file.sha256,
            barcode_prefix=call_file.barcode_prefix,
            barcode_suffix=call_file.barcode_suffix,
            barcode_pattern=call_file.barcode_pattern,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_guide_calls(
            arguments,
            assay.name,
            n_cells,
            len(guide_ids),
        )
        if planned.reused:
            # Saved calls report their barcodes as they did when imported.
            _report_barcodes(
                arguments, inspect_artifact(self.zw, planned.ref).diagnostics, call_file
            )
            return planned.ref

        cell_index = read_cell_selection(
            ledger, self.zw, selection, item="the selected cells' positions"
        )
        ledger.peak("counting the call file's rows", PROTOSPACER_SCAN_BYTES)
        size = protospacer_calls_size(path)
        # Only the selected cells' IDs are read, a sample's in a merged store.
        id_array = self.cells._get_array("ids")
        cell_ids = read_rows(
            ledger, id_array, cell_index, item="the selected cells' IDs"
        )
        _check_cell_ids(
            ledger,
            cell_ids,
            call_file,
            cells="the cells of cell_selection",
            restrict="select only",
            remedy="or call the guides from their counts with perturbation.call_guides",
        )
        file_calls = _parse_call_file(ledger, path, size)
        guide_positions = _guide_positions(
            ledger, file_calls.features, guide_ids, assay
        )
        self._require_writable(arguments.operation)
        # A barcode that matches no selected cell may be a cell of the store
        # outside the selection, such as one that QC filtering removed, which
        # the join tells from a barcode of no cell by the store's cell IDs.
        # What the join and the cells' IDs hold is let go once it is done.
        joining = ledger.charged
        join = CellBarcodeJoin(cell_ids, ledger=ledger, store_ids=id_array)
        # What matching the barcodes holds stays charged while the join reads
        # the store's cell IDs to tell the barcodes of no cell apart.
        matching = ledger.hold(
            "matching the call file's barcodes",
            join_bytes(call_file, size.n_rows, size.n_bytes),
        )
        row_cells = join.cells(call_file, file_calls.barcodes)
        ledger.release(matching)
        ledger.peak("checking that no two barcodes name one cell", 32 * len(row_cells))
        _refuse_shared_cells(row_cells, file_calls.barcodes, cell_ids)
        # The diagnostics hold what the messages name, so that saved calls
        # report their barcodes again as they did.
        diagnostics = join.diagnostics()
        ids_bytes = cell_ids.nbytes
        del join, cell_ids
        ledger.release(ledger.charged - joining + ids_bytes)
        n_calls = len(file_calls.rows)
        # Each call's cell, whether it matched, and the matched calls'
        # cells, guides and UMIs, which padding takes.
        gathering = ledger.hold(
            "each call's cell",
            8 * len(row_cells) + (_MATCHED_BYTES_PER_CALL + _BYTES_PER_CALL) * n_calls,
        )
        call_cells = row_cells[file_calls.rows]
        matched_calls = call_cells >= 0
        arrays = self._pad_calls(
            ledger,
            call_cells[matched_calls],
            guide_positions[matched_calls],
            file_calls.umis[matched_calls],
            n_cells=n_cells,
            width=max(1, min(size.most_calls, len(guide_ids))),
            guide_ids=guide_ids,
        )
        parsed = size.parsed_bytes() + guide_positions.nbytes
        del file_calls, guide_positions, row_cells, call_cells, matched_calls
        ledger.release(gathering + parsed)
        # The cells' summaries stay beside the calls while the counts stream.
        ledger.hold("each cell's summary", _SUMMARY_BYTES_PER_CELL * n_cells)
        counts = assay.rawData[cell_index, :]
        blocks = _count_stream(
            ledger,
            counts,
            nthreads=self.nthreads,
            msg=f"({assay.name}) Summarising guide counts",
            item="the guide counts read to summarise each cell's guides",
            scratch_bytes_per_row=_SUMMARY_BYTES_PER_ROW,
        )
        summary = summarize_guide_counts(blocks, n_cells, len(guide_ids))
        self._write_guide_calls(
            ledger, planned, arrays, summary, guide_ids, diagnostics
        )
        _report_barcodes(
            arguments, inspect_artifact(self.zw, planned.ref).diagnostics, call_file
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
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as a call that
                stopped before it finished leaves it.
            MemoryError: If its values do not fit the datastore memory budget.
        """
        _require_ref(calls, "calls", GuideCallImportArguments.artifact_kind)
        status = require_complete_artifact(self.zw, calls)
        group = artifact_group(self.zw, calls)
        ledger = self._artifact_ledger(calls)
        guide_ids = read_whole(ledger, group, "guides", item="the guide IDs")
        cell_ids, arrays = self._cell_arrays(ledger, calls, _GUIDE_CALL_ARRAYS)
        ledger.peak("the tables of cells and calls", _frame_bytes(cell_ids, arrays))
        cells, call_table = guide_calls_frames(cell_ids, **arrays, guide_ids=guide_ids)
        return GuideCalls(
            cells=cells,
            calls=call_table,
            guides=pd.Index(guide_ids, name="guide"),
            source=str(status.operation),
        )

    def _artifact_ledger(
        self, ref: ArtifactRef, *, remedy: str | None = None
    ) -> MemoryLedger:
        """The ledger of a call that reads the artifact ``ref`` within the store's budget, whose refusals name ``remedy``."""
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Reading the {ref.kind} artifact",
            remedy=remedy,
            codec_threads=self.nthreads,
        )
        ledger.hold(
            "the call's plans and the store's metadata it opens", _CALL_FIXED_BYTES
        )
        return ledger

    def _cell_arrays(
        self, ledger: MemoryLedger, ref: ArtifactRef, names: Sequence[str]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read per-cell arrays of a cell-aligned artifact and the cells' IDs, keeping them charged.

        One check of the memory budget covers every array and the IDs before
        any is read, and the IDs are read once.

        Raises:
            ValueError: If a name is not a per-cell array of the artifact's
                kind, or an array does not hold one row per cell.
            MemoryBudgetError: If the arrays and the IDs do not fit the budget.
        """
        for name in names[1:]:
            cell_value_array(ref.kind, name)
        group = artifact_group(self.zw, ref)
        others = {name: as_zarr_array(group[name], name=name) for name in names[1:]}
        id_array = self.cells._get_array("ids")
        resolved = resolve_cell_aligned_artifact(
            self.zw,
            ref,
            value_name=names[0],
            ndim=None,
            ledger=ledger,
            # The other arrays and the IDs are read at the same rows.
            caller_reads=(id_array, *others.values()),
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
            arrays[name] = read_rows(
                ledger, array, rows, item=f"{name!r} of the {ref.kind} artifact"
            )
        cell_ids = read_rows(ledger, id_array, resolved.cell_idx, item="the cells' IDs")
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
        ledger: MemoryLedger,
        assay: Any,
        *,
        counts: Any,
        codes: np.ndarray,
        n_samples: int,
        method: str,
        n_guides: int,
        operation: str,
    ) -> tuple[Any, GuideCountHistograms]:
        """Count the UMIs of each guide and fit it in every sample, keeping the fits and histograms charged.

        ``counts`` are the selected cells' guide counts, as
        ``assay.rawData[cell_index, :]``. The budget must hold the least that
        the histograms hold beside a block of counts before the counts are
        read, a block of the rows of a stored chunk; the store must then
        be writable for ``operation``. The counts stream in such blocks, and
        the histograms take the rest, which they charge as they grow. Returns
        the fits and the histograms, which the caller saves.
        """
        n_cells = len(codes)
        histogram_least = histogram_bytes(n_samples, n_guides, n_cells, n_distinct=0)
        # A block of the rows of a stored chunk of counts, since a smaller
        # block decodes the same chunk again for each block.
        _stream_least, stream_full = _stream_bytes(counts)
        ledger.peak(
            "the guide count histograms' least beside a block of counts",
            stream_full + histogram_least,
        )
        self._require_writable(operation)
        streaming = ledger.hold("the guide counts read for the histograms", stream_full)
        blocks = counts._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Counting guide UMIs",
            prefetch=None,
            row_mask=None,
            resident_bytes=(
                ledger.held_bytes
                if ledger.budget is None
                else ledger.budget - stream_full
            ),
            memory_bytes=ledger.budget,
        )
        try:
            histograms = guide_count_histograms(
                blocks, codes, n_samples=n_samples, n_guides=n_guides, ledger=ledger
            )
        except HistogramMemoryError as error:
            # What the histograms needed where they stopped, beside what the
            # call holds with the streamed counts. Distinct counts grow about
            # as fast as the rows read, or slower, so while rows remain the
            # histograms of every row are estimated from those read; with the
            # budget named, the histograms get at least what they needed where
            # they stopped.
            needed = error.needed_bytes - ledger.held_bytes
            item = error.item
            if error.counts_remain:
                needed = max(
                    needed, -(-needed * error.n_rows // max(error.rows_read, 1))
                )
                item += (
                    ", and at the rate of the rows read the histograms of every "
                    f"row need about {needed} bytes"
                )
            raise memory_refusal(
                ledger.doing,
                item,
                ledger.held_bytes + needed,
                int(error.budget_bytes),
                remedy="pass a smaller cell_selection or fewer samples",
            ) from error
        # The reader that decoded the last chunk of counts can keep it.
        ledger.leave(counts._held_read_bytes())
        ledger.release(streaming)
        ledger.hold(
            "the guide count histograms",
            histograms.offsets.nbytes
            + histograms.counts.nbytes
            + histograms.cells.nbytes,
        )
        n_entries = n_samples * n_guides
        ledger.peak("fitting the guides", _FIT_BYTES_PER_ENTRY * n_entries)
        with numba_threads(self.nthreads):
            fits = (
                fit_poisson_gaussian(histograms)
                if method == "poisson_gaussian"
                else fit_gaussian(histograms)
            )
        ledger.hold("the fits", _FITTED_BYTES_PER_ENTRY * n_entries)
        stopped = int(np.count_nonzero(fits.iterations >= MAX_ITERATIONS))
        if stopped:
            logger.info(
                f"{stopped} guide fits stopped at {MAX_ITERATIONS} iterations "
                "before their loss settled"
            )
        return fits, histograms

    def _write_guide_thresholds(
        self,
        ledger: MemoryLedger,
        planned: PlannedArtifact,
        *,
        fits: Any,
        histograms: GuideCountHistograms,
        method: str,
        samples: list[str | None],
        guide_ids: np.ndarray,
    ) -> None:
        """Save the fits and the histograms that they were fitted to."""
        n_samples, n_guides = histograms.n_samples, histograms.n_guides
        grid = (n_samples, n_guides)
        n_counts = len(histograms.counts)
        width = max(1, *(len(str(guide)) for guide in guide_ids))
        # The thresholds and maximum UMIs as uint32, the cells with UMIs and
        # the maximum UMIs as computed, and the counts as uint32, beside the
        # write of the arrays.
        ledger.peak(
            "saving the guide thresholds",
            (4 + 8 + 8 + 4) * n_samples * n_guides
            + 4 * n_counts
            + cell_data_write_bytes(
                {
                    "guides": ((n_guides,), f"U{width}"),
                    "n_cells": ((n_samples,), np.int64),
                    "status": (grid, np.int8),
                    "threshold": (grid, np.uint32),
                    "n_umi_cells": (grid, np.int64),
                    "max_umi": (grid, np.uint32),
                    "iterations": (grid, np.int32),
                    **{name: (grid, np.float64) for name in METHOD_PARAMETERS[method]},
                    "histogram_offsets": ((n_samples * n_guides + 1,), np.int64),
                    "histogram_counts": ((n_counts,), np.uint32),
                    "histogram_cells": ((n_counts,), np.int64),
                }
            ),
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

    def _stored_guide_fits(
        self, ledger: MemoryLedger, thresholds: ArtifactRef
    ) -> tuple[np.ndarray, np.ndarray, GuideCountHistograms, list[str | None]]:
        """Read the thresholds, statuses, histograms and samples of saved fits, keeping them charged."""
        group = artifact_group(self.zw, thresholds)

        def read(name: str, dtype: Any) -> np.ndarray:
            values = read_whole(ledger, group, name, item=f"the saved fits' {name}")
            if values.dtype == np.dtype(dtype):
                return values
            # The values as int64, which replace those read.
            ledger.hold(f"the saved fits' {name}", 8 * values.size)
            stored = values.nbytes
            values = values.astype(dtype)
            ledger.release(stored)
            return values

        threshold = read("threshold", np.int64)
        histograms = GuideCountHistograms(
            n_cells=read("n_cells", np.int64),
            n_guides=threshold.shape[1],
            offsets=read("histogram_offsets", np.int64),
            counts=read("histogram_counts", np.int64),
            cells=read("histogram_cells", np.int64),
        )
        return (
            threshold,
            read_whole(ledger, group, "status", item="the saved fits' statuses"),
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
        what the counts leave of the budget once they stream in blocks of the
        rows of a stored chunk. Each step is charged to one memory ledger
        before it allocates, the calling pass before it reads a count: each
        cell's detected guides bound its calls under ``fixed`` and the
        fitted methods, whose fits give the number of calls, and
        ``top_guide`` makes at most one call per cell. A refusal names the
        step and the budget that passes it; the histograms' refusal names a
        budget that gets past the point where they stopped, and says when
        more counts remained to be read. The fits are saved once the calls
        that they make have passed their steps.

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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        _require_ref(cell_selection, "cell_selection")
        resolved_min_umi, fraction = call_settings(
            method, min_umi, fraction_threshold, sample_column
        )
        fitted = method in FITTED_METHODS
        _require_cell_column(self, sample_column, "sample_column")
        assay, ledger, guide_ids, selection = self._guide_call_cells(
            from_assay,
            cell_selection,
            GuideCallArguments.operation,
            doing=(
                "Fitting guide thresholds for {n_cells} cells and calling their guides"
                if fitted
                else "Calling guides for {n_cells} cells"
            ),
        )
        n_cells = selection.cells
        cell_index = read_cell_selection(
            ledger, self.zw, selection, item="the selected cells' positions"
        )
        cell_samples = _cell_samples(
            self,
            ledger,
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
                fits_group = artifact_group(self.zw, thresholds.ref)
                status = read_whole(
                    ledger, fits_group, "status", item="the saved fits' statuses"
                )
                samples = _stored_samples(fits_group)
            stored = artifact_group(self.zw, planned.ref)
            self._warn_about_guide_calls(
                method,
                status,
                samples,
                guide_ids,
                read_whole(ledger, stored, "n_guides", item="each cell's calls"),
            )
            return planned.ref
        # A threshold of at least 1 UMI calls at most a cell's detected guides,
        # which bound its calls; top_guide calls at most one guide per cell.
        if method == "top_guide":
            detected_calls, detected_width = n_cells, 1
        else:
            detected = read_rows(
                ledger,
                self.cells._get_array(f"{assay.name}_nFeatures"),
                cell_index,
                item="each cell's detected guides",
            )
            detected_calls = int(detected.sum())
            detected_width = max(1, int(detected.max(initial=0)))
            detected_bytes = detected.nbytes
            del detected
            ledger.release(detected_bytes)
        fits: Any = None
        histograms: GuideCountHistograms | None = None
        counts = assay.rawData[cell_index, :]
        if thresholds is None:
            table = np.full(
                (1, n_guide_ids),
                resolved_min_umi if method == "fixed" else 0,
                dtype=np.int64,
            )
            n_calls, width = detected_calls, detected_width
        else:
            if thresholds.reused:
                threshold, status, histograms, samples = self._stored_guide_fits(
                    ledger, thresholds.ref
                )
            else:
                fitting = ledger.charged
                fits, histograms = self._fit_guide_thresholds(
                    ledger,
                    assay,
                    counts=counts,
                    codes=codes,
                    n_samples=len(samples),
                    method=method,
                    n_guides=n_guide_ids,
                    operation=arguments.operation,
                )
                threshold, status = fits.threshold, fits.status
                fitted_bytes = ledger.charged - fitting
            table = call_thresholds(threshold, resolved_min_umi)
            n_calls = histograms.n_calls(table)
            # Only the guides that some sample calls can be called.
            callable_guides = int(np.count_nonzero((table > 0).any(axis=0)))
            width = max(1, min(detected_width, callable_guides))
            if thresholds.reused:
                # Saved histograms are not saved again; the calls do not need
                # them.
                held = (
                    threshold.nbytes
                    + histograms.n_cells.nbytes
                    + histograms.offsets.nbytes
                    + histograms.counts.nbytes
                    + histograms.cells.nbytes
                )
                del threshold, histograms
                ledger.release(held)
        ledger.doing = (
            f"Calling guides for {n_cells} cells, up to {n_calls} calls and up to "
            f"{width} in one cell"
        )
        self._require_writable(arguments.operation)
        # The cells' summaries and the calls made stay beside the streamed
        # counts, and the calls of the blocks are then joined.
        calling = ledger.hold(
            "each cell's summary and the calls made",
            _SUMMARY_BYTES_PER_CELL * n_cells + _BYTES_PER_CALL * n_calls,
        )
        ledger.peak("joining the calls of the blocks", _BYTES_PER_CALL * n_calls)
        blocks = _count_stream(
            ledger,
            counts,
            nthreads=self.nthreads,
            msg=f"({assay.name}) Calling guides",
            item="the guide counts read to call the guides",
        )
        calls, summary = threshold_calls(blocks, codes, table)
        if method == "top_guide":
            # Whether each cell's top guide is called, its share of the UMIs,
            # and the calls' cells.
            ledger.peak("calling each cell's top guide", 26 * n_cells)
            calls = top_guide_calls(
                summary, min_umi=resolved_min_umi, fraction_threshold=fraction
            )
        arrays = self._pad_calls(
            ledger,
            calls.cells,
            calls.codes,
            calls.umis,
            n_cells=n_cells,
            width=width,
            guide_ids=guide_ids,
        )
        del calls
        ledger.release(calling - _SUMMARY_BYTES_PER_CELL * n_cells)
        if fits is not None:
            assert thresholds is not None and histograms is not None
            # The fits are saved once the calls that they make are known to
            # fit the budget, saving them included once the fits are let go,
            # so that a refused call writes nothing.
            ledger.refuse_beyond(
                "saving the guide calls",
                ledger.held_bytes
                - fitted_bytes
                + _guide_call_write_bytes(
                    n_cells, int(arrays.guide_codes.shape[1]), guide_ids
                ),
            )
            self._write_guide_thresholds(
                ledger,
                thresholds,
                fits=fits,
                histograms=histograms,
                method=method,
                samples=samples,
                guide_ids=guide_ids,
            )
            del fits, histograms
            ledger.release(fitted_bytes)
        self._write_guide_calls(ledger, planned, arrays, summary, guide_ids)
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
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as a call that
                stopped before it finished leaves it.
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
        ledger = self._artifact_ledger(source, remedy="pass guides to read fewer")
        guide_ids = read_whole(ledger, group, "guides", item="the guide IDs").astype(
            str
        )
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
        offsets = read_whole(
            ledger, group, "histogram_offsets", item="the histograms' offsets"
        ).astype(np.int64)
        samples = _stored_samples(group)
        n_samples = len(samples)
        entries = (
            np.arange(n_samples)[:, None] * len(guide_ids) + positions[None, :]
        ).reshape(-1)
        sizes = offsets[entries + 1] - offsets[entries]
        n_counts = int(sizes.sum())
        ledger.doing = (
            f"Reading the thresholds of {len(positions)} guides in {n_samples} samples"
        )
        if guides is None:
            counts = read_whole(
                ledger, group, "histogram_counts", item="the histograms' counts"
            )
            cells = read_whole(
                ledger, group, "histogram_cells", item="the histograms' cells"
            )
        else:
            # The rows of the guides' histograms, in the order of their
            # entries, which hold distinct rows.
            ledger.peak("the rows of the guides' histograms", 16 * n_counts)
            rows = np.concatenate(
                [
                    np.arange(start, start + size, dtype=np.int64)
                    for start, size in zip(offsets[entries], sizes, strict=True)
                ]
                or [np.zeros(0, dtype=np.int64)]
            )
            ledger.hold("the rows of the guides' histograms", rows.nbytes)
            counts = read_rows(
                ledger,
                as_zarr_array(group["histogram_counts"], name="histogram_counts"),
                rows,
                item="the histograms' counts",
            )
            cells = read_rows(
                ledger,
                as_zarr_array(group["histogram_cells"], name="histogram_cells"),
                rows,
                item="the histograms' cells",
            )
        # The tables hold each count and each fit as a row of their own,
        # beside the counts and cells read.
        ledger.peak(
            "the tables of thresholds and histograms",
            _TABLE_BYTES_PER_COUNT * n_counts + _TABLE_BYTES_PER_FIT * len(entries),
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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        _require_ref(calls, "calls", GuideCallImportArguments.artifact_kind)
        control_names = label_settings(controls, target_column, targets, policy)
        status = require_complete_artifact(self.zw, calls)
        assay = self._get_assay(str(calls.assay))
        group = artifact_group(self.zw, calls)
        codes = as_zarr_array(group["guide_codes"], name="guide_codes")
        n_cells, width = (int(size) for size in codes.shape)
        ledger = MemoryLedger(
            self.memoryBytes, f"Labelling {n_cells} cells", codec_threads=self.nthreads
        )
        ledger.hold(
            "the call's plans and the store's metadata it opens", _CALL_FIXED_BYTES
        )
        guide_ids = read_whole(ledger, group, "guides", item="the guide IDs").astype(
            str
        )
        column = self._guide_feature_column(
            assay, target_column, "target_column", ledger=ledger
        )
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
        guide_codes = read_whole(ledger, group, "guide_codes", item="the calls' guides")
        # Each called guide must have a target; a padded entry reads the last
        # guide's flag, which the call flags drop.
        ledger.peak(
            "checking that each called guide has a target", 2 * guide_codes.size
        )
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
        arrays = perturbation_label_arrays(
            guide_codes, resolved, policy=policy, ledger=ledger
        )
        codes_bytes = guide_codes.nbytes
        del guide_codes
        ledger.release(codes_bytes)
        # The labels of the calls, which hold one label per cell of the calls.
        guide_label = read_whole(ledger, group, "values", item="the calls' labels")
        label_length = longest_label(resolved, width, policy=policy)
        ledger.peak(
            "saving the labels",
            cell_data_write_bytes(
                {
                    "values": ((n_cells,), f"U{label_length}"),
                    "guide_label": ((n_cells,), guide_label.dtype),
                    "is_control": ((n_cells,), bool),
                    "n_targets": ((n_cells,), np.int32),
                    "target_codes": (arrays.target_codes.shape, np.int32),
                    "targets": (
                        (n_names,),
                        f"U{max(1, *(len(str(name)) for name in resolved.names))}",
                    ),
                    "target_is_control": ((n_names,), bool),
                    "guide_target_codes": ((n_guide_ids,), np.int32),
                }
            ),
        )
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
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as a call that
                stopped before it finished leaves it.
            MemoryError: If its values do not fit the datastore memory budget.
        """
        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        status = require_complete_artifact(self.zw, labels)
        group = artifact_group(self.zw, labels)
        calls = artifact_group(self.zw, status.input_ref("guide_calls"))

        ledger = self._artifact_ledger(labels)
        target_names = read_whole(ledger, group, "targets", item="the targets").astype(
            str
        )
        guide_ids = read_whole(ledger, calls, "guides", item="the guide IDs").astype(
            str
        )
        cell_ids, loaded = self._cell_arrays(
            ledger,
            labels,
            ("values", "guide_label", "is_control", "n_targets", "target_codes"),
        )
        ledger.peak(
            "the tables of cells, targets and guides", _frame_bytes(cell_ids, loaded)
        )
        cells, targets, guides = perturbation_label_frames(
            cell_ids,
            **loaded,
            target_names=target_names,
            target_is_control=read_array(group, "target_is_control"),
            guide_ids=guide_ids,
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
