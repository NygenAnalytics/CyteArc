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
    PerturbationAssociationArguments,
    PerturbationClusterArguments,
    PerturbationEDistanceArguments,
    PerturbationEfficiencyArguments,
    PerturbationLabelArguments,
    PerturbationProfileArguments,
    PerturbationSignatureArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    read_array_rows_chunkwise,
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
    FITTED_METHODS,
    METHOD_PARAMETERS,
    GuideThresholds,
    call_settings,
    call_thresholds,
    guide_threshold_frames,
)
from ...perturbation.edistance import (
    EDISTANCE_STATUSES,
    PerturbationEDistance,
    edistance_arrays,
    edistance_memory_bytes,
    edistance_settings,
)
from ...perturbation.efficiency import (
    EFFICIENCY_STATUSES,
    NO_FEATURE,
    PerturbationEfficiency,
    efficiency_arrays,
    efficiency_settings,
    target_features,
)
from ...perturbation.labels import (
    PerturbationLabels,
    guide_targets,
    label_settings,
    perturbation_label_arrays,
    perturbation_label_frames,
)
from ...perturbation.associations import (
    PerturbationAssociations,
    association_settings,
    covariate_columns,
)
from ...perturbation.clusters import (
    PerturbationClusters,
    cluster_settings,
    perturbation_names,
)
from ...perturbation.mixscape import MixscapeResult, mixscape_settings
from ...perturbation.profiles import PerturbationProfiles, profile_settings
from ...perturbation.signature import PerturbationSignature, signature_settings
from ...perturbation.histograms import (
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
# The share of the memory left beside the tests that held counts may take, so
# that the tests need not read the counts again.
_ASSOCIATION_HELD_SHARE = 0.5
# Bytes held for each pair of the association tests: its row and feature, QC
# counts and results, and the copies that the tests, the adjustment and the
# write make of them.
_ASSOCIATION_PAIR_BYTES = 320
# The most that each thread's copy of a group's genes takes in the high-MOI
# tests, the default of their engine, when the memory budget leaves room.
_ASSOCIATION_GATHER_BYTES = 1 << 26
# The most negative-control groups of a calibration check.
_CALIBRATION_GROUPS = 100
# Whether a calibration check ran, saved with the tests: it ran, it was turned
# off, or it could not run for too few control guides, for no discovery pair
# that passes QC by default, or for no calibration pair that passes QC.
_CALIBRATION_STATUSES = (
    "checked",
    "off",
    "few_control_guides",
    "no_passing_discovery",
    "no_passing_calibration",
)
# The integer and floating-point columns of a perturbation association table,
# one row per pair; each pair's group of treatment cells names its target and
# guides once per group.
_ASSOCIATION_COUNTS = (
    "sets",
    "groups",
    "features",
    "n_cells",
    "n_control_cells",
    "n_nonzero_cells",
    "n_nonzero_control_cells",
    "stage",
)
_ASSOCIATION_VALUES = (
    "z",
    "p_value",
    "adjusted_p_value",
    "log2_fold_change",
    "se_log2_fold_change",
    "skew_location",
    "skew_scale",
    "skew_shape",
)
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


@dataclass(frozen=True, slots=True)
class _ProfileCells:
    """The cells of perturbation profiles, from labels or Mixscape classes.

    Attributes:
        labels: The perturbation labels that name the targets and controls.
        rows: The perturbations: each target of the labels that is not a
            control, or each guide with a target.
        row_control: Whether each row is a control guide.
        cell_selection: The selection of the cells.
        cell_index: The cells' positions in the store.
        groups: Each cell's perturbation, a row of ``rows``, or -1.
        control: Whether each cell is a control cell.
    """

    labels: ArtifactRef
    rows: _LabelRows
    row_control: np.ndarray
    cell_selection: ArtifactRef
    cell_index: np.ndarray
    groups: np.ndarray
    control: np.ndarray


@dataclass(frozen=True, slots=True)
class _AssociationTables:
    """The tested pairs of association tests and the calibration groups behind them.

    Attributes:
        sets: Each pair's set, as a position in ``ASSOCIATION_SETS``.
        groups: Each pair's group of treatment cells: a tested row, as its
            position among the tested rows, or after them a calibration group.
        features: Each pair's feature.
        qc: Each pair's ``n_cells``, ``n_control_cells``, ``n_nonzero_cells``
            and ``n_nonzero_control_cells``.
        pass_qc: Whether each pair passes QC.
        values: Each pair's statistic, p-value, fold change and skew-normal
            fit by their names in the table, NaN for an untested pair; the
            adjusted p-values are filled in when the tests are saved.
        stage: The round that gave each pair's p-value, 0 for an untested pair.
        group_size: The guides of each calibration group.
        control_guides: The IDs of the control guides that the calibration
            groups draw from.
        calibration_groups: Each calibration group's guides, as positions in
            ``control_guides``.
        passing_discovery: The discovery pairs that pass QC.
    """

    sets: np.ndarray
    groups: np.ndarray
    features: np.ndarray
    qc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    pass_qc: np.ndarray
    values: dict[str, np.ndarray]
    stage: np.ndarray
    group_size: int
    control_guides: np.ndarray
    calibration_groups: list[np.ndarray]
    passing_discovery: int


def _cells_by_code(codes: np.ndarray, n_codes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the cells that hold each code, ascending, and where each code's cells start.

    ``codes`` holds each cell's codes from 0, padded with -1, cells x slots,
    and a cell holds each of its codes once.
    """
    cells, slots = np.nonzero(codes >= 0)
    held = codes[cells, slots].astype(np.int64)
    # A stable sort keeps each code's cells in the ascending order of nonzero.
    members = cells[np.argsort(held, kind="stable")].astype(np.int64)
    offsets = np.zeros(n_codes + 1, dtype=np.int64)
    np.cumsum(np.bincount(held, minlength=n_codes), out=offsets[1:])
    return members, offsets


def _pair_sets(
    *parts: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Join the pairs of each set, in the order of ``ASSOCIATION_SETS``.

    Each part holds its pairs' groups, features and four QC counts; the
    result holds every pair's set, group, feature and QC counts.
    """
    sets = np.concatenate(
        [np.full(len(part[0]), code, np.int8) for code, part in enumerate(parts)]
    )
    groups = np.concatenate([part[0] for part in parts]).astype(np.int64)
    features = np.concatenate([part[1] for part in parts]).astype(np.int64)
    n_cells, n_control_cells, n_nonzero_cells, n_nonzero_control_cells = (
        np.concatenate([part[2][k] for part in parts]).astype(np.int64)
        for k in range(4)
    )
    return (
        sets,
        groups,
        features,
        (n_cells, n_control_cells, n_nonzero_cells, n_nonzero_control_cells),
    )


def _most_calibration_pairs(
    n_groups: int, n_features: int, n_discovery: int, requested: int | None
) -> int:
    """Return the most calibration pairs that a check can draw, for the memory plan.

    A check draws as many pairs as requested, or by default as the discovery
    pairs that pass QC, and at most every pair of one of its ``n_groups``
    groups and one of the assay's ``n_features`` features.
    """
    wanted = n_discovery if requested is None else requested
    return min(wanted, n_groups * n_features)


def _calibration_group_size(rows: _LabelRows, by: str, used: np.ndarray) -> int:
    """Return the guides of each calibration group: the median of the tested targets'.

    Each guide tested alone makes groups of one guide.
    """
    if by != "target":
        return 1
    targeted = rows.guide_targets[rows.guide_targets >= 0]
    per_target = np.bincount(targeted, minlength=len(rows.target_names))
    return max(1, int(round(float(np.median(per_target[rows.codes[used]])))))


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


def _unmatched_targets_warning(
    names: np.ndarray, status: np.ndarray, assay: str
) -> str | None:
    """Name the efficiency rows whose target matches no feature, or return None.

    A measured table warns with this message, and so does a saved table when
    ``efficiency`` reuses it.

    Args:
        names: Each row's name: its target, or its guide.
        status: Each row's status, as a position in ``EFFICIENCY_STATUSES``.
        assay: The expression assay.
    """
    unmatched = names[np.asarray(status) == NO_FEATURE]
    if not len(unmatched):
        return None
    return (
        f"{len(unmatched)} of {len(names)} rows have no feature of assay "
        f"{assay!r} for their target, for example "
        + ", ".join(repr(str(name)) for name in unmatched[:5])
        + ". Check id_column, or the target names against the feature names."
    )


def _association_warnings(
    unmatched: np.ndarray,
    n_rows: int,
    by: str,
    assay: str,
    attributes: Mapping[str, Any],
    moi: str,
) -> list[str]:
    """Describe what association tests warn about, whether computed or reused.

    Args:
        unmatched: The rows, targets or guides, without a feature for a
            positive control.
        n_rows: The rows of the labels.
        by: ``"target"`` or ``"guide"``.
        assay: The expression assay.
        attributes: The attributes of the tests: ``calibration_status``,
            ``n_control_guides``, ``calibration_grade``,
            ``n_false_discoveries`` and ``n_negative_control_pairs``.
        moi: ``"low"`` or ``"high"``, which choose the control guides of the
            calibration groups.
    """
    messages = []
    if len(unmatched):
        messages.append(
            f"{len(unmatched)} of {n_rows} {by}s have no feature of assay "
            f"{assay!r} for a positive control, for example "
            + ", ".join(repr(str(name)) for name in unmatched[:5])
            + ". A target that is not a gene, such as an enhancer, has none; for "
            "a gene, check id_column, or the target names against the feature "
            "names."
        )
    status = attributes["calibration_status"]
    if status == "few_control_guides" and moi == "high":
        messages.append(
            "No labelled cell carries a control guide, which a calibration group "
            "needs, so no calibration check was run"
        )
    elif status == "few_control_guides":
        n_guides = int(attributes["n_control_guides"])
        guides = "1 control guide" if n_guides == 1 else f"{n_guides} control guides"
        messages.append(
            f"Control cells with one guide carry {guides}, fewer than the 2 that "
            "a calibration group and its comparison cells need, so no calibration "
            "check was run"
        )
    elif status == "no_passing_discovery":
        messages.append(
            "No discovery pair passes QC, and by default the calibration check "
            "tests as many pairs as pass, so no calibration check was run; pass "
            "n_calibration_pairs to run one"
        )
    elif status == "no_passing_calibration":
        messages.append(
            "No pair of a calibration group and a feature passes QC, so no "
            "calibration check was run"
        )
    grade = attributes["calibration_grade"]
    if grade in ("moderate", "severe"):
        messages.append(
            f"The calibration check found {attributes['n_false_discoveries']} "
            f"false discoveries among {attributes['n_negative_control_pairs']} "
            f"negative-control pairs, a {grade} miscalibration; check the "
            "covariates and the control guides before trusting the discoveries"
        )
    return messages


def _left_out_warning(
    names: np.ndarray,
    n_cells: np.ndarray,
    n_rows: int,
    min_cells: int,
    rows: str,
    cells: str,
) -> str | None:
    """Name the perturbations without a profile, or return None when every one has one.

    Args:
        names: The perturbations with fewer than ``min_cells`` cells.
        n_cells: Their cells.
        n_rows: The perturbations of the labels.
        min_cells: The fewest cells of a profiled perturbation.
        rows: What the perturbations are, ``"targets"`` or ``"guides"``.
        cells: What their cells are, such as ``"knocked-out cells"``.
    """
    if not len(names):
        return None
    examples = ", ".join(
        f"{str(name)!r} ({int(count)})"
        for name, count in zip(names[:5], n_cells[:5], strict=True)
    )
    more = ", and more" if len(names) > 5 else ""
    return (
        f"{len(names)} of {n_rows} {rows} have fewer than {min_cells} {cells} "
        f"and get no profile: {examples}{more}. perturbation.load_profiles lists "
        "them all in left_out."
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
        ``unmatched_barcode_examples`` attributes, and named in a logged
        warning, which a reused import logs again.

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
            required_attributes=(
                "n_file_barcodes",
                "n_unmatched_barcodes",
                "unmatched_barcode_examples",
            ),
            reuse_validator=has_guides,
        )
        if planned.reused:
            # Saved calls warn as they did when they were imported.
            message = _unmatched_barcodes_warning(
                artifact_group(self.zw, planned.ref).attrs
            )
            if message is not None:
                logger.warning(message)
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
        # The attributes also hold what the warning names, so that saved calls
        # warn again as they did.
        attributes = {
            "n_file_barcodes": int(len(file_calls.barcodes)),
            "n_unmatched_barcodes": int(len(unmatched)),
            "unmatched_barcode_examples": [str(value) for value in examples],
        }
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
        message = _unmatched_barcodes_warning(attributes)
        if message is not None:
            logger.warning(message)
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
        resolved_min_umi, fraction = call_settings(
            method, min_umi, fraction_threshold, sample_column
        )
        fitted = method in FITTED_METHODS
        if sample_column is not None:
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
        control_names = label_settings(controls, target_column, targets, policy)
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        status = require_complete_artifact(self.zw, calls)
        assay = self._get_assay(str(calls.assay))
        group = artifact_group(self.zw, calls)
        guide_ids = np.asarray(as_zarr_array(group["guides"], name="guides")[:]).astype(
            str
        )
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

    def _label_rows(
        self, labels: ArtifactRef, by: str, *, control_guides: bool = False
    ) -> _LabelRows:
        """Read the rows of a per-perturbation table: each target or each guide.

        A row is a target that is not a control, or with ``by="guide"`` a guide
        whose target is not a control, in assay order; ``control_guides``
        makes the guides of control targets rows too.
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
                (guide_targets >= 0)
                & (control_guides | ~is_control[np.maximum(guide_targets, 0)])
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
            ValueError: If no labelled cell carries control guides only.
        """
        group = artifact_group(self.zw, labels)

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        control = read("is_control").astype(bool)
        if not control.any():
            raise ValueError(
                "No labelled cell carries control guides only, and each "
                "perturbation is compared with such cells; check the controls "
                "passed to perturbation.labels, or in a screen whose cells carry "
                "several guides, test associations with moi='high'"
            )
        if by == "target":
            row_of = np.full(len(rows.target_names), -1, dtype=np.int64)
            row_of[rows.codes] = np.arange(len(rows.codes))
            single = read("n_targets") == 1
            # Only each cell's first target is read, not the other columns.
            first = np.asarray(
                as_zarr_array(group["target_codes"], name="target_codes")[:, 0]
            )
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
        ``"no_feature"`` and named in a warning, which a reused table repeats.

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
                no labelled cell carries control guides only.
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
        efficiency_settings(by, id_column)
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        rows = self._label_rows(labels, by)
        status = rows.status
        target_names, guide_targets_of = rows.target_names, rows.guide_targets
        row_codes, row_names = rows.codes, rows.names
        guide_assay = self._get_assay(str(labels.assay))
        expression = self._get_assay(from_assay)
        column = self._guide_feature_column(guide_assay, id_column, "id_column")
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
                ArrayRequirement("names", shape=(n_rows,)),
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
            # A saved table warns as it did when it was measured.
            stored = artifact_group(self.zw, planned.ref)
            message = _unmatched_targets_warning(
                np.asarray(as_zarr_array(stored["names"], name="names")[:]).astype(str),
                np.asarray(as_zarr_array(stored["status"], name="status")[:]),
                expression.name,
            )
            if message is not None:
                warn(message)
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
        message = _unmatched_targets_warning(row_names, arrays.status, expression.name)
        if message is not None:
            warn(message)
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
        permutations, seed = edistance_settings(by, n_permutations, random_seed)
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
        resolved_dims, neighbors_k, factor = signature_settings(
            dims, n_neighbors, scale_factor, sample_column
        )
        if sample_column is not None:
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
            # The frame wraps the values as read, the 4 bytes per value that the
            # budget check above counts; pandas would otherwise copy them.
            values=pd.DataFrame(
                values, index=index, columns=feature_ids[columns], copy=False
            ),
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
        if split_column is not None:
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

    def _association_pairs(
        self,
        frame: object,
        name: str,
        rows: _LabelRows,
        by: str,
        feature_ids: np.ndarray,
        feature_names: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve a table of target and feature pairs to rows and features.

        A target names a row, or with ``by="guide"`` every guide of the target;
        a feature is matched by ID, or by name when no ID matches and one
        feature has the name.

        Raises:
            TypeError: If ``frame`` is not a DataFrame.
            ValueError: If it lacks a column, or names a target or feature that
                is not there.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame of target and feature")
        missing = [column for column in ("target", "feature") if column not in frame]
        if missing:
            raise ValueError(f"{name} needs the columns 'target' and 'feature'")
        targets = frame["target"].astype(str).to_numpy()
        wanted = frame["feature"].astype(str).to_numpy()
        row_targets = (
            rows.names if by == "target" else rows.target_names[rows.codes]
        ).astype(str)
        known = set(row_targets.tolist())
        unknown = sorted(set(targets.tolist()) - known)
        if unknown:
            raise ValueError(
                f"{name} names targets that the labels do not test: "
                + ", ".join(repr(target) for target in unknown[:5])
            )
        by_id = pd.Index(feature_ids)
        positions = (
            by_id.get_indexer(wanted) if by_id.is_unique else np.full(len(wanted), -1)
        )
        names = pd.Series(np.arange(len(feature_names)), index=feature_names)
        counts = names.index.value_counts()
        for k in np.flatnonzero(positions < 0):
            if counts.get(wanted[k], 0) == 1:
                positions[k] = int(names[wanted[k]])
        if (positions < 0).any():
            absent = sorted(set(wanted[positions < 0].tolist()))
            raise ValueError(
                f"{name} names features that are not one feature of the assay: "
                + ", ".join(repr(feature) for feature in absent[:5])
            )
        row_lists = (
            pd.Series(np.arange(len(row_targets)))
            .groupby(row_targets)
            .apply(np.asarray)
        )
        pair_rows = np.concatenate(
            [row_lists[target] for target in targets] or [np.empty(0, np.int64)]
        ).astype(np.int64)
        repeats = np.asarray([len(row_lists[target]) for target in targets], np.int64)
        pair_features = np.repeat(positions.astype(np.int64), repeats)
        keys = np.unique(pair_rows * len(feature_ids) + pair_features)
        return keys // len(feature_ids), keys % len(feature_ids)

    @validation_scoped
    def _perturbation_associations(
        self,
        labels: ArtifactRef,
        *,
        from_assay: str = "RNA",
        pairs: str | pd.DataFrame = "trans",
        features: ArtifactRef | None = None,
        positive_controls: str | pd.DataFrame | None = "targets",
        id_column: str | None = "target_gene_id",
        integration: str = "union",
        moi: str = "low",
        resampling: str | None = None,
        sample_column: str | None = None,
        covariates: Sequence[str] | None = None,
        side: str = "both",
        n_resamples: Sequence[int] = (499, 4999, 24999),
        stage_threshold: float = 0.02,
        approximation: str = "skew_normal",
        min_nonzero: int = 7,
        n_calibration_pairs: int | None = None,
        alpha: float = 0.1,
        seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Test whether each perturbation changes each gene, with calibrated p-values.

        SCEPTRE's low-MOI test (Barry et al. 2024), the default, compares, for
        each pair of a perturbation and a gene, the cells that carry the
        perturbation alone with the control cells of the labels;
        ``moi="high"`` runs its high-MOI test, described below. A negative
        binomial regression of the gene's counts on the cells' covariates,
        fitted on those cells together under the null hypothesis, scores each
        cell, and the pair's statistic is the perturbed cells' standardized
        score. The fit takes Poisson coefficients and the size that maximizes
        the likelihood with the Poisson means held fixed. Its p-value comes
        from the same statistic over random sets of as many cells drawn from
        the perturbed and control cells: ``n_resamples[0]`` sets first; for a
        pair whose p-value is then at most ``stage_threshold``,
        ``n_resamples[1]`` fresh sets whose statistics a skew-normal fit
        extends to far smaller p-values; and when the fit's tail beyond those
        statistics' 99th percentile holds less than half of their own share
        there, ``n_resamples[2]`` fresh sets (``approximation="none"`` keeps
        the second round's p-values). The fit does not depend on which of the
        pair's cells carry the perturbation, so the test is exact.

        The covariates are log detected features and log UMIs of each cell in
        ``from_assay``, indicators of the samples of ``sample_column``, and the
        numeric cell columns of ``covariates``. Only the tested cells, the
        control cells and the cells of the tested targets or guides, need a
        sample label and covariate values that are not missing. A pair is
        tested when at least ``min_nonzero`` perturbed and ``min_nonzero``
        control cells express the gene, and p-values are adjusted by
        Benjamini-Hochberg within each set of pairs. The fold change is log2
        of the perturbed cells' counts over their means under the pair's fit,
        the estimator of the SCEPTRE book.

        Three sets of pairs are tested:

        - Discovery pairs: with ``pairs="trans"``, every target with every
          feature of ``features``; or the rows of a DataFrame of ``target``
          and ``feature``.
        - Calibration pairs: up to 100 distinct groups of control guides,
          drawn at random with as many guides as the median target and
          possibly sharing guides, paired at random with features of the whole
          assay that pass QC; as many pairs as the discovery pairs that pass
          QC by default. A group's treatment cells are the control cells
          whose one guide is in the group, and every other control cell is
          its comparison: the cells of the other control guides, and the
          control cells with several control guides, even guides of the
          group, since under the null hypothesis no control guide has an
          effect. The pairs should give no discovery; the grade of their
          false discoveries follows the SCEPTRE book: excellent (0 or 1), mild
          (up to 6), moderate (up to 14) or severe. The grade is ``"none"``
          when no check runs: with ``n_calibration_pairs=0``; with fewer than
          2 control guides that are the one guide of a control cell; by
          default, when no discovery pair passes QC; or when no pair of a
          group and a feature passes QC. A warning says why in all but the
          first case.
        - Positive-control pairs: with ``positive_controls="targets"``, each
          target with its own gene, matched as
          :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.efficiency`
          matches them; or a DataFrame of ``target`` and ``feature``.

        With ``integration="singleton"``, each guide is tested alone instead
        of the guides of a target together, and the calibration groups hold
        one control guide each.

        In a high-MOI screen each cell carries several guides, so few cells
        carry a target alone and almost none carry control guides only. With
        ``moi="high"``, as in SCEPTRE's high-MOI test (Barry et al. 2021,
        2024), a target's treatment cells are the labelled cells whose guides
        target it, or with ``"singleton"`` the cells that carry the guide,
        whatever else they carry, and every other labelled cell is its
        comparison: cells of other targets or of sibling guides, cells with
        control guides and cells without a called guide. Every labelled cell
        is tested, so each one needs a sample label, covariate values, and
        guide UMIs, since the covariates add log detected guides and log guide
        UMIs of the guide assay. Each gene is fitted once on all labelled
        cells, and all of its pairs share that fit.
        ``resampling="conditional"``, the default at high MOI, fits a logistic
        regression of each group's cells on the covariates and draws every
        cell into a random set with its fitted probability, the conditional
        randomization test, which stays valid when sequencing depth drives
        both the detection of guides and the expression of genes;
        ``"permutation"`` draws as many cells as the group uniformly from all
        cells. The calibration groups take any control guide called in a
        labelled cell, as many guides as the median target or all of them when
        fewer, and their treatment cells, the cells that carry any guide of
        the group, are compared with all other labelled cells, so one control
        guide is enough; the grade is ``"none"`` when no labelled cell carries
        one. QC counts the expressing cells of each group against all other
        cells.

        Args:
            labels: A ``perturbation_labels`` artifact.
            from_assay: The expression assay, whose counts are tested.
            pairs: ``"trans"``, or a DataFrame with columns ``target`` and
                ``feature`` (feature IDs, or names that one feature has).
            features: A ``feature_selection`` artifact of ``from_assay`` whose
                features trans pairs test, such as highly variable genes.
            positive_controls: ``"targets"``, a DataFrame like ``pairs``, or
                None.
            id_column: The feature column of the guide assay that holds each
                guide's target feature ID, or None to match targets by name.
            integration: ``"union"`` or ``"singleton"``.
            moi: ``"low"`` for a screen of one guide per cell, or ``"high"``
                for a screen of several guides per cell.
            resampling: ``"conditional"`` or ``"permutation"`` at high MOI,
                or None for ``"conditional"`` there and ``"permutation"``,
                the only choice, at low MOI.
            sample_column: A cell metadata column whose samples get
                indicators, or None.
            covariates: Numeric cell metadata columns to adjust for, or None.
            side: ``"both"``, ``"left"`` for a decrease, or ``"right"`` for an
                increase.
            n_resamples: The random sets of the three rounds, a sequence of
                three integers such as a tuple or a list.
            stage_threshold: The p-value at or below which a pair takes the
                second round.
            approximation: ``"skew_normal"`` or ``"none"``.
            min_nonzero: The fewest perturbed and control cells that express a
                tested gene.
            n_calibration_pairs: The calibration pairs, None for as many as
                the discovery pairs that pass QC, or 0 for no calibration
                check. A positive number runs the check without discovery
                pairs too.
            alpha: The false discovery rate of ``significant``.
            seed: The seed of the random sets and the calibration pairs.
            invalidate_cache: Test again instead of reusing saved tests.

        Returns:
            A complete ``perturbation_associations`` artifact scoped to
            ``from_assay``, with one row per pair. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_associations`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``labels`` is not a ``perturbation_labels``
                artifact, an assay, column, target or feature is unknown, a
                number is out of range, ``resampling="conditional"`` goes
                with ``moi="low"``, trans pairs have no ``features``, a table
                of pairs is malformed, at low MOI no labelled cell carries
                control guides only, or a tested cell has no sample label, a
                covariate value that is missing or not finite, or at high MOI
                no guide UMIs.
            PermissionError: If the store is read-only and the tests are not
                saved yet.
            MemoryError: If one feature's counts over the tested cells, or at
                high MOI the tests of one gene, do not fit the memory budget.
        """
        from ...perturbation.associations import (
            ASSOCIATION_SETS,
            TreatmentGroups,
            association_covariates,
            calibration_pairs,
            negative_control_groups,
            nonzero_counts,
            pair_tests,
            record_batch,
        )

        from ...storage.artifacts import fingerprint_array
        from ...storage.feature_selection import read_feature_selection_indices

        if not isinstance(labels, ArtifactRef):
            raise TypeError("labels must be an ArtifactRef")
        if labels.kind != PerturbationLabelArguments.artifact_kind:
            raise ValueError(
                f"labels must be a perturbation_labels artifact, not {labels.kind!r}"
            )
        if not isinstance(from_assay, str) or from_assay not in self.assay_names:
            raise ValueError(f"from_assay must name an assay, not {from_assay!r}")
        if isinstance(pairs, str):
            if pairs != "trans":
                raise ValueError(f"pairs must be 'trans' or a DataFrame, not {pairs!r}")
            if not isinstance(features, ArtifactRef):
                raise ValueError(
                    "Trans pairs test the features of a feature selection; pass "
                    "features, such as highly variable genes"
                )
        elif features is not None:
            raise ValueError("features goes with pairs='trans' only")
        if features is not None and features.kind != "feature_selection":
            raise ValueError(
                f"features must be a feature_selection artifact, not {features.kind!r}"
            )
        (
            draws,
            threshold,
            floor,
            requested,
            rate,
            resolved_seed,
            resolved_resampling,
        ) = association_settings(
            positive_controls,
            id_column,
            integration,
            side,
            approximation,
            n_resamples,
            stage_threshold,
            min_nonzero,
            n_calibration_pairs,
            alpha,
            seed,
            moi,
            resampling,
        )
        high = moi == "high"
        if sample_column is not None:
            if not isinstance(sample_column, str) or not sample_column:
                raise TypeError("sample_column must be a non-empty string or None")
            if sample_column not in self.cells.columns:
                raise ValueError(
                    f"sample_column '{sample_column}' not found in cell metadata"
                )
        covariate_names = covariate_columns(covariates)
        for column in covariate_names:
            if column not in self.cells.columns:
                raise ValueError(f"covariate '{column}' not found in cell metadata")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        by = "target" if integration == "union" else "guide"
        rows = self._label_rows(labels, by)
        status = rows.status
        expression = self._get_assay(from_assay)
        cell_selection = status.input_ref("cell_selection")
        self._require_measured_cells(
            expression.name,
            cell_selection,
            operation=PerturbationAssociationArguments.operation,
        )
        feature_ids = np.asarray(expression.feats.fetch_all("ids")).astype(str)
        feature_names = np.asarray(expression.feats.fetch_all("names")).astype(str)
        n_rows = len(rows.names)

        def pair_fingerprint(pair_rows: np.ndarray, pair_features: np.ndarray) -> str:
            return fingerprint_strings(
                np.char.add(
                    np.char.add(rows.names[pair_rows].astype(str), "\t"),
                    feature_ids[pair_features],
                )
            )

        pairs_fingerprint = None
        if isinstance(pairs, str):
            assert features is not None
            if str(features.assay) != expression.name:
                raise ValueError(
                    f"features must be a feature selection of assay {expression.name!r}"
                )
            selected = np.asarray(
                read_feature_selection_indices(self.zw, expression.name, features),
                dtype=np.int64,
            )
            discovery = (
                np.repeat(np.arange(n_rows, dtype=np.int64), len(selected)),
                np.tile(selected, n_rows),
            )
        else:
            discovery = self._association_pairs(
                pairs, "pairs", rows, by, feature_ids, feature_names
            )
            pairs_fingerprint = pair_fingerprint(*discovery)
        positive_fingerprint = None
        unmatched = np.empty(0, dtype=str)
        if positive_controls is None:
            positive = (np.empty(0, np.int64), np.empty(0, np.int64))
        elif isinstance(positive_controls, str):
            guide_feature_ids = self._guide_feature_column(
                self._get_assay(str(labels.assay)), id_column, "id_column"
            )
            row_features = target_features(
                rows.target_names,
                rows.guide_targets,
                guide_feature_ids,
                feature_ids,
                feature_names,
            )[rows.codes]
            matched = np.flatnonzero(row_features >= 0)
            positive = (
                matched.astype(np.int64),
                row_features[matched].astype(np.int64),
            )
            unmatched = rows.names[row_features < 0].astype(str)
            positive_fingerprint = pair_fingerprint(*positive)
        else:
            positive = self._association_pairs(
                positive_controls,
                "positive_controls",
                rows,
                by,
                feature_ids,
                feature_names,
            )
            positive_fingerprint = pair_fingerprint(*positive)
        if not len(discovery[0]) and not len(positive[0]):
            raise ValueError(
                "There is no pair to test; pass pairs or positive controls"
            )
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_labelled = len(cell_index)
        n_discovery = len(discovery[0])
        used = np.unique(np.concatenate([discovery[0], positive[0]]))
        compact = np.full(n_rows, -1, dtype=np.int64)
        compact[used] = np.arange(len(used))
        if high:
            # Every labelled cell is tested, as a treatment cell of its
            # groups and a comparison cell of the others.
            tested = np.arange(n_labelled, dtype=np.int64)
            needing = "every labelled cell needs one"
        else:
            # The tested cells: the control cells, then the cells of each
            # tested row, row by row. Only their samples and covariates enter
            # the tests.
            row_of, control = self._label_groups(labels, rows, by)
            controls = np.flatnonzero(control)
            target_row = np.where(row_of >= 0, compact[np.maximum(row_of, 0)], -1)
            targets_cells = np.flatnonzero(~control & (target_row >= 0))
            targets_cells = targets_cells[
                np.argsort(target_row[targets_cells], kind="stable")
            ]
            tested = np.concatenate([controls, targets_cells])
            needing = f"every control cell and every cell of a tested {by} needs one"
        tested_cells = cell_index[tested]
        sample_columns = () if sample_column is None else (sample_column,)
        samples, _, assignments = sample_codes(
            sample_columns,
            [
                self._guide_sample_labels(column, tested_cells)
                for column in sample_columns
            ],
            len(tested),
            reason=f"Each sample gets an indicator, so {needing}",
        )
        extra: np.ndarray | None = None
        covariates_fingerprint = None
        if covariate_names:
            columns = []
            for column in covariate_names:
                values = np.asarray(
                    read_metadata_rows_chunkwise(self.cells, column, tested_cells)
                )
                if values.dtype.kind not in "biuf":
                    raise ValueError(f"covariate '{column}' must be numeric")
                # A missing value becomes NaN, which the covariates refuse.
                missing = read_metadata_missing_rows_chunkwise(
                    self.cells, column, tested_cells
                )
                columns.append(apply_missing_mask(values.astype(np.float64), missing))
            extra = np.column_stack(columns)
            covariates_fingerprint = fingerprint_array(extra)
        arguments = PerturbationAssociationArguments(
            labels=labels,
            dataset_fingerprint=self._ensure_dataset_fingerprint(expression.name),
            features=features,
            pairs_fingerprint=pairs_fingerprint,
            positive_controls_fingerprint=positive_fingerprint,
            sample_assignments_fingerprint=assignments,
            covariates_fingerprint=covariates_fingerprint,
            integration=integration,
            moi=moi,
            resampling=resolved_resampling,
            side=side,
            n_resamples=draws,
            stage_threshold=threshold,
            approximation=approximation,
            min_nonzero=floor,
            n_calibration_pairs=requested,
            alpha=rate,
            seed=resolved_seed,
            sample_columns=sample_columns,
            covariate_columns=covariate_names,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=expression.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                *(
                    ArrayRequirement(name, shape=(None,), dtype_kind="i")
                    for name in _ASSOCIATION_COUNTS
                ),
                *(
                    ArrayRequirement(name, shape=(None,), dtype_kind="f")
                    for name in _ASSOCIATION_VALUES
                ),
                ArrayRequirement("group_targets", shape=(None,)),
                ArrayRequirement("group_guides", shape=(None,)),
            ),
            required_attributes=(
                AttributeRequirement(
                    "set_names",
                    predicate=lambda value: list(value) == list(ASSOCIATION_SETS),
                ),
                AttributeRequirement("calibration_grade", expected_types=(str,)),
                AttributeRequirement(
                    "calibration_status",
                    predicate=lambda value: value in _CALIBRATION_STATUSES,
                ),
                *(
                    AttributeRequirement(name, expected_types=(int,))
                    for name in (
                        "n_false_discoveries",
                        "n_negative_control_pairs",
                        "n_control_guides",
                    )
                ),
            ),
        )
        if planned.reused:
            # Saved tests warn as they did when they were computed.
            for message in _association_warnings(
                unmatched,
                n_rows,
                by,
                expression.name,
                artifact_group(self.zw, planned.ref).attrs,
                moi,
            ):
                warn(message)
            return planned.ref
        if high:
            self._require_writable(arguments.operation)
            complement = self._complement_association_tables(
                labels,
                rows,
                by,
                used,
                discovery,
                positive,
                expression,
                tested_cells,
                samples if sample_columns else None,
                extra,
                covariate_names,
                n_resamples=draws,
                stage_threshold=threshold,
                approximation=approximation,
                side=side,
                min_nonzero=floor,
                n_calibration_pairs=requested,
                seed=resolved_seed,
                resampling=resolved_resampling,
            )
            return self._save_association_tables(
                planned,
                complement,
                rows=rows,
                by=by,
                used=used,
                unmatched=unmatched,
                assay=expression.name,
                alpha=rate,
                requested=requested,
                moi=moi,
            )
        calls = artifact_group(self.zw, status.input_ref("guide_calls"))
        guide_names = np.asarray(
            as_zarr_array(calls["guides"], name="guides")[:]
        ).astype(str)
        label_group = artifact_group(self.zw, labels)
        target_is_control = np.asarray(
            as_zarr_array(label_group["target_is_control"], name="target_is_control")[
                :
            ],
            dtype=bool,
        )
        guide_targets_of = rows.guide_targets
        # Each control cell's one guide, or -1 when it carries several control
        # guides; only the labels of the control cells are read.
        one_guide = pd.Index(guide_names).get_indexer(
            np.asarray(
                read_array_rows_chunkwise(
                    as_zarr_array(label_group["guide_label"], name="guide_label"),
                    controls,
                )
            ).astype(str)
        )
        # The control guides that are the one guide of a control cell make up
        # the calibration groups.
        control_guides = np.flatnonzero(
            (guide_targets_of >= 0) & target_is_control[np.maximum(guide_targets_of, 0)]
        )
        control_guides = control_guides[np.isin(control_guides, one_guide)]
        n_control_guides = len(control_guides)
        # A group leaves at least one control guide for its control cells.
        group_size = min(
            _calibration_group_size(rows, by, used), max(n_control_guides - 1, 1)
        )
        # A check runs unless it is turned off, or by default when there is no
        # discovery pair whose number it matches.
        checking = requested != 0 and (requested is not None or n_discovery > 0)
        nt_groups = (
            negative_control_groups(
                n_control_guides, group_size, _CALIBRATION_GROUPS, resolved_seed
            )
            if checking and n_control_guides >= 2
            else []
        )
        n_groups_nt = len(nt_groups)
        guide_of_control = pd.Index(control_guides).get_indexer(one_guide)
        n_controls = len(controls)
        n_tested = len(tested)
        # Each group of treatment cells: the tested rows, whose cells follow
        # the control cells, then the calibration groups among the controls.
        row_sizes = np.bincount(target_row[targets_cells], minlength=len(used))
        nt_members = [
            np.flatnonzero(np.isin(guide_of_control, members)) for members in nt_groups
        ]
        nt_sizes = np.asarray([len(cells) for cells in nt_members], dtype=np.int64)
        groups = TreatmentGroups(
            members=np.concatenate(
                [np.arange(n_controls, n_tested, dtype=np.int64), *nt_members]
            ).astype(np.int64),
            offsets=np.concatenate(
                [[0], np.cumsum(np.concatenate([row_sizes, nt_sizes]))]
            ).astype(np.int64),
            among_controls=np.concatenate(
                [np.zeros(len(used), bool), np.ones(n_groups_nt, bool)]
            ),
            codes=np.concatenate([used, n_rows + np.arange(n_groups_nt)]).astype(
                np.int64
            ),
            n_controls=n_controls,
        )
        del nt_members
        design = association_covariates(
            expression._cell_count_totals(tested_cells),
            read_metadata_rows_chunkwise(
                self.cells, f"{expression.name}_nFeatures", tested_cells
            ),
            samples if sample_columns else None,
            extra,
            names=covariate_names,
        )
        n_covariates = design.shape[1]
        # The units of the QC counts: each tested row, each control guide, and
        # the control cells without one control guide. Calibration pairs may
        # take any feature, so the QC reads every feature then.
        n_units = len(used) + n_control_guides + 1
        unit = np.concatenate(
            [
                len(used)
                + np.where(guide_of_control >= 0, guide_of_control, n_control_guides),
                target_row[targets_cells],
            ]
        ).astype(np.int64)
        target_features_checked = np.unique(np.concatenate([discovery[1], positive[1]]))
        checked = (
            np.arange(len(feature_ids), dtype=np.int64)
            if n_groups_nt
            else target_features_checked
        )
        largest = int(np.diff(groups.offsets).max(initial=0))
        # Every discovery and positive-control pair, and the most calibration
        # pairs that the check can draw.
        n_pairs_most = (
            n_discovery
            + len(positive[0])
            + _most_calibration_pairs(
                n_groups_nt, len(feature_ids), n_discovery, requested
            )
        )
        # Each labelled cell's position, row and flags; each tested cell's
        # positions, sample, unit and control guide, and its covariates as read
        # and as fitted; and the treatment cells of every group.
        resident = n_labelled * 48 + groups.members.nbytes
        resident += n_tested * (80 + 8 * (n_covariates + len(covariate_names)))
        resident += _ASSOCIATION_PAIR_BYTES * n_pairs_most
        resident += 4 * n_units * len(checked) + 9 * n_groups_nt * len(checked)
        # Each thread's buffers of one pair's fit and the records of a batch of
        # pairs, and one group's random sets of the largest round.
        n_union = n_controls + largest
        batch = record_batch(n_union, n_covariates)
        scratch = self.nthreads * (
            8 * n_union * (n_covariates + 4)
            + 8 * n_union * (n_covariates + 1) * batch
            + 8 * n_covariates * n_covariates * (batch + 4)
        )
        one_set = 4 * max(draws) * largest
        spare = max(self.memoryBytes - resident - scratch - one_set, 0)
        draw_bytes = one_set + min((self.nthreads - 1) * one_set, spare // 4)
        statistics_bytes = max(8 * draws[1], min(1 << 28, spare // 4))
        # One feature of a band: its float32 counts over the tested cells, the
        # block they are read from, and its QC counts and fit.
        per_feature = (
            24 * n_tested + 16 * n_units + 8 * n_covariates * (n_covariates + 2)
        )
        # The memory left for the counts: the second round holds its
        # statistics and one working copy no larger than them, so twice
        # statistics_bytes is kept for it.
        free = max(
            self.memoryBytes - resident - scratch - draw_bytes - 2 * statistics_bytes,
            0,
        )
        band_cap = max(1, _MIXSCAPE_BAND_BYTES // (8 * max(n_tested, 1)))
        width = min(max(len(checked), 1), free // per_feature, band_cap)
        if width < 1:
            raise MemoryError(
                f"The counts of one feature over {n_tested} tested cells hold about "
                f"{self.memoryBytes - free + per_feature} bytes with the tests and "
                f"their random sets, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        # The counts are read once and held for the tests when they take at
        # most a share of the memory left, so that the reads keep the rest;
        # otherwise the tests read them again.
        held_bytes = 4 * n_tested * len(checked)
        hold = held_bytes <= _ASSOCIATION_HELD_SHARE * free
        self._require_writable(arguments.operation)
        order = np.argsort(tested_cells, kind="stable")
        sorted_cells = tested_cells[order]
        # The reads keep clear of the held counts, which fill as they go.
        holding = held_bytes if hold else 0

        def counts_of(band: np.ndarray, out: np.ndarray) -> np.ndarray:
            """Write the counts of features over the tested cells, features x cells."""
            return self._association_counts(
                expression,
                band,
                sorted_cells,
                order,
                out,
                resident + scratch + holding + per_feature * len(band),
            )

        held: np.ndarray | None = None
        if hold:
            width = min(width, max(1, (free - held_bytes) // per_feature))
            held = np.empty((len(checked), n_tested), dtype=np.float32)
        detected = np.zeros((n_units, len(checked)), dtype=np.int32)
        with _numba_threads(self.nthreads):
            for start in range(0, len(checked), width):
                band = checked[start : start + width]
                detected[:, start : start + len(band)] = nonzero_counts(
                    counts_of(
                        band,
                        np.empty((len(band), n_tested), dtype=np.float32)
                        if held is None
                        else held[start : start + len(band)],
                    ),
                    unit,
                    n_units,
                )
        column_of = np.full(len(feature_ids), -1, dtype=np.int64)
        column_of[checked] = np.arange(len(checked))
        guide_detected = detected[len(used) : len(used) + n_control_guides]
        control_detected = detected[len(used) :].sum(axis=0, dtype=np.int64)

        def target_qc(
            pair_rows: np.ndarray, pair_features: np.ndarray, counted: np.ndarray
        ) -> tuple[np.ndarray, ...]:
            columns = column_of[pair_features]
            row_units = compact[pair_rows]
            return (
                row_sizes[row_units],
                np.full(len(pair_rows), n_controls, dtype=np.int64),
                counted[row_units, columns],
                control_detected[columns],
            )

        discovery_qc = target_qc(*discovery, detected)
        positive_qc = target_qc(*positive, detected)
        passing_discovery = int(
            np.count_nonzero((discovery_qc[2] >= floor) & (discovery_qc[3] >= floor))
        )
        if n_groups_nt:
            in_group = np.stack(
                [guide_detected[members].sum(axis=0) for members in nt_groups]
            )
            passing = (in_group >= floor) & (
                control_detected[None, :] - in_group >= floor
            )
            chosen_groups, chosen_columns = calibration_pairs(
                passing,
                passing_discovery if requested is None else requested,
                resolved_seed,
            )
            calibration_in = in_group[chosen_groups, chosen_columns].astype(np.int64)
            del in_group, passing
        else:
            chosen_groups = chosen_columns = np.empty(0, dtype=np.int64)
            calibration_in = np.empty(0, dtype=np.int64)
        calibration_features = checked[chosen_columns]
        calibration_qc = (
            nt_sizes[chosen_groups],
            n_controls - nt_sizes[chosen_groups],
            calibration_in,
            control_detected[chosen_columns] - calibration_in,
        )
        sets, pair_group, pair_feature, qc = _pair_sets(
            (compact[discovery[0]], discovery[1], discovery_qc),
            (len(used) + chosen_groups, calibration_features, calibration_qc),
            (compact[positive[0]], positive[1], positive_qc),
        )
        pass_qc = (qc[2] >= floor) & (qc[3] >= floor)
        n_pairs = len(sets)
        values_out = {name: np.full(n_pairs, np.nan) for name in _ASSOCIATION_VALUES}
        stage = np.zeros(n_pairs, dtype=np.int64)
        tested_features = np.unique(pair_feature[pass_qc])
        # With the counts held, every pair is tested at once; otherwise a band
        # of features at a time, read again.
        bands = (
            [tested_features]
            if held is not None
            else [
                tested_features[start : start + width]
                for start in range(0, len(tested_features), width)
            ]
        )
        with _numba_threads(self.nthreads):
            for band in bands:
                chosen = np.flatnonzero(pass_qc & np.isin(pair_feature, band))
                if held is not None:
                    values, rows_of = held, column_of[pair_feature[chosen]]
                else:
                    values = counts_of(
                        band, np.empty((len(band), n_tested), dtype=np.float32)
                    )
                    rows_of = np.searchsorted(band, pair_feature[chosen])
                result = pair_tests(
                    values,
                    design,
                    groups,
                    pair_group[chosen],
                    rows_of,
                    seed=resolved_seed,
                    n_resamples=draws,
                    stage_threshold=threshold,
                    approximation=approximation,
                    side=side,
                    statistics_bytes=statistics_bytes,
                    draw_bytes=draw_bytes,
                )
                for name in _ASSOCIATION_VALUES:
                    if name != "adjusted_p_value":
                        values_out[name][chosen] = getattr(result, name)
                stage[chosen] = result.stage
                del values, result
        # Only the pairs' arrays remain for the tables: the counts, the QC
        # counts and the inputs of the tests go first.
        del held, detected, guide_detected, design, groups
        tables = _AssociationTables(
            sets=sets,
            groups=pair_group,
            features=pair_feature,
            qc=qc,
            pass_qc=pass_qc,
            values=values_out,
            stage=stage,
            group_size=group_size,
            control_guides=guide_names[control_guides],
            calibration_groups=nt_groups,
            passing_discovery=passing_discovery,
        )
        return self._save_association_tables(
            planned,
            tables,
            rows=rows,
            by=by,
            used=used,
            unmatched=unmatched,
            assay=expression.name,
            alpha=rate,
            requested=requested,
            moi=moi,
        )

    def _association_counts(
        self,
        expression: Any,
        features: np.ndarray,
        sorted_cells: np.ndarray,
        order: np.ndarray,
        out: np.ndarray,
        resident_bytes: int,
    ) -> np.ndarray:
        """Write the counts of features over cells into ``out``, features x cells.

        The cells are read in store order, ``sorted_cells``, and ``order``
        gives the column of ``out`` that each of them fills.
        """
        from ...assay.normalization import feature_value_blocks
        from ...metadata.selection import NormalizationSpec

        _, blocks = feature_value_blocks(
            expression,
            features,
            sorted_cells,
            NormalizationSpec(source="raw"),
            nthreads=self.nthreads,
            resident_bytes=resident_bytes,
            scratch_bytes_per_row=8 * len(features),
        )
        offset = 0
        for block in blocks:
            out[:, order[offset : offset + len(block)]] = np.asarray(block).T
            offset += len(block)
        return out

    def _complement_association_tables(
        self,
        labels: ArtifactRef,
        rows: _LabelRows,
        by: str,
        used: np.ndarray,
        discovery: tuple[np.ndarray, np.ndarray],
        positive: tuple[np.ndarray, np.ndarray],
        expression: Any,
        cells: np.ndarray,
        samples: np.ndarray | None,
        extra: np.ndarray | None,
        covariate_names: tuple[str, ...],
        *,
        n_resamples: tuple[int, int, int],
        stage_threshold: float,
        approximation: str,
        side: str,
        min_nonzero: int,
        n_calibration_pairs: int | None,
        seed: int,
        resampling: str,
    ) -> _AssociationTables:
        """Test each tested row and calibration group against all other labelled cells.

        A row's treatment cells are the labelled cells whose guides target it,
        or with ``by="guide"`` that carry the guide, and a calibration group's
        those that carry any of its control guides; ``cells`` are every
        labelled cell's positions in the store. The counts are read once for
        the QC, every feature of the assay when a calibration check runs, and
        held for the tests when they fit half the memory left; the tests take
        bands of genes, each gene fitted once on all cells.
        """
        from ...perturbation.associations import (
            association_covariates,
            calibration_pairs,
            negative_control_groups,
        )
        from ...perturbation.high_moi import (
            ComplementGroups,
            complement_tests,
            fit_detection_models,
            nonzero_group_counts,
        )

        def read(stored: zarr.Group, name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(stored[name], name=name)[:])

        calls = artifact_group(self.zw, rows.status.input_ref("guide_calls"))
        label_group = artifact_group(self.zw, labels)
        guide_names = read(calls, "guides").astype(str)
        target_is_control = read(label_group, "target_is_control").astype(bool)
        guide_codes = read(calls, "guide_codes")
        n_cells = len(cells)
        n_rows = len(rows.names)
        n_discovery = len(discovery[0])
        floor = min_nonzero
        # The control guides that some labelled cell carries make up the
        # calibration groups, which hold the median target's guides, or every
        # control guide when there are fewer; one is enough, since a group is
        # compared with all other cells.
        called = np.zeros(len(guide_names), dtype=bool)
        called[guide_codes[guide_codes >= 0]] = True
        targets_of = rows.guide_targets
        control_guides = np.flatnonzero(
            called & (targets_of >= 0) & target_is_control[np.maximum(targets_of, 0)]
        )
        n_control_guides = len(control_guides)
        group_size = min(
            _calibration_group_size(rows, by, used), max(n_control_guides, 1)
        )
        # A check runs unless it is turned off, or by default when there is no
        # discovery pair whose number it matches.
        checking = n_calibration_pairs != 0 and (
            n_calibration_pairs is not None or n_discovery > 0
        )
        nt_groups = (
            negative_control_groups(
                n_control_guides, group_size, _CALIBRATION_GROUPS, seed
            )
            if checking and n_control_guides
            else []
        )
        n_groups_nt = len(nt_groups)
        # Each tested row's cells: the cells whose target codes hold its
        # target, or whose called guides hold its guide.
        if by == "target":
            row_codes = read(label_group, "target_codes")
            lookup = np.full(len(rows.target_names), -1, dtype=np.int64)
            lookup[rows.codes[used]] = np.arange(len(used))
        else:
            row_codes = guide_codes
            lookup = np.full(len(guide_names), -1, dtype=np.int64)
            lookup[pd.Index(guide_names).get_indexer(rows.names[used])] = np.arange(
                len(used)
            )
        row_members, row_offsets = _cells_by_code(
            np.where(row_codes >= 0, lookup[np.maximum(row_codes, 0)], -1), len(used)
        )
        # Each calibration group's cells: those that carry any of its guides.
        slot = np.full(len(guide_names), -1, dtype=np.int64)
        slot[control_guides] = np.arange(n_control_guides)
        guide_members, guide_offsets = _cells_by_code(
            np.where(guide_codes >= 0, slot[np.maximum(guide_codes, 0)], -1),
            n_control_guides,
        )
        del row_codes, guide_codes
        nt_members = [
            np.unique(
                np.concatenate(
                    [
                        guide_members[guide_offsets[g] : guide_offsets[g + 1]]
                        for g in group
                    ]
                )
            )
            for group in nt_groups
        ]
        del guide_members
        members = np.concatenate([row_members, *nt_members]).astype(np.int64)
        offsets = np.concatenate(
            [
                row_offsets,
                row_offsets[-1]
                + np.cumsum([len(cells_of) for cells_of in nt_members], dtype=np.int64),
            ]
        ).astype(np.int64)
        del row_members, nt_members
        sizes = np.diff(offsets)
        codes = np.concatenate([used, n_rows + np.arange(n_groups_nt)]).astype(np.int64)
        guide_assay = self._get_assay(str(labels.assay))
        design = association_covariates(
            expression._cell_count_totals(cells),
            read_metadata_rows_chunkwise(
                self.cells, f"{expression.name}_nFeatures", cells
            ),
            samples,
            extra,
            names=covariate_names,
            guide_umis=guide_assay._cell_count_totals(cells),
            guide_detected=read_metadata_rows_chunkwise(
                self.cells, f"{guide_assay.name}_nFeatures", cells
            ),
        )
        n_covariates = design.shape[1]
        # The QC counts each row's expressing cells in the features of its
        # pairs and each calibration group's in every feature, which its pairs
        # may take.
        n_features = int(expression.feats.N)
        target_features = np.unique(np.concatenate([discovery[1], positive[1]]))
        checked = (
            np.arange(n_features, dtype=np.int64) if n_groups_nt else target_features
        )
        column_of = np.full(n_features, -1, dtype=np.int64)
        column_of[checked] = np.arange(len(checked))
        target_column = np.full(n_features, -1, dtype=np.int64)
        target_column[target_features] = np.arange(len(target_features))
        n_pairs_most = (
            n_discovery
            + len(positive[0])
            + _most_calibration_pairs(
                n_groups_nt, n_features, n_discovery, n_calibration_pairs
            )
        )
        # Each labelled cell's positions, sample, totals and covariates as read
        # and as fitted; every group's cells, with the cells' groups that the
        # QC finds; every pair's arrays; and the QC counts.
        threads = self.nthreads
        resident = n_cells * (144 + 8 * (n_covariates + len(covariate_names)))
        resident += 32 * len(members)
        resident += _ASSOCIATION_PAIR_BYTES * n_pairs_most
        resident += 4 * len(used) * len(target_features)
        resident += 13 * n_groups_nt * len(checked) + 8 * len(checked)
        # Each thread draws a group's conditional sets in 24 bytes per cell
        # and copies some of its genes, 16 bytes per cell and gene, which also
        # bounds its gene and logistic fits; the second round holds its
        # statistics and one working copy no larger than them.
        one_copy = 16 * n_cells
        spare = max(
            self.memoryBytes
            - resident
            - threads * (24 * n_cells + one_copy)
            - 16 * n_resamples[1],
            0,
        )
        gather_bytes = one_copy + min(
            max(_ASSOCIATION_GATHER_BYTES - one_copy, 0), spare // (4 * threads)
        )
        statistics_bytes = max(8 * n_resamples[1], min(1 << 28, spare // 4))
        scratch = threads * (24 * n_cells + gather_bytes)
        free = max(self.memoryBytes - resident - scratch - 2 * statistics_bytes, 0)
        # A feature of the QC pass: its float32 counts, the block they are read
        # from and its QC counts. A gene of a band of tests: its scores and
        # weights on every cell and its fit, beside its counts when they are
        # read again.
        per_feature = 24 * n_cells + 16 * len(sizes)
        per_gene = 16 * n_cells + 8 * n_covariates * (n_covariates + 2)
        if free < per_feature + per_gene:
            raise MemoryError(
                f"Testing one gene over {n_cells} labelled cells holds about "
                f"{self.memoryBytes - free + per_feature + per_gene} bytes with the "
                "tests and their random sets, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        # The counts are read once and held for the tests when they take at
        # most a share of the memory left and leave room for a feature and a
        # gene; otherwise the tests read them again.
        held_bytes = 4 * n_cells * len(checked)
        hold = (
            held_bytes <= _ASSOCIATION_HELD_SHARE * free
            and free - held_bytes >= per_feature + per_gene
        )
        room = free - held_bytes if hold else free
        # A band holds at most so many bytes of float64 values, or of scores
        # and weights, whatever the budget.
        qc_width = min(
            len(checked),
            room // per_feature,
            max(1, _MIXSCAPE_BAND_BYTES // (8 * max(n_cells, 1))),
        )
        test_width = min(
            room // (per_gene + (0 if hold else per_feature)),
            max(1, _MIXSCAPE_BAND_BYTES // (16 * max(n_cells, 1))),
        )
        order = np.argsort(cells, kind="stable")
        sorted_cells = cells[order]
        # The reads keep clear of the held counts, which fill as they go.
        holding = held_bytes if hold else 0

        def counts_of(band: np.ndarray, out: np.ndarray) -> np.ndarray:
            """Write the counts of features over the labelled cells, features x cells."""
            return self._association_counts(
                expression,
                band,
                sorted_cells,
                order,
                out,
                resident + scratch + holding + per_feature * len(band),
            )

        proper = (sizes > 0) & (sizes < n_cells)

        def counted_groups(chosen: np.ndarray) -> ComplementGroups:
            """Return the groups of ``chosen`` that hold some cells but not all."""
            kept = chosen[proper[chosen]]
            return ComplementGroups(
                members=np.concatenate(
                    [np.empty(0, dtype=np.int64)]
                    + [members[offsets[g] : offsets[g + 1]] for g in kept]
                ),
                offsets=np.concatenate([[0], np.cumsum(sizes[kept])]).astype(np.int64),
                codes=np.arange(len(kept), dtype=np.int64),
            )

        row_groups = np.arange(len(used))
        nt_groups_of = len(used) + np.arange(n_groups_nt)
        row_counting = counted_groups(row_groups)
        nt_counting = counted_groups(nt_groups_of)

        def group_counts(
            values: np.ndarray, chosen: np.ndarray, counting: ComplementGroups
        ) -> tuple[np.ndarray, np.ndarray]:
            """Count the expressing cells of the groups ``chosen`` and of all cells.

            An empty group holds none of a feature's expressing cells and a
            group of every cell holds them all.
            """
            in_groups, totals = nonzero_group_counts(values, counting)
            counted = np.zeros((len(chosen), values.shape[0]), dtype=np.int64)
            counted[proper[chosen]] = in_groups
            counted[sizes[chosen] == n_cells] = totals
            return counted, totals

        held: np.ndarray | None = None
        if hold:
            held = np.empty((len(checked), n_cells), dtype=np.float32)
        row_detected = np.zeros((len(used), len(target_features)), dtype=np.int32)
        nt_detected = np.zeros((n_groups_nt, len(checked)), dtype=np.int32)
        expressing = np.zeros(len(checked), dtype=np.int64)
        with _numba_threads(self.nthreads):
            for start in range(0, len(checked), qc_width):
                band = checked[start : start + qc_width]
                span = slice(start, start + len(band))
                values = counts_of(
                    band,
                    np.empty((len(band), n_cells), dtype=np.float32)
                    if held is None
                    else held[span],
                )
                if n_groups_nt:
                    nt_detected[:, span], expressing[span] = group_counts(
                        values, nt_groups_of, nt_counting
                    )
                targeted = np.flatnonzero(target_column[band] >= 0)
                if len(targeted) == len(band):
                    counted, expressing[span] = group_counts(
                        values, row_groups, row_counting
                    )
                    row_detected[:, target_column[band]] = counted
                elif len(targeted):
                    counted, _ = group_counts(
                        values[targeted], row_groups, row_counting
                    )
                    row_detected[:, target_column[band[targeted]]] = counted
                del values
        del row_counting, nt_counting
        compact = np.full(n_rows, -1, dtype=np.int64)
        compact[used] = np.arange(len(used))
        row_sizes = sizes[: len(used)]
        nt_sizes = sizes[len(used) :]

        def target_qc(
            pair_rows: np.ndarray, pair_features: np.ndarray, counted: np.ndarray
        ) -> tuple[np.ndarray, ...]:
            groups_of = compact[pair_rows]
            in_group = counted[groups_of, target_column[pair_features]].astype(np.int64)
            return (
                row_sizes[groups_of],
                n_cells - row_sizes[groups_of],
                in_group,
                expressing[column_of[pair_features]] - in_group,
            )

        discovery_qc = target_qc(*discovery, row_detected)
        positive_qc = target_qc(*positive, row_detected)
        passing_discovery = int(
            np.count_nonzero((discovery_qc[2] >= floor) & (discovery_qc[3] >= floor))
        )
        if n_groups_nt:
            passing = (nt_detected >= floor) & (
                expressing[None, :] - nt_detected >= floor
            )
            chosen_groups, chosen_columns = calibration_pairs(
                passing,
                passing_discovery
                if n_calibration_pairs is None
                else n_calibration_pairs,
                seed,
            )
            calibration_in = nt_detected[chosen_groups, chosen_columns].astype(np.int64)
            del passing
        else:
            chosen_groups = chosen_columns = np.empty(0, dtype=np.int64)
            calibration_in = np.empty(0, dtype=np.int64)
        calibration_qc = (
            nt_sizes[chosen_groups],
            n_cells - nt_sizes[chosen_groups],
            calibration_in,
            expressing[chosen_columns] - calibration_in,
        )
        sets, pair_group, pair_feature, qc = _pair_sets(
            (compact[discovery[0]], discovery[1], discovery_qc),
            (len(used) + chosen_groups, checked[chosen_columns], calibration_qc),
            (compact[positive[0]], positive[1], positive_qc),
        )
        del row_detected, nt_detected
        pass_qc = (qc[2] >= floor) & (qc[3] >= floor)
        n_pairs = len(sets)
        values_out = {name: np.full(n_pairs, np.nan) for name in _ASSOCIATION_VALUES}
        stage = np.zeros(n_pairs, dtype=np.int64)
        # The groups of the tested pairs keep their codes, which key their
        # random sets, so that the sets do not depend on the other groups.
        active = np.unique(pair_group[pass_qc])
        engine_group = np.full(len(sizes), -1, dtype=np.int64)
        engine_group[active] = np.arange(len(active))
        engine = ComplementGroups(
            members=np.concatenate(
                [np.empty(0, dtype=np.int64)]
                + [members[offsets[g] : offsets[g + 1]] for g in active]
            ),
            offsets=np.concatenate([[0], np.cumsum(sizes[active])]).astype(np.int64),
            codes=codes[active],
        )
        tested_features = np.unique(pair_feature[pass_qc])
        with _numba_threads(self.nthreads):
            models = (
                fit_detection_models(design, engine)
                if resampling == "conditional" and len(active)
                else None
            )
            for start in range(0, len(tested_features), test_width):
                band = tested_features[start : start + test_width]
                chosen = np.flatnonzero(pass_qc & np.isin(pair_feature, band))
                if held is not None:
                    values, rows_of = held, column_of[pair_feature[chosen]]
                else:
                    values = counts_of(
                        band, np.empty((len(band), n_cells), dtype=np.float32)
                    )
                    rows_of = np.searchsorted(band, pair_feature[chosen])
                result = complement_tests(
                    values,
                    design,
                    engine,
                    engine_group[pair_group[chosen]],
                    rows_of,
                    resampling=resampling,
                    detection_models=models,
                    seed=seed,
                    n_resamples=n_resamples,
                    stage_threshold=stage_threshold,
                    approximation=approximation,
                    side=side,
                    statistics_bytes=statistics_bytes,
                    gather_bytes=gather_bytes,
                )
                for name in _ASSOCIATION_VALUES:
                    if name != "adjusted_p_value":
                        values_out[name][chosen] = getattr(result, name)
                stage[chosen] = result.stage
                del values, result
        return _AssociationTables(
            sets=sets,
            groups=pair_group,
            features=pair_feature,
            qc=qc,
            pass_qc=pass_qc,
            values=values_out,
            stage=stage,
            group_size=group_size,
            control_guides=guide_names[control_guides],
            calibration_groups=nt_groups,
            passing_discovery=passing_discovery,
        )

    def _save_association_tables(
        self,
        planned: PlannedArtifact,
        tables: _AssociationTables,
        *,
        rows: _LabelRows,
        by: str,
        used: np.ndarray,
        unmatched: np.ndarray,
        assay: str,
        alpha: float,
        requested: int | None,
        moi: str,
    ) -> ArtifactRef:
        """Adjust the tested pairs' p-values, grade the calibration and save the tests.

        The p-values are adjusted by Benjamini-Hochberg within each set of
        pairs. The attributes also hold what the warnings name, so that saved
        tests warn again as they did.
        """
        from ...perturbation.associations import (
            ASSOCIATION_SETS,
            adjusted_p_values,
            calibration_grade,
        )

        sets, pass_qc, values_out = tables.sets, tables.pass_qc, tables.values
        n_calibration = int(np.count_nonzero(sets == 1))
        significant = np.zeros(len(sets), dtype=bool)
        for code in range(len(ASSOCIATION_SETS)):
            members = np.flatnonzero(sets == code)
            adjusted, found = adjusted_p_values(
                values_out["p_value"][members], pass_qc[members], alpha
            )
            values_out["adjusted_p_value"][members] = adjusted
            significant[members] = found
        n_false = int(np.count_nonzero(significant & (sets == 1)))
        grade = calibration_grade(n_false) if n_calibration else "none"
        n_control_guides = len(tables.control_guides)
        # At low MOI a group leaves one control guide for its comparison
        # cells; at high MOI all other cells are its comparison.
        if n_calibration:
            calibration_status = "checked"
        elif requested == 0:
            calibration_status = "off"
        elif n_control_guides < (1 if moi == "high" else 2):
            calibration_status = "few_control_guides"
        elif requested is None and not tables.passing_discovery:
            calibration_status = "no_passing_discovery"
        else:
            calibration_status = "no_passing_calibration"
        attributes = {
            "set_names": list(ASSOCIATION_SETS),
            "calibration_grade": grade,
            "n_false_discoveries": n_false,
            "calibration_group_size": tables.group_size,
            "calibration_status": calibration_status,
            "n_control_guides": n_control_guides,
            "n_negative_control_pairs": n_calibration,
        }
        # Each group of treatment cells, the tested rows and then the
        # calibration groups, names its target and guides once, and each pair
        # names its group.
        n_groups_nt = len(tables.calibration_groups)
        row_targets = rows.names if by == "target" else rows.target_names[rows.codes]
        group_targets = np.concatenate(
            [
                row_targets[used].astype(object),
                np.asarray(
                    [f"control group {group + 1}" for group in range(n_groups_nt)],
                    dtype=object,
                ),
            ]
        )
        group_guides = np.concatenate(
            [
                np.full(len(used), "", dtype=object)
                if by == "target"
                else rows.names[used].astype(object),
                np.asarray(
                    [
                        "+".join(tables.control_guides[members])
                        for members in tables.calibration_groups
                    ],
                    dtype=object,
                ),
            ]
        )
        qc = tables.qc
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "sets": sets,
                "groups": tables.groups,
                "group_targets": group_targets,
                "group_guides": group_guides,
                "features": tables.features,
                "n_cells": qc[0],
                "n_control_cells": qc[1],
                "n_nonzero_cells": qc[2],
                "n_nonzero_control_cells": qc[3],
                "pass_qc": pass_qc,
                "significant": significant,
                "stage": tables.stage,
                **values_out,
            },
            attributes=attributes,
        )
        for message in _association_warnings(
            unmatched, len(rows.names), by, assay, attributes, moi
        ):
            warn(message)
        logger.info(
            f"Tested {int(np.count_nonzero(pass_qc[sets == 0]))} of "
            f"{int(np.count_nonzero(sets == 0))} discovery pairs: "
            f"{int(np.count_nonzero(significant[sets == 0]))} significant at "
            f"{alpha}; calibration {grade} with {n_false} false discoveries in "
            f"{n_calibration} pairs"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_associations(
        self, tests: ArtifactRef
    ) -> PerturbationAssociations:
        """Read saved association tests.

        Args:
            tests: A ``perturbation_associations`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.associations`.

        Returns:
            The discovery, calibration and positive-control pairs.

        Raises:
            TypeError: If ``tests`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_associations`` artifact,
                or a development build saved it before high-MOI tests, without
                their ``moi`` and ``resampling``.
        """
        from ...perturbation.associations import ASSOCIATION_SETS

        if not isinstance(tests, ArtifactRef):
            raise TypeError("tests must be an ArtifactRef")
        if tests.kind != PerturbationAssociationArguments.artifact_kind:
            raise ValueError(
                "tests must be a perturbation_associations artifact, not "
                f"{tests.kind!r}"
            )
        status = require_complete_artifact(self.zw, tests)
        parameters = status.parameters or {}
        if "moi" not in parameters or "resampling" not in parameters:
            raise ValueError(
                "These association tests were saved by a development build before "
                "high-MOI tests, so their record lacks moi and resampling; call "
                "perturbation.associations again with the same arguments, which "
                "tests again and saves new tests"
            )
        group = artifact_group(self.zw, tests)
        assay = self._get_assay(str(tests.assay))

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        integration = str(parameters["integration"])
        feature_positions = read("features")
        # Each pair names its group of treatment cells, whose target and
        # guides are stored once.
        groups = read("groups")
        group_guides = read("group_guides").astype(str)
        table: dict[str, Any] = {
            "target": read("group_targets").astype(str)[groups],
            "feature": np.asarray(assay.feats.fetch_all("ids")).astype(str)[
                feature_positions
            ],
            "feature_name": np.asarray(assay.feats.fetch_all("names")).astype(str)[
                feature_positions
            ],
        }
        for name in (
            "n_cells",
            "n_control_cells",
            "n_nonzero_cells",
            "n_nonzero_control_cells",
            "pass_qc",
            "z",
            "p_value",
            "adjusted_p_value",
            "significant",
            "log2_fold_change",
            "se_log2_fold_change",
            "stage",
            "skew_location",
            "skew_scale",
            "skew_shape",
        ):
            table[name] = read(name)
        frame = pd.DataFrame(table)
        sets = read("sets")
        parts = {}
        for code, name in enumerate(ASSOCIATION_SETS):
            members = sets == code
            part = frame[members].reset_index(drop=True)
            if name == "calibration":
                part.insert(1, "control_guides", group_guides[groups[members]])
            elif integration == "singleton":
                part.insert(1, "guide", group_guides[groups[members]])
            parts[name] = part
        return PerturbationAssociations(
            discovery=parts["discovery"],
            calibration=parts["calibration"],
            positive=parts["positive"],
            alpha=float(parameters["alpha"]),
            side=str(parameters["side"]),
            integration=integration,
            moi=str(parameters["moi"]),
            resampling=str(parameters["resampling"]),
            from_assay=str(tests.assay),
            n_false_discoveries=int(str(group.attrs["n_false_discoveries"])),
            calibration_grade=str(group.attrs["calibration_grade"]),
        )

    def _profile_cells(self, labels: ArtifactRef, by: str) -> _ProfileCells:
        """Read the rows and cells of perturbation profiles.

        The rows are the targets of the labels that are not controls, or with
        ``by="guide"`` every guide with a target, the control guides included.
        A row's cells carry its target, or its guide, alone. With Mixscape
        classes, the cells are those of the classes, and of a row that is not
        a control guide only those that Mixscape called knocked out count.

        Raises:
            ValueError: If no labelled cell carries control guides only.
        """

        def selected_cells(selection: ArtifactRef) -> np.ndarray:
            return read_stored_selection_indices(
                self.zw,
                selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)

        classes = None
        perturbation_labels = labels
        if labels.kind == MixscapeArguments.artifact_kind:
            classes = require_complete_artifact(self.zw, labels)
            perturbation_labels = require_complete_artifact(
                self.zw, classes.input_ref("signature")
            ).input_ref("labels")
        rows = self._label_rows(perturbation_labels, by, control_guides=True)
        groups, control = self._label_groups(perturbation_labels, rows, by)
        row_control = np.asarray(
            as_zarr_array(
                artifact_group(self.zw, perturbation_labels)["target_is_control"],
                name="target_is_control",
            )[:],
            dtype=bool,
        )[rows.codes]
        selection = rows.status.input_ref("cell_selection")
        cell_index = selected_cells(selection)
        if classes is not None:
            # The cells of the classes, with the rows and flags of those that
            # the labels cover.
            selection = classes.input_ref("cell_selection")
            classified = selected_cells(selection)
            positions = np.searchsorted(classified, cell_index)
            present = positions < len(classified)
            present[present] = classified[positions[present]] == cell_index[present]
            aligned = np.full(len(classified), -1, dtype=np.int64)
            aligned[positions[present]] = groups[present]
            control_flags = np.zeros(len(classified), dtype=bool)
            control_flags[positions[present]] = control[present]
            knocked_out = np.asarray(
                self._artifacts_load_values(labels, value="global_class").values
            ).astype(str) == str((classes.parameters or {})["perturbation_type"])
            kept = aligned >= 0
            kept[kept] = knocked_out[kept] | row_control[aligned[kept]]
            cell_index, groups, control = (
                classified,
                np.where(kept, aligned, -1),
                control_flags,
            )
        return _ProfileCells(
            labels=perturbation_labels,
            rows=rows,
            row_control=row_control,
            cell_selection=selection,
            cell_index=cell_index,
            groups=groups,
            control=control,
        )

    @validation_scoped
    def _perturbation_profiles(
        self,
        labels: ArtifactRef,
        features: ArtifactRef,
        *,
        by: str = "target",
        sample_column: str | None = None,
        min_cells: int = 25,
        mask_targets: bool = True,
        id_column: str | None = "target_gene_id",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Average each perturbation's z-normalized expression into its profile.

        Replogle et al. (2022) compare perturbations by their mean expression
        profiles, each feature measured against the control cells:

        1. Each cell's counts in ``features`` are scaled so that its total,
           the sum of its stored counts over every feature of the assay,
           equals the median total of the control cells of the labels.
        2. Within each sample of ``sample_column``, such as a GEM group, each
           feature is z-normalized against the sample's control cells,
           ``z = (x - mean) / sd``, with the standard deviation's n - 1
           divisor, for every cell, control cells included.
        3. A perturbation's profile is the mean z-score of its cells over all
           samples, computed in float64 and stored as float32.

        This recipe reproduces the published profiles of Replogle et al.'s K562
        essential screen from its cells to float32 rounding. A feature whose
        control cells in a sample hold one value has no spread there, and its
        z-scores in that sample are 0, CyteArc's choice where Replogle et al.'s
        genome-wide profiles hold infinite values; ``load_profiles`` counts
        such samples for each feature. A sample needs 2 control cells or more.

        A perturbation is a target of the labels that is not a control, or
        with ``by="guide"`` a guide, each control guide included, as Replogle
        et al.'s profiles have a row for each guide pair, control pairs
        included; with targets, the control cells get no row, since their mean
        z-score is 0. A perturbation's cells carry its target, or its guide,
        alone. With Mixscape classes as ``labels``, only the cells that
        Mixscape called knocked out count, among the cells of the classes, so
        that escapers do not dilute the profiles; a control guide keeps its
        cells. A perturbation with fewer than ``min_cells`` cells, 25 as in
        Replogle et al., gets no profile and is named in a warning, which a
        reused result repeats. With ``mask_targets``, each profile's own
        target feature is set to 0, the control mean, so that the knockdown
        of the target itself does not make two perturbations alike; a target
        is matched to a feature as
        :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.efficiency`
        matches it, and control guides mask nothing.

        The counts are read twice, the control cells and then the cells of
        the profiled perturbations, in bands of features that fit the memory
        budget, and the profiles do not depend on the bands or the threads.

        Args:
            labels: A ``perturbation_labels`` artifact, or a
                ``mixscape_classes`` artifact whose knocked-out cells alone
                count.
            features: A ``feature_selection`` artifact, such as highly
                variable genes, whose features the profiles hold; its assay is
                the expression assay.
            by: ``"target"`` for one profile per target gene, or ``"guide"``
                for one per guide, control guides included.
            sample_column: A cell metadata column whose samples are
                z-normalized against their own control cells, or None for one
                sample.
            min_cells: The fewest cells of a profiled perturbation.
            mask_targets: Set each profile's own target feature to 0.
            id_column: The feature column of the guide assay that holds each
                guide's target feature ID, or None to match targets by name
                only; it matters only with ``mask_targets``.
            invalidate_cache: Compute again instead of reusing saved
                profiles.

        Returns:
            A complete ``perturbation_profiles`` artifact scoped to the assay
            of ``features``, with one profile per perturbation. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_profiles`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``labels`` or ``features`` is not an artifact of
                its kinds, ``by`` is unknown, ``min_cells`` is below 1,
                ``sample_column`` is not a cell metadata column or leaves a
                control cell or a cell of a profiled perturbation without a
                sample, a sample has fewer than 2 control cells, ``id_column``
                is not a feature column of the guide assay, the labels have no
                control cell, or no perturbation has ``min_cells`` cells.
            PermissionError: If the store is read-only and the profiles are
                not saved yet.
            MemoryError: If the profiles and one feature's sums do not fit the
                memory budget.
        """
        from ...assay.normalization import feature_value_blocks
        from ...metadata.selection import NormalizationSpec
        from ...perturbation.profiles import (
            CONTROL_SD_DDOF,
            control_moments,
            depth_target,
            profile_values,
            z_score_sums,
        )
        from ...storage.feature_selection import read_feature_selection_indices

        if not isinstance(labels, ArtifactRef):
            raise TypeError("labels must be an ArtifactRef")
        if labels.kind not in (
            PerturbationLabelArguments.artifact_kind,
            MixscapeArguments.artifact_kind,
        ):
            raise ValueError(
                "labels must be a perturbation_labels or mixscape_classes "
                f"artifact, not {labels.kind!r}"
            )
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if features.kind != "feature_selection":
            raise ValueError(
                f"features must be a feature_selection artifact, not {features.kind!r}"
            )
        floor = profile_settings(by, min_cells, mask_targets, id_column, sample_column)
        if sample_column is not None:
            if sample_column not in self.cells.columns:
                raise ValueError(
                    f"sample_column '{sample_column}' not found in cell metadata"
                )
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        cells = self._profile_cells(labels, by)
        expression = self._get_assay(str(features.assay))
        self._require_measured_cells(
            expression.name,
            cells.cell_selection,
            operation=PerturbationProfileArguments.operation,
        )
        feature_index = np.asarray(
            read_feature_selection_indices(self.zw, expression.name, features),
            dtype=np.int64,
        )
        n_features = len(feature_index)
        rows = cells.rows
        n_rows = len(rows.names)
        grouped = cells.groups >= 0
        sizes = np.bincount(cells.groups[grouped], minlength=n_rows)
        kept = np.flatnonzero(sizes >= floor)
        thin = np.flatnonzero(sizes < floor)
        if labels.kind == PerturbationLabelArguments.artifact_kind:
            noun = "cells"
        elif by == "target":
            noun = "knocked-out cells"
        else:
            noun = "knocked-out or control cells"
        if not len(kept):
            raise ValueError(
                f"No {by} has {floor} {noun} or more, so none gets a profile; "
                "lower min_cells"
            )
        n_kept = len(kept)
        names = rows.names[kept].astype(str)
        row_targets = rows.target_names[rows.codes].astype(str)
        feature_ids = np.asarray(expression.feats.fetch_all("ids")).astype(str)
        # Each profile's masked feature, as a position in the assay.
        masked = np.full(n_kept, -1, dtype=np.int64)
        targets_fingerprint = None
        if mask_targets:
            guide_feature_ids = self._guide_feature_column(
                self._get_assay(str(cells.labels.assay)), id_column, "id_column"
            )
            matched = target_features(
                rows.target_names,
                rows.guide_targets,
                guide_feature_ids,
                feature_ids,
                np.asarray(expression.feats.fetch_all("names")).astype(str),
            )[rows.codes[kept]]
            # A control target is no gene of the genome.
            matched[cells.row_control[kept]] = -1
            targets_fingerprint = fingerprint_strings(
                np.char.add(
                    np.char.add(names, "\t"),
                    np.where(matched >= 0, feature_ids[np.maximum(matched, 0)], ""),
                )
            )
            # Only a target among the profiles' features has a value to mask.
            inside = np.isin(matched, feature_index)
            masked[inside] = matched[inside]
        slot = np.full(n_rows, -1, dtype=np.int64)
        slot[kept] = np.arange(n_kept)
        groups = np.where(grouped, slot[np.maximum(cells.groups, 0)], -1)
        # The cells read: the control cells and the cells of the profiled
        # perturbations, which by guide include the control cells of a
        # control guide, in store order.
        used = np.flatnonzero(cells.control | (groups >= 0))
        used_cells = cells.cell_index[used]
        is_control = cells.control[used]
        columns = () if sample_column is None else (sample_column,)
        samples, sample_labels, assignments = sample_codes(
            columns,
            [self._guide_sample_labels(column, used_cells) for column in columns],
            len(used),
            reason=(
                "Each sample's control cells z-normalize its cells, so every "
                "control cell and every cell of a profiled perturbation needs one"
            ),
        )
        sample_names = [
            None if sample_column is None else labels_by_column[sample_column]
            for labels_by_column in sample_labels
        ]
        n_samples = len(sample_names)
        n_controls = np.bincount(samples[is_control], minlength=n_samples)
        # The standard deviation's divisor needs this many control cells.
        needed = CONTROL_SD_DDOF + 1
        short = np.flatnonzero(n_controls < needed)
        if len(short):
            where = (
                "the labels have"
                if sample_column is None
                else f"sample {sample_names[short[0]]!r} has"
            )
            raise ValueError(
                f"The control standard deviations of a sample need {needed} "
                f"control cells or more, but {where} "
                f"{int(n_controls[short[0]])}; pass a sample_column whose "
                f"samples each hold {needed} or more, or label more control cells"
            )
        arguments = PerturbationProfileArguments(
            labels=labels,
            feature_selection=features,
            dataset_fingerprint=self._ensure_dataset_fingerprint(expression.name),
            sample_assignments_fingerprint=assignments,
            target_features_fingerprint=targets_fingerprint,
            by=by,
            min_cells=floor,
            mask_targets=mask_targets,
            sample_columns=columns,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=expression.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("values", shape=(n_kept, n_features), dtype_kind="f"),
                *(
                    ArrayRequirement(name, shape=(n_kept,))
                    for name in ("perturbations", "targets")
                ),
                ArrayRequirement("is_control", shape=(n_kept,), dtype_kind="b"),
                *(
                    ArrayRequirement(name, shape=(n_kept,), dtype_kind="i")
                    for name in ("n_cells", "masked_features")
                ),
                *(
                    ArrayRequirement(name, shape=(len(thin),))
                    for name in ("left_out", "left_out_targets")
                ),
                ArrayRequirement(
                    "left_out_n_cells", shape=(len(thin),), dtype_kind="i"
                ),
                *(
                    ArrayRequirement(name, shape=(n_features,), dtype_kind="i")
                    for name in ("features", "n_zero_sd")
                ),
                ArrayRequirement("n_control_cells", shape=(n_samples,), dtype_kind="i"),
                *(
                    ArrayRequirement(
                        name, shape=(n_samples, n_features), dtype_kind="f"
                    )
                    for name in ("control_mean", "control_sd")
                ),
            ),
            required_attributes=(
                AttributeRequirement(
                    "sample_names", predicate=lambda value: value == sample_names
                ),
            ),
        )
        left_out = _left_out_warning(
            rows.names[thin].astype(str), sizes[thin], n_rows, floor, f"{by}s", noun
        )
        if planned.reused:
            # Saved profiles warn as they did when they were computed.
            if left_out is not None:
                warn(left_out)
            return planned.ref
        # Each labelled cell's position, perturbation and flag, each read
        # cell's position, sample and total, the float32 profiles, and each
        # sample's control mean and standard deviation of every feature.
        held = (
            24 * len(cells.cell_index)
            + 32 * len(used)
            + 4 * n_kept * n_features
            + 16 * n_samples * n_features
            + 16 * n_features
        )
        # One feature of a band: its float64 sums over the perturbations,
        # which are divided straight into the profiles counted above, its
        # running moments in each sample, and a cell's value as read and as a
        # C-ordered copy.
        per_feature = 8 * n_kept + 24 * n_samples + 16
        width = min(n_features, max(self.memoryBytes - held, 0) // per_feature)
        if width < 1:
            raise MemoryError(
                f"The profiles of {n_kept} {by}s in {n_features} features hold "
                f"about {held + per_feature} bytes with one feature's sums, more "
                f"than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget, or pass fewer features."
            )
        self._require_writable(arguments.operation)
        totals = expression._cell_count_totals(used_cells)
        target = depth_target(totals, is_control)
        controls = np.flatnonzero(is_control)
        members = np.flatnonzero(groups[used] >= 0)
        member_groups = groups[used][members]
        n_cells = sizes[kept].astype(np.int64)
        masked_columns = np.full(n_kept, -1, dtype=np.int64)
        masked_columns[masked >= 0] = np.searchsorted(
            feature_index, masked[masked >= 0]
        )
        values = np.empty((n_kept, n_features), dtype=np.float32)
        means = np.empty((n_samples, n_features))
        sds = np.empty((n_samples, n_features))

        def counts(band: np.ndarray, positions: np.ndarray) -> Iterator[np.ndarray]:
            """Stream the counts of a band of features over some read cells."""
            _, blocks = feature_value_blocks(
                expression,
                band,
                used_cells[positions],
                NormalizationSpec(source="raw"),
                nthreads=self.nthreads,
                resident_bytes=held + (per_feature - 16) * len(band),
                scratch_bytes_per_row=16 * len(band),
            )
            yield from blocks

        with _numba_threads(self.nthreads):
            for start in range(0, n_features, width):
                stop = min(start + width, n_features)
                band = feature_index[start:stop]
                _, means[:, start:stop], sds[:, start:stop] = control_moments(
                    counts(band, controls),
                    totals[controls],
                    target,
                    samples[controls],
                    n_samples,
                    len(band),
                )
                sums = z_score_sums(
                    counts(band, members),
                    totals[members],
                    target,
                    samples[members],
                    member_groups,
                    means[:, start:stop],
                    sds[:, start:stop],
                    n_kept,
                )
                inside = (masked_columns >= start) & (masked_columns < stop)
                # The means go straight into the band of the float32 profiles.
                profile_values(
                    sums,
                    n_cells,
                    np.where(inside, masked_columns - start, -1),
                    out=values[:, start:stop],
                )
                del sums
        n_zero_sd = np.count_nonzero(sds == 0, axis=0).astype(np.int64)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": values,
                "perturbations": names,
                "targets": row_targets[kept],
                "is_control": cells.row_control[kept],
                "n_cells": n_cells,
                "masked_features": masked,
                "left_out": rows.names[thin].astype(str),
                "left_out_targets": row_targets[thin],
                "left_out_n_cells": sizes[thin].astype(np.int64),
                "features": feature_index,
                "n_zero_sd": n_zero_sd,
                "n_control_cells": n_controls.astype(np.int64),
                "control_mean": means,
                "control_sd": sds,
            },
            attributes={"sample_names": sample_names},
        )
        if left_out is not None:
            warn(left_out)
        flat = int(np.count_nonzero(n_zero_sd))
        logger.info(
            f"Computed the profiles of {n_kept} {by}s in {n_features} features "
            f"against {len(controls)} control cells in {n_samples} samples"
            + (
                f"; {flat} features hold one value over the control cells of a "
                "sample, where their z-scores are 0"
                if flat
                else ""
            )
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_profiles(
        self, profiles: ArtifactRef
    ) -> PerturbationProfiles:
        """Read saved perturbation profiles.

        Args:
            profiles: A ``perturbation_profiles`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.profiles`.

        Returns:
            Each perturbation's profile with its cells and masked feature, the
            perturbations left out, and each sample's control cells and
            control statistics of each feature.

        Raises:
            TypeError: If ``profiles`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_profiles`` artifact.
            MemoryError: If its profiles do not fit the memory budget.
        """
        if not isinstance(profiles, ArtifactRef):
            raise TypeError("profiles must be an ArtifactRef")
        if profiles.kind != PerturbationProfileArguments.artifact_kind:
            raise ValueError(
                "profiles must be a perturbation_profiles artifact, not "
                f"{profiles.kind!r}"
            )
        status = require_complete_artifact(self.zw, profiles)
        group = artifact_group(self.zw, profiles)
        assay = self._get_assay(str(profiles.assay))

        def read(name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(group[name], name=name)[:])

        stored = as_zarr_array(group["values"], name="values")
        # The frame holds the float32 values as read, without a copy.
        n_bytes = 4 * int(np.prod(stored.shape))
        if n_bytes > self.memoryBytes:
            raise MemoryError(
                f"The profiles of {stored.shape[0]} perturbations in "
                f"{stored.shape[1]} features hold {n_bytes} bytes, more than the "
                f"memory budget of {self.memoryBytes} bytes. Raise mem_budget."
            )
        feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
        positions = read("features")
        columns = pd.Index(feature_ids[positions], name="feature")
        index = pd.Index(read("perturbations").astype(str), name="perturbation")
        masked = read("masked_features")
        samples = pd.Index(
            np.asarray(_stored_samples(group, "sample_names"), dtype=object),
            name="sample",
        )
        parameters = status.parameters or {}
        sample_columns = list(parameters["sample_columns"])
        return PerturbationProfiles(
            values=pd.DataFrame(
                np.asarray(stored[:]), index=index, columns=columns, copy=False
            ),
            perturbations=pd.DataFrame(
                {
                    "target": read("targets").astype(str),
                    "is_control": read("is_control").astype(bool),
                    "n_cells": read("n_cells"),
                    "masked_feature": pd.Series(
                        [
                            None if code < 0 else str(feature_ids[code])
                            for code in masked
                        ],
                        dtype=object,
                        index=index,
                    ),
                },
                index=index,
            ),
            left_out=pd.DataFrame(
                {
                    "perturbation": read("left_out").astype(str),
                    "target": read("left_out_targets").astype(str),
                    "n_cells": read("left_out_n_cells"),
                }
            ),
            features=pd.DataFrame(
                {
                    "feature_name": np.asarray(assay.feats.fetch_all("names")).astype(
                        str
                    )[positions],
                    "n_zero_sd": read("n_zero_sd"),
                },
                index=columns,
            ),
            samples=pd.DataFrame(
                {"n_control_cells": read("n_control_cells")}, index=samples
            ),
            control_mean=pd.DataFrame(
                read("control_mean"), index=samples, columns=columns
            ),
            control_sd=pd.DataFrame(read("control_sd"), index=samples, columns=columns),
            by=str(parameters["by"]),
            from_assay=assay.name,
            sample_column=sample_columns[0] if sample_columns else None,
            min_cells=int(parameters["min_cells"]),
            mask_targets=bool(parameters["mask_targets"]),
            labels_kind=str(status.input_ref("labels").kind),
        )

    @validation_scoped
    def _perturbation_cluster(
        self,
        profiles: ArtifactRef,
        *,
        perturbations: Sequence[str] | None = None,
        metric: str = "correlation",
        min_cluster_size: int = 4,
        min_samples: int = 2,
        selection: str = "eom",
        seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Cluster the perturbations whose profiles change expression alike.

        As Replogle et al. (2022) did, the distance between two profiles is 1
        minus their Pearson correlation over the features, which ignores how
        strongly each perturbation acts; ``metric="euclidean"`` keeps it.
        HDBSCAN finds the clusters on these distances, through scikit-learn's
        ``HDBSCAN(metric="precomputed")`` with ``alpha=1`` and without a
        single cluster, and a perturbation it cannot place gets the cluster
        -1. ``min_samples`` follows scikit-learn, which counts the point
        itself, so the ``min_samples=1`` of the ``hdbscan`` package, which
        Replogle et al. ran with ``min_cluster_size=4`` and
        ``selection="eom"``, is 2 here, the default. Average linkage of the
        same distances orders the perturbations for a heatmap. A UMAP layout
        of each perturbation's 15 nearest perturbations by the same distances
        places them in two dimensions, from the first two principal
        components of the profiles, on one thread, so that it depends only on
        ``seed``.

        Args:
            profiles: A ``perturbation_profiles`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.profiles`.
            perturbations: The names of the perturbations to cluster, in any
                order, such as those with a significant E-distance and a low
                remaining fraction, or None for every profile.
            metric: ``"correlation"`` or ``"euclidean"``.
            min_cluster_size: The fewest perturbations of a cluster, at least
                2.
            min_samples: The perturbations within a perturbation's core
                distance, itself included, at least 1.
            selection: ``"eom"`` to select clusters by excess of mass, or
                ``"leaf"`` to take the leaves of HDBSCAN's cluster tree.
            seed: The seed of the layout, from 0 to 2**32 - 1.
            invalidate_cache: Cluster again instead of reusing saved clusters.

        Returns:
            A complete ``perturbation_clusters`` artifact scoped to the assay
            of the profiles, with one row per clustered perturbation. Read it
            with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_clusters`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``profiles`` is not a ``perturbation_profiles``
                artifact, ``perturbations`` names a perturbation without a
                profile, fewer than 3 perturbations or 2 features remain,
                ``min_samples`` exceeds the perturbations, or a setting is
                unknown or outside its range.
            PermissionError: If the store is read-only and the clusters are
                not saved yet.
            MemoryError: If the profiles and the distances between them,
                with their working copies, do not fit the memory budget.
        """
        from threadpoolctl import threadpool_limits

        from ...perturbation.clusters import (
            cluster_memory_bytes,
            cluster_profiles,
        )

        if not isinstance(profiles, ArtifactRef):
            raise TypeError("profiles must be an ArtifactRef")
        if profiles.kind != PerturbationProfileArguments.artifact_kind:
            raise ValueError(
                "profiles must be a perturbation_profiles artifact, not "
                f"{profiles.kind!r}"
            )
        cluster_size, core_size, resolved_seed = cluster_settings(
            metric, min_cluster_size, min_samples, selection, seed
        )
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        require_complete_artifact(self.zw, profiles)
        group = artifact_group(self.zw, profiles)
        names = np.asarray(
            as_zarr_array(group["perturbations"], name="perturbations")[:]
        ).astype(str)
        if perturbations is None:
            chosen = np.arange(len(names))
        else:
            requested = perturbation_names(perturbations)
            positions = pd.Index(names).get_indexer(requested)
            unknown = sorted(
                {
                    name
                    for name, position in zip(requested, positions, strict=True)
                    if position < 0
                }
            )
            if unknown:
                raise ValueError(
                    "perturbations names perturbations without a profile: "
                    + ", ".join(repr(name) for name in unknown[:5])
                    + "; the profiles hold the perturbations with at least "
                    "min_cells cells"
                )
            chosen = np.unique(positions)
        stored = as_zarr_array(group["values"], name="values")
        n_chosen = len(chosen)
        n_features = int(stored.shape[1])
        if n_chosen < 3 or n_features < 2:
            raise ValueError(
                "Clustering needs at least 3 perturbations and 2 features, not "
                f"{n_chosen} and {n_features}"
            )
        if core_size > n_chosen:
            raise ValueError(
                f"min_samples={core_size} must be at most the {n_chosen} "
                "perturbations clustered"
            )
        selected = names[chosen]
        arguments = PerturbationClusterArguments(
            perturbation_profiles=profiles,
            # Every profile, named or not, is the same input.
            perturbations_fingerprint=(
                None if n_chosen == len(names) else fingerprint_strings(selected)
            ),
            metric=metric,
            min_cluster_size=cluster_size,
            min_samples=core_size,
            selection=selection,
            seed=resolved_seed,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=str(profiles.assay),
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("perturbations", shape=(n_chosen,)),
                *(
                    ArrayRequirement(name, shape=(n_chosen,), dtype_kind="i")
                    for name in ("cluster", "order")
                ),
                ArrayRequirement("probability", shape=(n_chosen,), dtype_kind="f"),
                ArrayRequirement("embedding", shape=(n_chosen, 2), dtype_kind="f"),
            ),
        )
        if planned.reused:
            return planned.ref
        # The float32 profiles as read, and the clustering's peak.
        needed = 4 * n_chosen * n_features + cluster_memory_bytes(n_chosen, n_features)
        if needed > self.memoryBytes:
            raise MemoryError(
                f"Clustering {n_chosen} perturbations in {n_features} features "
                f"holds about {needed} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Pass fewer perturbations, or raise "
                "mem_budget."
            )
        self._require_writable(arguments.operation)
        values = np.asarray(stored.oindex[chosen, :])
        # The distances' matrix products use as many BLAS threads as the store.
        with threadpool_limits(limits=self.nthreads, user_api="blas"):
            arrays = cluster_profiles(
                values,
                metric=metric,
                min_cluster_size=cluster_size,
                min_samples=core_size,
                selection=selection,
                seed=resolved_seed,
            )
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "perturbations": selected,
                "cluster": arrays.cluster,
                "probability": arrays.probability,
                "order": arrays.order,
                "embedding": arrays.embedding,
            },
        )
        logger.info(
            f"Clustered {n_chosen} perturbations into "
            f"{int(arrays.cluster.max(initial=-1)) + 1} clusters, with "
            f"{int(np.count_nonzero(arrays.cluster < 0))} unassigned"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_clusters(
        self, clusters: ArtifactRef
    ) -> PerturbationClusters:
        """Read saved perturbation clusters with the distances between their profiles.

        The distances are computed again from the saved profiles, as the
        clustering computed them.

        Args:
            clusters: A ``perturbation_clusters`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.cluster`.

        Returns:
            Each perturbation's cluster, membership strength, cells and place
            in the layout, the leaf order and the distances.

        Raises:
            TypeError: If ``clusters`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_clusters`` artifact.
            MemoryError: If the distances between the profiles do not fit the
                memory budget.
        """
        from threadpoolctl import threadpool_limits

        from ...perturbation.clusters import profile_distances

        if not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef")
        if clusters.kind != PerturbationClusterArguments.artifact_kind:
            raise ValueError(
                "clusters must be a perturbation_clusters artifact, not "
                f"{clusters.kind!r}"
            )
        status = require_complete_artifact(self.zw, clusters)
        group = artifact_group(self.zw, clusters)
        source_ref = status.input_ref("perturbation_profiles")
        require_complete_artifact(self.zw, source_ref)
        source = artifact_group(self.zw, source_ref)

        def read(stored: zarr.Group, name: str) -> np.ndarray:
            return np.asarray(as_zarr_array(stored[name], name=name)[:])

        names = read(group, "perturbations").astype(str)
        positions = pd.Index(read(source, "perturbations").astype(str)).get_indexer(
            names
        )
        values = as_zarr_array(source["values"], name="values")
        n_clustered = len(names)
        n_features = int(values.shape[1])
        # The distances and their symmetric copy, and the profiles as read,
        # as float64 and centred.
        n_bytes = 16 * n_clustered * n_clustered + 20 * n_clustered * n_features
        if n_bytes > self.memoryBytes:
            raise MemoryError(
                f"The distances between {n_clustered} profiles hold about "
                f"{n_bytes} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        parameters = status.parameters or {}
        metric = str(parameters["metric"])
        with threadpool_limits(limits=self.nthreads, user_api="blas"):
            distances = profile_distances(
                np.asarray(values.oindex[positions, :]), metric
            )
        index = pd.Index(names, name="perturbation")
        embedding = read(group, "embedding")
        return PerturbationClusters(
            table=pd.DataFrame(
                {
                    "cluster": read(group, "cluster"),
                    "probability": read(group, "probability"),
                    "n_cells": read(source, "n_cells")[positions],
                    "x": embedding[:, 0],
                    "y": embedding[:, 1],
                },
                index=index,
            ),
            order=names[read(group, "order")],
            distances=pd.DataFrame(distances, index=index, columns=index.copy()),
            metric=metric,
            min_cluster_size=int(parameters["min_cluster_size"]),
            min_samples=int(parameters["min_samples"]),
            selection=str(parameters["selection"]),
            seed=int(parameters["seed"]),
            from_assay=str(clusters.assay),
        )
