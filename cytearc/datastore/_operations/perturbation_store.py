from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.artifacts import plan_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import resolve_cell_aligned_artifact, sample_codes
from ...graph.arguments import OperationArguments
from ...perturbation.efficiency import target_features
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactStatus,
    ArtifactRef,
    artifact_group,
    fingerprint_strings,
    require_complete_artifact,
)
from ...storage.types import as_zarr_array, read_array

# The most bytes of values that one band of features holds, in Mixscape and
# the association tests, whatever the budget, so that a large budget does not
# fill memory.
_BAND_BYTES = 1 << 29
# The coordinate artifacts that E-distances and signatures read, in the order
# that their refusal names them: the kinds of NATIVE_COORDINATE_KINDS.
_COORDINATE_KINDS = (
    "reduction",
    "batch_correction",
    "imported_coordinates",
    "normalized",
)


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
class _CellSamples:
    """The samples of some cells, by their label in one cell metadata column.

    Attributes:
        columns: The column, or no column for one sample of every cell.
        codes: Each cell's sample, numbered in the sorted order of the labels.
        names: Each sample's label, None for the one sample without a column.
        assignments: The fingerprint of the cells' samples, None without a
            column.
    """

    columns: tuple[str, ...]
    codes: np.ndarray
    names: list[str | None]
    assignments: str | None


def _require_budget(
    needed: int,
    budget: int,
    holding: str,
    *,
    beside: str = "",
    remedy: str = "Raise mem_budget.",
) -> None:
    """Refuse a memory budget below the ``needed`` bytes of an operation.

    The message reads ``holding``, which names what holds the bytes and ends
    with its verb, such as ``"Labelling 9 cells holds about"``, then the
    bytes, ``beside``, which names what else they cover, the budget, and
    ``remedy``.
    """
    if needed > budget:
        raise MemoryError(
            f"{holding} {needed} bytes{beside}, more than the memory budget of "
            f"{budget} bytes. {remedy}"
        )


def _require_ref(ref: object, name: str, *kinds: str, optional: bool = False) -> None:
    """Refuse an argument ``name`` that is not a reference to an artifact of ``kinds``.

    Without ``kinds`` any kind is taken, and with ``optional`` None too.

    Raises:
        TypeError: If ``ref`` is not an ArtifactRef.
        ValueError: If its kind is none of ``kinds``.
    """
    if optional and ref is None:
        return
    if not isinstance(ref, ArtifactRef):
        raise TypeError(
            f"{name} must be an ArtifactRef" + (" or None" if optional else "")
        )
    if kinds and ref.kind not in kinds:
        named = ", ".join(kinds[:-1]) + " or " + kinds[-1] if kinds[1:] else kinds[0]
        raise ValueError(f"{name} must be a {named} artifact, not {ref.kind!r}")


def _positions_in(
    sorted_cells: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return where each of ``cells`` sits in ``sorted_cells`` and whether it is there.

    ``sorted_cells`` holds store positions in ascending order. The position of
    a cell that is not there means nothing.
    """
    positions = np.searchsorted(sorted_cells, cells)
    present = positions < len(sorted_cells)
    present[present] = sorted_cells[positions[present]] == cells[present]
    return positions, present


def _matched_ids(features: np.ndarray, feature_ids: np.ndarray) -> np.ndarray:
    """Return the ID of each matched feature position, empty text for -1."""
    ids: np.ndarray = np.where(features >= 0, feature_ids[np.maximum(features, 0)], "")
    return ids


def _pairs_fingerprint(names: np.ndarray, feature_ids: np.ndarray) -> str:
    """Fingerprint pairs of a row and a feature by their ``name\\tfeature`` texts.

    An empty feature ID stands for a row without a feature.
    """
    return fingerprint_strings(
        np.char.add(np.char.add(np.asarray(names).astype(str), "\t"), feature_ids)
    )


def _stored_samples(group: zarr.Group, name: str = "samples") -> list[str | None]:
    """Return the sample labels saved in an attribute, None for one sample."""
    labels = group.attrs.get(name)
    if not isinstance(labels, list):
        raise ValueError(
            f"The {group.attrs.get('kind')} artifact has no list of samples"
        )
    return [None if label is None else str(label) for label in labels]


def _require_status_names(group: zarr.Group, statuses: Sequence[str]) -> None:
    """Refuse an artifact whose stored status names are not ``statuses``.

    The artifact's ``status_names`` attribute must equal ``statuses``, the
    status names of the method's module, so that every reader, a loader or a
    stage summary, decodes a code as the same status.

    Raises:
        ValueError: If the artifact names its statuses otherwise.
    """
    stored = group.attrs.get("status_names")
    if not isinstance(stored, list) or [str(name) for name in stored] != list(statuses):
        raise ValueError(
            f"The {group.attrs.get('kind')} artifact names its statuses "
            f"{stored!r}, not {list(statuses)!r}, so its status codes cannot be "
            "read; compute it again"
        )


def _status_labels(
    group: zarr.Group, statuses: Sequence[str], codes: np.ndarray | None = None
) -> np.ndarray:
    """Return the status of each stored code, after checking the stored names.

    ``codes`` defaults to the artifact's ``status`` array; the stored names
    must equal ``statuses``, as :func:`_require_status_names` checks.

    Raises:
        ValueError: If the artifact names its statuses otherwise.
    """
    _require_status_names(group, statuses)
    if codes is None:
        codes = read_array(group, "status")
    labels: np.ndarray = np.asarray(statuses, dtype=object)[codes]
    return labels


def _count_blocks(
    store: Any,
    assay: Any,
    features: np.ndarray,
    cells: np.ndarray,
    *,
    resident_bytes: int,
    scratch_bytes_per_row: int,
) -> tuple[int, Iterable[np.ndarray]]:
    """Stream the raw counts of features over cells, cells x features.

    The blocks fit the memory budget beside ``resident_bytes`` and the
    ``scratch_bytes_per_row`` that the caller makes of each block. Returns
    the most rows of one block and the blocks, in the order of ``cells``.
    """
    from ...assay.normalization import feature_value_blocks
    from ...metadata.selection import NormalizationSpec

    return feature_value_blocks(
        assay,
        features,
        cells,
        NormalizationSpec(source="raw"),
        nthreads=store.nthreads,
        resident_bytes=resident_bytes,
        scratch_bytes_per_row=scratch_bytes_per_row,
    )


def _plan_cells(
    store: Any,
    arguments: OperationArguments,
    *,
    assay: str,
    cell_selection: ArtifactRef,
    arrays: Mapping[str, tuple[tuple[int | None, ...], str | None]],
    invalidate_cache: bool,
    reuse_validator: Callable[[ArtifactRef, zarr.Group], bool],
    required_attributes: tuple[str, ...] = (),
) -> PlannedArtifact:
    """Plan an assay-scoped artifact of arrays with one row per selected cell."""
    record = arguments.to_record()
    return plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay=assay,
        kind=arguments.artifact_kind,
        operation=arguments.operation,
        parameters=record.parameters,
        inputs=record.inputs,
        execution_options=record.execution_options,
        cell_selection=cell_selection,
        arrays=arrays,
        invalidate_cache=invalidate_cache,
        required_attributes=required_attributes,
        reuse_validator=reuse_validator,
    )


def _cell_values(store: Any, ref: ArtifactRef, value: str | None = None) -> np.ndarray:
    """Read one per-cell array of a cell-aligned artifact, without cell IDs.

    ``value`` names the array, None for the canonical one. The read is
    checked against the memory budget.
    """
    resolved = resolve_cell_aligned_artifact(
        store.zw, ref, value_name=value, ndim=None, max_bytes=store.memoryBytes
    )
    values: np.ndarray = np.asarray(resolved.values)
    return values


def _require_cell_column(store: Any, column: str | None, argument: str) -> None:
    """Refuse an ``argument`` that names a cell metadata column the store lacks.

    Raises:
        ValueError: If ``column`` is not None and not a cell column.
    """
    if column is not None and column not in store.cells.columns:
        raise ValueError(f"{argument} '{column}' not found in cell metadata")


def _cell_samples(
    store: Any, column: str | None, cell_index: np.ndarray, *, reason: str
) -> _CellSamples:
    """Group cells into samples by their label in a cell metadata column.

    Without a column every cell is in one sample. ``reason`` says why every
    cell needs a label; it ends the error for a cell without one.
    """
    columns = () if column is None else (column,)
    labels = [
        apply_missing_mask(
            np.asarray(read_metadata_rows_chunkwise(store.cells, name, cell_index)),
            read_metadata_missing_rows_chunkwise(store.cells, name, cell_index),
            labels=True,
        )
        for name in columns
    ]
    codes, sample_labels, assignments = sample_codes(
        columns, labels, len(cell_index), reason=reason
    )
    return _CellSamples(
        columns=columns,
        codes=codes,
        names=[None if column is None else by[column] for by in sample_labels],
        assignments=assignments,
    )


def _row_features(
    store: Any,
    labels: ArtifactRef,
    rows: _LabelRows,
    id_column: str | None,
    feature_ids: np.ndarray,
    feature_names: np.ndarray,
) -> np.ndarray:
    """Match each row's target to a feature of the expression assay.

    A target takes the feature whose ID its guides give in ``id_column``
    of the guide assay of ``labels``, or else the one feature with its
    name, as :func:`~cytearc.perturbation.efficiency.target_features`
    describes. Returns each row's feature position, -1 without a match.

    Raises:
        ValueError: If ``id_column`` is not a feature column of the guide
            assay.
    """
    guide_feature_ids = store._guide_feature_column(
        store._get_assay(str(labels.assay)), id_column, "id_column"
    )
    matched: np.ndarray = target_features(
        rows.target_names,
        rows.guide_targets,
        guide_feature_ids,
        feature_ids,
        feature_names,
    )[rows.codes]
    return matched


def _label_rows(
    store: Any, labels: ArtifactRef, by: str, *, control_guides: bool = False
) -> _LabelRows:
    """Read the rows of a per-perturbation table: each target or each guide.

    A row is a target that is not a control, or with ``by="guide"`` a guide
    whose target is not a control, in assay order; ``control_guides``
    makes the guides of control targets rows too.
    """
    status = require_complete_artifact(store.zw, labels)
    group = artifact_group(store.zw, labels)
    calls = artifact_group(store.zw, status.input_ref("guide_calls"))

    target_names = read_array(group, "targets").astype(str)
    is_control = read_array(group, "target_is_control").astype(bool)
    guide_targets = read_array(group, "guide_target_codes").astype(np.int64)
    if by == "target":
        codes = np.flatnonzero(~is_control)
        names = target_names[codes]
    else:
        guides = np.flatnonzero(
            (guide_targets >= 0)
            & (control_guides | ~is_control[np.maximum(guide_targets, 0)])
        )
        codes = guide_targets[guides]
        names = read_array(calls, "guides").astype(str)[guides]
    return _LabelRows(
        status=status,
        target_names=target_names,
        guide_targets=guide_targets,
        codes=codes,
        names=names,
    )


def _label_groups(
    store: Any, labels: ArtifactRef, rows: _LabelRows, by: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the row that each labelled cell carries alone, or -1, and its control flag.

    A cell carries a target alone when its called guides target that gene
    and no other, and a guide alone when it is the cell's one called guide.

    Raises:
        ValueError: If no labelled cell carries control guides only.
    """
    group = artifact_group(store.zw, labels)

    control = read_array(group, "is_control").astype(bool)
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
        single = read_array(group, "n_targets") == 1
        # Only each cell's first target is read, not the other columns.
        first = np.asarray(
            as_zarr_array(group["target_codes"], name="target_codes")[:, 0]
        )
        return np.where(single, row_of[np.maximum(first, 0)], -1), control
    guide_labels = _cell_values(store, labels, "guide_label").astype(str)
    return pd.Index(rows.names).get_indexer(guide_labels).astype(np.int64), control
