from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from ...metadata.artifacts import plan_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    array_row_selection_parts,
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
from ...storage.geometry import array_geometry
from ...storage.partition import scan_band
from ...storage.selections import (
    iter_stored_selection_blocks,
    validate_cell_selection,
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
        guides: Each row's guide, as a position in the guide assay, for rows
            by guide; empty for rows by target.
    """

    status: ArtifactStatus
    target_names: np.ndarray
    guide_targets: np.ndarray
    codes: np.ndarray
    names: np.ndarray
    guides: np.ndarray


@dataclass(frozen=True, slots=True)
class _ComparedCells:
    """The cells that a per-perturbation table compares.

    They are the cells that carry a row alone and the control cells.

    Attributes:
        cells: Their positions in the store, ascending.
        groups: The row that each carries alone, -1 for a control cell.
        control: Whether each is a control cell.
    """

    cells: np.ndarray
    groups: np.ndarray
    control: np.ndarray


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


def _stream_bytes(counts: Any, scratch_bytes_per_row: int = 0) -> tuple[int, int]:
    """Return what a budgeted row stream of ``counts`` holds: its least, and with full blocks.

    ``ChunkedArray._stream_blocks`` with a memory budget holds the matrix's
    own arrays, the decode of one stored chunk, and the rows of each block
    with ``scratch_bytes_per_row`` for each, beside the caller's resident
    bytes. It shrinks its blocks to fit, down to one row, and refuses a
    budget without room for that row; this follows its arithmetic.

    Args:
        counts: A chunked matrix of counts, such as ``assay.rawData[cells, :]``.
        scratch_bytes_per_row: The bytes that the caller makes of each row of
            a block.

    Returns:
        The bytes of a stream of one-row blocks, and of a stream of full
        blocks, each beside the caller's resident bytes.
    """
    rows = min(counts._block_size, max(1, counts.shape[0]))
    per_row = max(1, counts._block_owned_bytes() // rows + scratch_bytes_per_row)
    fixed = counts._resident_bytes() + counts._max_decode_bytes()
    return fixed + per_row, fixed + rows * per_row


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

    The blocks hold the rows of a stored chunk beside ``resident_bytes``
    and the ``scratch_bytes_per_row`` that the caller makes of each row,
    which :func:`_count_stream_bytes` counts, and shrink to fit the memory
    budget when it leaves less room. Returns the most rows of one block as
    stored and the blocks, in the order of ``cells``.
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
        memory_bytes=store.memoryBytes,
    )


def _count_matrix(assay: Any, features: np.ndarray, cells: np.ndarray) -> Any:
    """Return the lazy raw counts of features over cells that :func:`_count_blocks` streams."""
    from ...assay.normalization import feature_values
    from ...metadata.selection import NormalizationSpec

    return feature_values(
        assay, np.asarray(features), np.asarray(cells), NormalizationSpec(source="raw")
    )


def _count_stream_bytes(
    assay: Any, features: np.ndarray, cells: np.ndarray, *, scratch_bytes_per_row: int
) -> int:
    """Return the bytes that :func:`_count_blocks` holds beside its caller's.

    That is the stream's index arrays and one block of the counts of
    ``features`` over the ``cells`` of a stored chunk, with its decode and
    the ``scratch_bytes_per_row`` that the caller makes of each row.
    """
    held: int = _count_matrix(assay, features, cells).stream_bytes(
        scratch_bytes_per_row
    )
    return held


def _band_bytes(
    assay: Any,
    features: np.ndarray,
    cell_sets: Sequence[np.ndarray],
    *,
    held_bytes: int,
    per_feature: int,
    scratch_per_feature: int,
) -> int:
    """Return the bytes that reading one band of ``features`` holds.

    The band holds ``held_bytes``, ``per_feature`` bytes for each of its
    features, and the stream of its counts over each of ``cell_sets`` in
    turn, with ``scratch_per_feature`` bytes for each feature of a row, as
    :func:`_count_stream_bytes` counts it. A band of one feature holds this
    for ``features[:1]``, and as much for any other feature.
    """
    width = len(features)
    return (
        held_bytes
        + per_feature * width
        + max(
            _count_stream_bytes(
                assay,
                features,
                cells,
                scratch_bytes_per_row=scratch_per_feature * width,
            )
            for cells in cell_sets
        )
    )


def _count_bands(
    assay: Any,
    features: np.ndarray,
    cell_sets: Sequence[np.ndarray],
    *,
    budget: int,
    held_bytes: int,
    per_feature: int,
    scratch_per_feature: int,
    most: int,
) -> list[tuple[int, int]]:
    """Split ``features`` into consecutive bands whose reads fit ``budget``.

    A band holds what :func:`_band_bytes` counts for it. Each band takes as
    many features as fit, at most ``most``; a band of one feature is taken
    whatever it holds, since the caller refuses a budget that it does not
    fit. What a stream decodes depends on the stored chunks of the band's
    features, so each band is sized on its own.

    Returns:
        Each band's first position in ``features`` and the position past
        its last feature, in order.
    """
    matrices = [_count_matrix(assay, features, cells) for cells in cell_sets]

    def fits(start: int, stop: int) -> bool:
        width = stop - start
        streams: int = max(
            matrix[:, start:stop].stream_bytes(scratch_per_feature * width)
            for matrix in matrices
        )
        return held_bytes + per_feature * width + streams <= budget

    bands: list[tuple[int, int]] = []
    start = 0
    while start < len(features):
        widest = max(1, min(most, len(features) - start))
        low, high = (widest, widest) if fits(start, start + widest) else (1, widest - 1)
        # The widest band from ``start`` that fits, by bisection.
        while low < high:
            middle = (low + high + 1) // 2
            if fits(start, start + middle):
                low = middle
            else:
                high = middle - 1
        bands.append((start, start + low))
        start += low
    return bands


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
        guides = np.empty(0, dtype=np.int64)
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
        guides=guides,
    )


_NO_CONTROL_CELLS = (
    "No labelled cell carries control guides only, and each perturbation is "
    "compared with such cells; check the controls passed to perturbation.labels, "
    "or in a screen whose cells carry several guides, test associations with "
    "moi='high'"
)
# The cells of labels that are read at once to find the compared cells, at
# least one chunk of codes, and the cells of a selection that are read at once
# to place them.
_LABEL_BLOCK_ROWS = 1 << 16
_SELECTION_BLOCK_ROWS = 1 << 14
# The bytes that finding the compared cells holds for each cell of a block of
# labels, beside the chunks that its reads decode: the cells' flags, counts,
# first codes and rows, and the positions, rows and flags of those compared.
_LABEL_BLOCK_BYTES_PER_CELL = 64
# The bytes that the compared cells hold for each of them while they are
# found: their positions among the labels, rows and flags, gathered from the
# blocks and joined, and their positions in the store.
_COMPARED_BYTES_PER_CELL = 48


@dataclass(frozen=True, slots=True)
class _CellRowArrays:
    """The stored arrays that give each labelled cell the row it carries alone.

    A cell carries a target alone when its called guides target that gene and
    no other, and a guide alone when it is the cell's one called guide: by
    target, the labels' ``n_targets`` and ``target_codes``; by guide, the
    calls' ``n_guides`` and ``guide_codes``. Only each cell's first code is
    read.

    Attributes:
        is_control: Whether each cell carries control guides only.
        counts: Each cell's number of targets, or of called guides.
        codes: Each cell's targets, or called guides, as positions, padded
            with -1.
        row_of: The row of each target, or guide, -1 for none.
    """

    is_control: zarr.Array
    counts: zarr.Array
    codes: zarr.Array
    row_of: np.ndarray

    def rows(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the row that each cell from ``start`` to ``stop`` carries alone, or -1, and its control flag."""
        control = np.asarray(self.is_control[start:stop], dtype=bool)
        single = np.asarray(self.counts[start:stop]) == 1
        first = np.asarray(self.codes[start:stop, 0])
        return np.where(single, self.row_of[np.maximum(first, 0)], -1), control

    def block_bytes(self, block_rows: int) -> int:
        """Return the bytes that finding the compared cells of one block holds."""
        decoded = max(
            array_row_selection_parts(array)[0]
            for array in (self.is_control, self.counts)
        )
        return (
            block_rows * _LABEL_BLOCK_BYTES_PER_CELL
            + array_row_selection_parts(self.codes)[0]
            + decoded
        )


def _cell_row_arrays(
    store: Any, labels: ArtifactRef, rows: _LabelRows, by: str
) -> _CellRowArrays:
    """Return the stored arrays from which each labelled cell's row is read."""
    group = artifact_group(store.zw, labels)
    if by == "target":
        row_of = np.full(len(rows.target_names), -1, dtype=np.int64)
        row_of[rows.codes] = np.arange(len(rows.codes))
        source, counts, codes = group, "n_targets", "target_codes"
    else:
        row_of = np.full(len(rows.guide_targets), -1, dtype=np.int64)
        row_of[rows.guides] = np.arange(len(rows.guides))
        source = artifact_group(store.zw, rows.status.input_ref("guide_calls"))
        counts, codes = "n_guides", "guide_codes"
    return _CellRowArrays(
        is_control=as_zarr_array(group["is_control"], name="is_control"),
        counts=as_zarr_array(source[counts], name=counts),
        codes=as_zarr_array(source[codes], name=codes),
        row_of=row_of,
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
    arrays = _cell_row_arrays(store, labels, rows, by)
    control = np.asarray(arrays.is_control[:], dtype=bool)
    if not control.any():
        raise ValueError(_NO_CONTROL_CELLS)
    groups, _ = arrays.rows(0, len(control))
    return groups, control


def _compared_cells(
    store: Any, labels: ArtifactRef, rows: _LabelRows, by: str
) -> tuple[_ComparedCells, int]:
    """Return the cells that a per-perturbation table compares, and the bytes that finding them holds.

    The labels are read a block of cells at a time, so that only the arrays of
    the compared cells grow with the cells: a screen with many labelled cells
    and few compared ones holds little. The bytes are the most that finding
    the compared cells holds beside what the caller holds.

    Raises:
        ValueError: If no labelled cell carries control guides only.
    """
    arrays = _cell_row_arrays(store, labels, rows, by)
    n_labelled = int(arrays.is_control.shape[0])
    # The labels' cells are checked against the cell IDs before any array
    # of the compared cells is held.
    selection = rows.status.input_ref("cell_selection")
    validate_cell_selection(store.zw, selection)
    # Whole chunks of the codes, whose reads decode the most, up to a bound.
    chunk_rows = max(1, int(arrays.codes.chunks[0]))
    block_rows = max(chunk_rows, _LABEL_BLOCK_ROWS // chunk_rows * chunk_rows)
    positions, groups, control = [], [], []
    for start in range(0, n_labelled, block_rows):
        row, flag = arrays.rows(start, min(start + block_rows, n_labelled))
        kept = np.flatnonzero((row >= 0) | flag)
        positions.append(kept + start)
        groups.append(row[kept])
        control.append(flag[kept])
    if not any(flags.any() for flags in control):
        raise ValueError(_NO_CONTROL_CELLS)
    compared = np.concatenate(positions)
    del positions
    cells = _store_positions(store, selection, compared)
    del compared
    found = _ComparedCells(
        cells=cells, groups=np.concatenate(groups), control=np.concatenate(control)
    )
    held = max(
        _cell_selection_check_bytes(store),
        len(cells) * _COMPARED_BYTES_PER_CELL
        + max(
            arrays.block_bytes(min(block_rows, n_labelled)),
            _selection_block_bytes(store, selection),
        ),
    )
    return found, held


def _selection_blocks(store: Any, selection: ArtifactRef) -> Iterable[Any]:
    """Yield a validated cell selection in blocks of cells."""
    return iter_stored_selection_blocks(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
        block_rows=_SELECTION_BLOCK_ROWS,
    )


def _selection_block_bytes(store: Any, selection: ArtifactRef) -> int:
    """Return the bytes that one block of a cell selection holds as it is read.

    A block holds its flags and the store positions of its cells, beside
    those of the block before it, and the positions of the cells looked up in
    it, at most 64 bytes for each of its cells, and the two chunks of flags
    that its read may decode.
    """
    flags = as_zarr_array(artifact_group(store.zw, selection)["values"], name="values")
    block_rows = min(_SELECTION_BLOCK_ROWS, int(flags.shape[0]))
    return block_rows * 64 + 2 * array_row_selection_parts(flags)[0]


def _store_positions(
    store: Any, selection: ArtifactRef, compact: np.ndarray
) -> np.ndarray:
    """Return the store positions of a selection's cells at ascending ``compact`` positions.

    The selection is read a block of cells at a time.
    """
    positions = np.empty(len(compact), dtype=np.int64)
    for block in _selection_blocks(store, selection):
        low, high = np.searchsorted(compact, (block.compact_start, block.compact_stop))
        positions[low:high] = block.selected_indices[
            compact[low:high] - block.compact_start
        ]
    return positions


def _selected_rows(
    store: Any, selection: ArtifactRef, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return where each of ascending store positions sits among a selection's cells.

    Also returns whether each is a cell of the selection; the position of one
    that is not means nothing. The selection is read a block of cells at a
    time.
    """
    found = np.zeros(len(cells), dtype=np.int64)
    present = np.zeros(len(cells), dtype=bool)
    for block in _selection_blocks(store, selection):
        low, high = np.searchsorted(cells, (block.start, block.stop))
        within, here = _positions_in(block.selected_indices, cells[low:high])
        found[low:high] = block.compact_start + within
        present[low:high] = here
    return found, present


def _cell_selection_check_bytes(store: Any, *, selected_ids: bool = False) -> int:
    """Return the bytes that checking a cell selection against the cell IDs holds.

    The check reads the selection's flags a band at a time. With
    ``selected_ids``, as for imported coordinates, it also fingerprints the
    selected cells' IDs a band at a time, which decodes whole chunks of IDs
    and holds a band of IDs as read and as selected. The fingerprint of every
    cell ID, which the first check in a process computes a chunk at a time,
    is kept while the IDs are unchanged, so it is not counted.
    """
    ids = as_zarr_array(store.zw["cellData"]["ids"], name="ids")
    n_cells = int(ids.shape[0])
    held = 2 * n_cells
    if selected_ids:
        band = min(n_cells, scan_band(array_geometry(ids), fallback=n_cells))
        itemsize = int(np.dtype(ids.dtype).itemsize)
        held += array_row_selection_parts(ids)[0] + 2 * band * itemsize
    return held


def _column_read_bytes(metadata: Any, column: str | None) -> int:
    """Return the bytes that reading a whole metadata column holds, 0 without one.

    Zarr decodes whole chunks, also where a chunk reaches past the column's
    end, so that reading a column shorter than its chunk holds the chunk as
    stored and decoded, beside the column's values as read and as copied.
    """
    if column is None:
        return 0
    array = metadata._get_array(column)
    return array_row_selection_parts(array)[0] + 2 * int(array.nbytes)


def _read_chunkwise(array: zarr.Array) -> np.ndarray:
    """Read a whole array one chunk of rows at a time.

    Each read decodes one chunk of rows, so that reading holds the array and
    one chunk beside it.
    """
    values = np.empty(array.shape, dtype=array.dtype)
    step = max(1, int(array.chunks[0]))
    for start in range(0, int(array.shape[0]), step):
        values[start : start + step] = array[start : start + step]
    return values
