from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from ...metadata.artifacts import plan_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    array_row_selection_parts,
    metadata_missing_mask,
    read_rows,
    read_text_rows,
    read_whole,
)
from ...metadata.selection import resolve_cell_aligned_artifact, sample_codes
from ...graph.arguments import OperationArguments
from ...perturbation.efficiency import target_feature_bytes, target_features
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactStatus,
    ArtifactRef,
    artifact_group,
    fingerprint_strings,
    require_complete_artifact,
)
from ...storage.async_execution import zarr_decoding_threads
from ...storage.geometry import array_geometry
from ...storage.selections import (
    iter_stored_selection_blocks,
    validate_cell_selection,
)
from ...storage.types import as_zarr_array, read_array
from ...utils.memory_budget import (
    MemoryBudgetError,
    MemoryLedger,
    memory_refusal,
    object_array_bytes,
)

# The most bytes of values that one band of features holds, in Mixscape and
# the association tests, whatever the budget, so that a large budget does not
# fill memory.
_BAND_BYTES = 1 << 29
# The bytes of a label that is not text as a Python object, such as a
# number, beside its reference.
_NUMBER_BYTES = 32
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


#: The Python objects that a screen call holds besides its arrays: its plans
#: and the store's metadata that it opens, as traced for guide calling.
_CALL_FIXED_BYTES = 1 << 18


def _call_ledger(store: Any, doing: str, *, remedy: str | None = None) -> MemoryLedger:
    """Return the memory ledger of one screen call, holding the call's own objects.

    The ledger refuses the store's budget as ``doing``, such as ``"Measuring
    the efficiency of 3 targets"``, naming ``remedy`` beside a larger budget,
    and its reads decode on the store's ``nthreads`` codec threads.
    """
    ledger = MemoryLedger(
        store.memoryBytes, doing, remedy=remedy, codec_threads=store.nthreads
    )
    ledger.hold("the call's plans and the store's metadata it opens", _CALL_FIXED_BYTES)
    return ledger


def _converted(
    ledger: MemoryLedger, values: np.ndarray, dtype: Any, item: str
) -> np.ndarray:
    """Return ``values``, which ``ledger`` holds, as ``dtype``, holding the copy in place of them.

    Values of that dtype already are returned as they are.
    """
    if values.dtype == np.dtype(dtype):
        return values
    ledger.hold(item, values.size * np.dtype(dtype).itemsize)
    converted = values.astype(dtype)
    ledger.release(values.nbytes)
    return converted


def _text_width(values: np.ndarray) -> int:
    """Return the characters of the widest of text ``values``, at least 1."""
    if values.dtype.kind == "U":
        return max(1, values.dtype.itemsize // 4)
    return max((len(str(value)) for value in values.ravel()), default=1) or 1


def _table_write_bytes(columns: Mapping[str, np.ndarray]) -> int:
    """Return what writing ``columns`` with ``write_cell_data_artifact`` holds beside them.

    Each column is counted as ``cell_data_write_bytes`` counts it, with its
    file buffers, a text column at the width of its widest value.
    """
    from ...metadata.artifacts import cell_data_write_bytes

    specs: dict[str, tuple[tuple[int, ...], Any]] = {}
    for name, values in columns.items():
        array = np.asarray(values)
        dtype = f"U{_text_width(array)}" if array.dtype.kind in "OU" else array.dtype
        specs[name] = (tuple(int(size) for size in array.shape), dtype)
    return cell_data_write_bytes(specs)


#: A table's bytes for each of its values, beside the values that its
#: columns copy: the references of a text value and its index.
_FRAME_BYTES_PER_VALUE = 16


def _frame_bytes(columns: Mapping[str, np.ndarray]) -> int:
    """Return what building a table of ``columns`` holds beside them.

    A numeric column is copied into the table, and a text column becomes
    Python strings, each at the width of the column's widest value, as
    ``python_text_bytes`` counts them, beside the references of every value.
    """
    from ...utils.memory_budget import python_text_bytes

    held = 0
    for values in columns.values():
        array = np.asarray(values)
        n_values = int(array.size)
        held += _FRAME_BYTES_PER_VALUE * n_values
        if array.dtype.kind in "OU":
            # The widest code point of stored text, or any for objects.
            widest = (
                int(array.view(np.uint32).max())
                if array.dtype.kind == "U" and array.dtype.itemsize and n_values
                else 0x10FFFF
            )
            held += python_text_bytes(
                n_values, n_values * _text_width(array), widest=widest
            )
        else:
            held += array.nbytes
    return held


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


#: What finding where each of some cells sits among sorted cells holds for
#: each of them beside the position and flag that :func:`_positions_in`
#: returns: the masks and gathered positions that compare them, 17 bytes a
#: cell of 26 in all as traced at a million cells.
_POSITIONS_SCRATCH_BYTES = 24


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
    """Return what a budgeted row stream of ``counts`` holds: its least, and with blocks of a stored chunk's rows.

    ``ChunkedArray._stream_blocks`` with a memory budget holds the matrix's
    own arrays, the decode of one stored chunk beside the chunk that the
    previous read can still hold, and the rows of each block with
    ``scratch_bytes_per_row`` for each, beside the caller's resident bytes.
    It shrinks its blocks to fit, down to one row, and refuses a budget
    without room for that row; this follows its arithmetic. A block of the
    rows of one stored chunk, an inner chunk of a sharded array, decodes each
    chunk once, where a smaller block decodes the same chunk again for each
    block.

    Args:
        counts: A chunked matrix of counts, such as ``assay.rawData[cells, :]``.
        scratch_bytes_per_row: The bytes that the caller makes of each row of
            a block.

    Returns:
        The bytes of a stream of one-row blocks, and of a stream of blocks of
        a stored chunk's rows, each beside the caller's resident bytes.
    """
    rows = min(counts._block_size, max(1, counts.shape[0]))
    per_row = max(1, counts._block_owned_bytes() // rows + scratch_bytes_per_row)
    fixed = counts._resident_bytes() + counts._max_decode_bytes()
    geometry = counts._geometry()
    chunk_rows = rows if geometry is None else min(rows, geometry.axisChunk(0))
    return fixed + per_row, fixed + chunk_rows * per_row


def _stream_beside(ledger: MemoryLedger, counts: Any, block: int) -> int:
    """Return what a stream of ``counts`` whose blocks hold ``block`` bytes needs beside the bytes that ``ledger`` charges.

    ``block`` counts, beside the block's rows and decode, the chunk that the
    reader's previous read can still hold, as ``ChunkedArray._held_read_bytes``
    states it; before the stream's first block that previous read is the
    ledger's last, whose leftover ``ledger.left`` counts. The two are one
    chunk, so the stream needs the larger of them beside the rest of its
    block, which a later stream of the same counts, after the leftover of
    this one, needs too.
    """
    held = int(counts._held_read_bytes())
    return max(ledger.left, held) + block - held


def _charge_stream(
    ledger: MemoryLedger, counts: Any, scratch_bytes_per_row: int, item: str
) -> int:
    """Charge ``ledger`` a stream of ``counts`` in blocks of a stored chunk's rows as ``item``, plan the room it reads ahead in, and return the resident bytes it plans beside.

    The stream holds a block of the rows of a stored chunk beside what the
    ledger charges, as :func:`_stream_bytes` counts it, with the chunk that
    the ledger's last read can still hold in place of the one that the
    block's previous read can, as :func:`_stream_beside` counts them; the
    ledger refuses that beyond its budget: a smaller block decodes the same
    chunk again for each block, so it would save little memory for much
    time. The stream then reads as many blocks ahead as the room holds,
    which the ledger plans for, beside the returned resident bytes: what the
    ledger charges and what its last read can hold beyond the block's own
    held chunk.
    """
    _least, block = _stream_bytes(counts, scratch_bytes_per_row)
    beside = _stream_beside(ledger, counts, block)
    ledger.peak(item, beside - ledger.left)
    room = ledger.room
    ledger.peak(
        item, beside - ledger.left if room is None else max(beside - ledger.left, room)
    )
    return ledger.charged + beside - block


def _count_stream(
    ledger: MemoryLedger,
    counts: Any,
    *,
    nthreads: int,
    msg: str,
    item: str,
    scratch_bytes_per_row: int = 0,
) -> Iterable[np.ndarray]:
    """Stream the row blocks of ``counts`` within the ledger's room, after charging a block of a stored chunk's rows as ``item``.

    The ledger is charged as :func:`_charge_stream` charges it, and the
    stream plans within its room, beside ``ledger.held_bytes``. Once the
    blocks are read, the ledger counts the chunk that the last read can
    still hold.
    """
    resident = _charge_stream(ledger, counts, scratch_bytes_per_row, item)
    blocks: Iterator[np.ndarray] = counts._stream_blocks(
        nthreads=nthreads,
        msg=msg,
        prefetch=None,
        row_mask=None,
        resident_bytes=resident,
        memory_bytes=ledger.budget,
        scratch_bytes_per_row=scratch_bytes_per_row,
    )
    return _leaving(ledger, blocks, counts._held_read_bytes())


def _leaving(
    ledger: MemoryLedger, blocks: Iterator[np.ndarray], held: int
) -> Iterator[np.ndarray]:
    """Yield ``blocks``, then record that the stream's last read may still hold ``held`` bytes.

    The reader that decoded the last chunk of a stream can keep it, as stored
    and decoded, after the stream ends, as the chunk of a metadata read does.
    """
    yield from blocks
    ledger.leave(held)


def _count_blocks(
    store: Any,
    ledger: MemoryLedger,
    assay: Any,
    features: np.ndarray,
    cells: np.ndarray,
    *,
    scratch_bytes_per_row: int,
    item: str,
) -> Iterable[np.ndarray]:
    """Stream the raw counts of features over cells, cells x features, in the order of ``cells``.

    The ledger is charged a block of a stored chunk's rows with the
    ``scratch_bytes_per_row`` that the caller makes of each row, as
    :func:`_charge_stream` charges it, as ``item``, and the blocks shrink to
    fit its room when it leaves less than whole blocks. The counts may come
    from their stored transpose, ``countsT``, when that reads less. Once the
    blocks are read, the ledger counts the chunk that the last read of either
    array can still hold.
    """
    from ...assay.normalization import feature_value_blocks
    from ...metadata.selection import NormalizationSpec

    counts = _count_matrix(assay, features, cells)
    resident = _charge_stream(ledger, counts, scratch_bytes_per_row, item)
    _, blocks = feature_value_blocks(
        assay,
        features,
        cells,
        NormalizationSpec(source="raw"),
        nthreads=store.nthreads,
        resident_bytes=resident,
        scratch_bytes_per_row=scratch_bytes_per_row,
        memory_bytes=ledger.budget,
    )
    return _leaving(ledger, iter(blocks), _count_leftover(counts))


def _count_leftover(counts: Any) -> int:
    """Return what the last read of a stream of ``counts`` from :func:`_count_matrix` can still hold, from the counts or their stored transpose."""
    held = int(counts._held_read_bytes())
    transposed = getattr(counts, "_transposed", None)
    if transposed is not None:
        geometry = array_geometry(transposed)
        if geometry is not None:
            held = max(held, geometry.heldReadBytes())
    return held


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
    """Return the bytes that :func:`_count_blocks` holds beside its caller's, with blocks of a stored chunk's rows.

    That is the stream's index arrays and one block of the counts of
    ``features`` over the ``cells`` of a stored chunk, with its decode, the
    chunk that the previous read can still hold, and the
    ``scratch_bytes_per_row`` that the caller makes of each row.
    """
    _least, block = _stream_bytes(
        _count_matrix(assay, features, cells), scratch_bytes_per_row
    )
    return block


def _band_bytes(
    ledger: MemoryLedger,
    assay: Any,
    features: np.ndarray,
    cell_sets: Sequence[np.ndarray],
    *,
    per_feature: int,
    scratch_per_feature: int,
) -> int:
    """Return what reading one band of ``features`` needs beside what ``ledger`` holds, the bytes to charge it as a peak.

    The band holds ``per_feature`` bytes for each of its features, and the
    stream of its counts over each of ``cell_sets`` in turn, with
    ``scratch_per_feature`` bytes for each feature of a row, as
    :func:`_count_stream_bytes` counts it, beside the chunk that the ledger's
    last read can still hold, as :func:`_stream_beside` counts them. A band
    of one feature holds this for ``features[:1]``, and as much for any other
    feature.
    """
    width = len(features)
    streams = max(
        _stream_beside(
            ledger, matrix, _stream_bytes(matrix, scratch_per_feature * width)[1]
        )
        for matrix in (_count_matrix(assay, features, cells) for cells in cell_sets)
    )
    return per_feature * width + streams - ledger.left


def _count_bands(
    ledger: MemoryLedger,
    assay: Any,
    features: np.ndarray,
    cell_sets: Sequence[np.ndarray],
    *,
    per_feature: int,
    scratch_per_feature: int,
    most: int,
) -> list[tuple[int, int]]:
    """Split ``features`` into consecutive bands whose reads fit the ledger's room.

    A band holds what :func:`_band_bytes` counts for it beside what
    ``ledger`` charges, its counts streamed in blocks of a stored chunk's rows
    beside the chunk that the ledger's last read, or the band's stream before
    it, can still hold, as :func:`_stream_beside` counts them. Each band takes as many features as fit, at most ``most``, and
    every band of ``most`` features without a budget; a band of one feature
    is taken whatever it holds, since the caller refuses a budget that it
    does not fit. What a stream decodes depends on the stored chunks of the
    band's features, so each band is sized on its own.

    Returns:
        Each band's first position in ``features`` and the position past
        its last feature, in order.
    """
    matrices = [_count_matrix(assay, features, cells) for cells in cell_sets]
    budget, charged = ledger.budget, ledger.charged

    def fits(start: int, stop: int) -> bool:
        if budget is None:
            return True
        width = stop - start
        streams: int = max(
            _stream_beside(
                ledger,
                band,
                _stream_bytes(band, scratch_per_feature * width)[1],
            )
            for band in (matrix[:, start:stop] for matrix in matrices)
        )
        return charged + per_feature * width + streams <= budget

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


def _cell_values(
    store: Any, ledger: MemoryLedger, ref: ArtifactRef, value: str | None = None
) -> np.ndarray:
    """Read one per-cell array of a cell-aligned artifact, without cell IDs.

    ``value`` names the array, None for the canonical one. The read charges
    ``ledger``, which keeps the values and the cells' positions charged, and
    a refusal of the read names the call's step in the one form of every
    screen refusal.

    Raises:
        MemoryBudgetError: If the read would pass the ledger's budget.
    """
    try:
        resolved = resolve_cell_aligned_artifact(
            store.zw, ref, value_name=value, ndim=None, ledger=ledger
        )
    except MemoryBudgetError as refusal:
        raise memory_refusal(
            ledger.doing,
            refusal.item,
            refusal.needed_bytes,
            refusal.budget_bytes,
            remedy=ledger.remedy,
        ) from refusal
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
    store: Any,
    ledger: MemoryLedger,
    column: str | None,
    cell_index: np.ndarray,
    *,
    reason: str,
) -> _CellSamples:
    """Group cells into samples by their label in a cell metadata column.

    Without a column every cell is in one sample. ``reason`` says why every
    cell needs a label; it ends the error for a cell without one. The labels
    and their missing mask are read through ``ledger``, and numbering them
    charges it as :func:`~cytearc.metadata.selection.sample_codes` states;
    the cells' sample codes stay charged and the labels are let go.
    """
    columns = () if column is None else (column,)
    n_cells = len(cell_index)
    labels = []
    before = ledger.charged
    for name in columns:
        array = store.cells._get_array(name)
        item = f"the cells' labels in {name!r}"
        if np.dtype(array.dtype).kind == "U":
            values = read_text_rows(ledger, array, cell_index, item=item)
        else:
            values = read_rows(ledger, array, cell_index, item=item)
            # The labels as Python objects, as numbering them makes them.
            ledger.hold(item, object_array_bytes(n_cells) + _NUMBER_BYTES * n_cells)
            values = values.astype(object)
        mask = metadata_missing_mask(store.cells, name)
        missing = (
            None
            if mask is None
            else read_rows(
                ledger, mask, cell_index, item=f"the cells' missing labels in {name!r}"
            ).astype(bool, copy=False)
        )
        if missing is not None and missing.any():
            # The labels with the missing ones as None, an object copy.
            ledger.hold(item, object_array_bytes(n_cells))
        labels.append(apply_missing_mask(values, missing, labels=True))
        del values, missing
    held = ledger.charged - before
    codes, sample_labels, assignments = sample_codes(
        columns, labels, n_cells, reason=reason, ledger=ledger
    )
    del labels
    ledger.release(held)
    return _CellSamples(
        columns=columns,
        codes=codes,
        names=[None if column is None else by[column] for by in sample_labels],
        assignments=assignments,
    )


def _feature_text(ledger: MemoryLedger, assay: Any, column: str) -> np.ndarray:
    """Read the feature column ``column`` of ``assay`` as fixed-width text, keeping it charged.

    The column is read whole through ``read_whole``, which charges the
    chunks that its read decodes; a column of numbers is held again as text,
    at the width that NumPy gives its dtype, charged before it is made.

    Raises:
        KeyError: If the assay has no feature column of that name.
        TypeError: If the column holds neither text nor numbers.
    """
    array = assay.feats._get_array(column)
    item = f"the features' {column}"
    values = read_whole(ledger, assay.feats._group, column, item=item)
    if values.dtype.kind == "U":
        return values
    if values.dtype.kind not in "biuf":
        raise TypeError(
            f"The feature column {column!r} of assay {assay.name!r} must hold "
            f"text or numbers, not {array.dtype}"
        )
    width = np.empty(0, dtype=values.dtype).astype(str).dtype.itemsize // 4
    ledger.hold(item, 4 * width * values.size)
    text: np.ndarray = values.astype(str)
    stored_bytes = values.nbytes
    del values
    ledger.release(stored_bytes)
    return text


def _cell_totals(
    ledger: MemoryLedger, assay: Any, cells: np.ndarray, *, item: str
) -> np.ndarray:
    """Read the count totals of distinct ``cells`` of ``assay`` as float64, keeping them charged as ``item``."""
    totals = read_rows(
        ledger,
        assay.cells._get_array(assay._totals_name),
        np.asarray(cells, dtype=np.int64),
        item=item,
    )
    return _converted(ledger, totals, np.float64, item)


def _row_features(
    store: Any,
    ledger: MemoryLedger,
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
    describes. The guides' column is read through ``ledger``, and it and
    what matching builds are let go once the targets are matched. Returns
    each row's feature position, -1 without a match, which stays charged.

    Raises:
        ValueError: If ``id_column`` is not a feature column of the guide
            assay.
    """
    before = ledger.charged
    guide_feature_ids = store._guide_feature_column(
        store._get_assay(str(labels.assay)), id_column, "id_column", ledger=ledger
    )
    column_bytes = ledger.charged - before
    ledger.hold("each row's feature", 8 * len(rows.codes))
    ledger.peak(
        "matching the targets to features",
        target_feature_bytes(
            len(rows.target_names), feature_ids, feature_names, guide_feature_ids
        ),
    )
    matched: np.ndarray = target_features(
        rows.target_names,
        rows.guide_targets,
        guide_feature_ids,
        feature_ids,
        feature_names,
    )[rows.codes]
    del guide_feature_ids
    ledger.release(column_bytes)
    return matched


def _label_rows(
    store: Any,
    ledger: MemoryLedger,
    labels: ArtifactRef,
    by: str,
    *,
    control_guides: bool = False,
) -> _LabelRows:
    """Read the rows of a per-perturbation table: each target or each guide.

    A row is a target that is not a control, or with ``by="guide"`` a guide
    whose target is not a control, in assay order; ``control_guides``
    makes the guides of control targets rows too. The labels' targets and
    the guides' targets are read through ``ledger``, and the rows stay
    charged with them.
    """
    status = require_complete_artifact(store.zw, labels)
    group = artifact_group(store.zw, labels)
    calls = artifact_group(store.zw, status.input_ref("guide_calls"))

    target_names = read_whole(ledger, group, "targets", item="the targets")
    flags_item = "the targets' control flags"
    is_control = _converted(
        ledger,
        read_whole(ledger, group, "target_is_control", item=flags_item),
        bool,
        flags_item,
    )
    targets_item = "the guides' targets"
    guide_targets = _converted(
        ledger,
        read_whole(ledger, group, "guide_target_codes", item=targets_item),
        np.int64,
        targets_item,
    )
    # Each row's target, name and guide, at most one row for each target or
    # guide.
    n_candidates = len(target_names) if by == "target" else len(guide_targets)
    ledger.hold("the rows", 16 * n_candidates)
    if by == "target":
        guides = np.empty(0, dtype=np.int64)
        codes = np.flatnonzero(~is_control)
        names = target_names[codes]
        ledger.hold("the rows", names.nbytes)
    else:
        guides = np.flatnonzero(
            (guide_targets >= 0)
            & (control_guides | ~is_control[np.maximum(guide_targets, 0)])
        )
        codes = guide_targets[guides]
        guide_ids = read_whole(ledger, calls, "guides", item="the guide IDs")
        names = guide_ids[guides]
        ledger.hold("the rows", names.nbytes)
        ids_bytes = guide_ids.nbytes
        del guide_ids
        ledger.release(ids_bytes)
    flags_bytes = is_control.nbytes
    del is_control
    ledger.release(flags_bytes)
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
# The bytes that finding a block's rows holds for each of its cells beside
# the arrays it reads: their positions, rows and the checks that pick them.
_LABEL_BLOCK_BYTES_PER_CELL = 32


@dataclass(frozen=True, slots=True)
class _CellRowArrays:
    """The stored arrays that give each labelled cell the row it carries alone.

    A cell carries a target alone when its called guides target that gene and
    no other, and a guide alone when it is the cell's one called guide: by
    target, the labels' ``n_targets`` and ``target_codes``; by guide, the
    calls' ``n_guides`` and ``guide_codes``. Only each cell's first code is
    used.

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

    def blocks(self) -> Iterator[tuple[int, int]]:
        """Yield the first and past-last cell of each block of cells read at once: whole chunks of the codes, up to a bound."""
        n_labelled = int(self.is_control.shape[0])
        chunk_rows = max(1, int(self.codes.chunks[0]))
        block_rows = max(chunk_rows, _LABEL_BLOCK_ROWS // chunk_rows * chunk_rows)
        for start in range(0, n_labelled, block_rows):
            yield start, min(start + block_rows, n_labelled)

    def rows(
        self, ledger: MemoryLedger, start: int, stop: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the row that each cell from ``start`` to ``stop`` carries alone, or -1, and its control flag.

        The cells' flags, counts and codes are read through ``ledger``; the
        rows and flags stay charged, and the rest is let go.
        """
        n_cells = stop - start
        before = ledger.charged
        ledger.hold("the labelled cells' rows", _LABEL_BLOCK_BYTES_PER_CELL * n_cells)
        cells = np.arange(start, stop, dtype=np.int64)
        control = read_rows(
            ledger, self.is_control, cells, item="the labelled cells' control flags"
        ).astype(bool, copy=False)
        counts = read_rows(
            ledger, self.counts, cells, item="the labelled cells' numbers of targets"
        )
        codes = read_rows(ledger, self.codes, cells, item="the labelled cells' targets")
        row = np.where(counts == 1, self.row_of[np.maximum(codes[:, 0], 0)], -1)
        del counts, codes, cells
        ledger.release(ledger.charged - before - row.nbytes - control.nbytes)
        return row, control


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
    store: Any, ledger: MemoryLedger, labels: ArtifactRef, rows: _LabelRows, by: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the row that each labelled cell carries alone, or -1, and its control flag.

    A cell carries a target alone when its called guides target that gene
    and no other, and a guide alone when it is the cell's one called guide.
    The labels are read through ``ledger`` a block of cells at a time, and
    the rows and flags of every cell stay charged.

    Raises:
        ValueError: If no labelled cell carries control guides only.
    """
    arrays = _cell_row_arrays(store, labels, rows, by)
    n_labelled = int(arrays.is_control.shape[0])
    ledger.hold("the labelled cells' rows and control flags", 9 * n_labelled)
    groups = np.empty(n_labelled, dtype=np.int64)
    control = np.empty(n_labelled, dtype=bool)
    for start, stop in arrays.blocks():
        row, flag = arrays.rows(ledger, start, stop)
        groups[start:stop] = row
        control[start:stop] = flag
        block_bytes = row.nbytes + flag.nbytes
        del row, flag
        ledger.release(block_bytes)
    if not control.any():
        raise ValueError(_NO_CONTROL_CELLS)
    return groups, control


def _compared_cells(
    store: Any, ledger: MemoryLedger, labels: ArtifactRef, rows: _LabelRows, by: str
) -> _ComparedCells:
    """Return the cells that a per-perturbation table compares, keeping them charged.

    The labels are read through ``ledger`` a block of cells at a time, so
    that only the arrays of the compared cells grow with the cells: a screen
    with many labelled cells and few compared ones holds little.

    Raises:
        ValueError: If no labelled cell carries control guides only.
    """
    arrays = _cell_row_arrays(store, labels, rows, by)
    # The labels' cells are checked against the cell IDs before any array
    # of the compared cells is held.
    selection = rows.status.input_ref("cell_selection")
    ledger.peak(
        "checking the labels' cells against the cell IDs",
        _cell_selection_check_bytes(store),
    )
    validate_cell_selection(store.zw, selection)
    positions, groups, control = [], [], []
    for start, stop in arrays.blocks():
        row, flag = arrays.rows(ledger, start, stop)
        # The block's compared cells: their positions among the labels, rows
        # and flags.
        ledger.peak("the compared cells", 8 * len(row))
        kept = np.flatnonzero((row >= 0) | flag)
        ledger.hold("the compared cells", 17 * len(kept))
        positions.append(kept + start)
        groups.append(row[kept])
        control.append(flag[kept])
        block_bytes = row.nbytes + flag.nbytes
        del row, flag, kept
        ledger.release(block_bytes)
    if not any(flags.any() for flags in control):
        raise ValueError(_NO_CONTROL_CELLS)
    n_compared = sum(len(block) for block in positions)
    # The positions joined, then placed in the store, beside the blocks.
    ledger.hold("the compared cells", 8 * n_compared)
    compared = np.concatenate(positions)
    del positions
    ledger.release(8 * n_compared)
    cells = _store_positions(store, ledger, selection, compared)
    ledger.hold("the compared cells", 9 * n_compared)
    found = _ComparedCells(
        cells=cells, groups=np.concatenate(groups), control=np.concatenate(control)
    )
    del groups, control
    ledger.release(9 * n_compared)
    ledger.release(compared.nbytes)
    return found


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


def _selection_read(
    store: Any, ledger: MemoryLedger, selection: ArtifactRef
) -> tuple[int, int]:
    """Return what reading a cell selection a block of cells at a time holds, and what its last read leaves.

    A block holds its flags and the store positions of its cells, beside
    those of the block before it, and the positions of the cells looked up in
    it, at most 64 bytes for each of its cells, and the chunk of flags that
    its read decodes beside one on each of the ledger's codec threads that
    the read before can still hold.
    """
    flags = as_zarr_array(artifact_group(store.zw, selection)["values"], name="values")
    n_rows = int(flags.shape[0])
    block_rows = min(_SELECTION_BLOCK_ROWS, n_rows)
    geometry = array_geometry(flags)
    # A selection's flags are one-dimensional, so stored in chunks.
    assert geometry is not None
    chunks = -(-max(1, n_rows) // geometry.axisChunk(0))
    held = geometry.heldReadBytes(ledger.holding_threads(chunks))
    # Before the first block, the read before is the ledger's last, whose
    # leftover the ledger counts already.
    reading = block_rows * 64 + array_row_selection_parts(flags)[0]
    return reading + max(0, held - ledger.left), held


def _store_positions(
    store: Any, ledger: MemoryLedger, selection: ArtifactRef, compact: np.ndarray
) -> np.ndarray:
    """Return the store positions of a selection's cells at ascending ``compact`` positions, keeping them charged.

    The selection is read a block of cells at a time through ``ledger``.
    """
    ledger.hold("the cells' positions in the store", 8 * len(compact))
    positions = np.empty(len(compact), dtype=np.int64)
    reading, held = _selection_read(store, ledger, selection)
    ledger.peak("reading the cell selection", reading)
    with zarr_decoding_threads(ledger.codec_threads):
        for block in _selection_blocks(store, selection):
            low, high = np.searchsorted(
                compact, (block.compact_start, block.compact_stop)
            )
            positions[low:high] = block.selected_indices[
                compact[low:high] - block.compact_start
            ]
    ledger.leave(held)
    return positions


def _selected_rows(
    store: Any, ledger: MemoryLedger, selection: ArtifactRef, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return where each of ascending store positions sits among a selection's cells, keeping them charged.

    Also returns whether each is a cell of the selection; the position of one
    that is not means nothing. The selection is read a block of cells at a
    time through ``ledger``.
    """
    ledger.hold("where the cells sit in the cell selection", 9 * len(cells))
    found = np.zeros(len(cells), dtype=np.int64)
    present = np.zeros(len(cells), dtype=bool)
    reading, held = _selection_read(store, ledger, selection)
    ledger.peak("reading the cell selection", reading)
    with zarr_decoding_threads(ledger.codec_threads):
        for block in _selection_blocks(store, selection):
            low, high = np.searchsorted(cells, (block.start, block.stop))
            within, here = _positions_in(block.selected_indices, cells[low:high])
            found[low:high] = block.compact_start + within
            present[low:high] = here
    ledger.leave(held)
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
    from ...storage.partition import scan_band

    ids = as_zarr_array(store.zw["cellData"]["ids"], name="ids")
    n_cells = int(ids.shape[0])
    held = 2 * n_cells
    if selected_ids:
        band = min(n_cells, scan_band(array_geometry(ids), fallback=n_cells))
        itemsize = int(np.dtype(ids.dtype).itemsize)
        held += array_row_selection_parts(ids)[0] + 2 * band * itemsize
    return held
