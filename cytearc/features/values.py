"""Resolve assay features and fetch their values without presentation dependencies."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metadata.membership import (
    FeatureRecord,
    cell_sources,
    measured_features_by_source,
    measured_rows,
    present_sources,
)
from ..metadata.selection import (
    FeatureReduction,
    FeatureRef,
    LookupBy,
    NormalizationSpec,
)
from ..metadata.table import CaseInsensitiveIndex

__all__ = [
    "ResolvedFeature",
    "feature_description",
    "fetch_normalized_feature_matrix",
    "measured_feature_means",
    "resolve_feature",
    "resolve_feature_batch",
    "resolve_feature_rows",
]


@dataclass(frozen=True, slots=True)
class ResolvedFeature:
    """Concrete feature identity resolved against an assay."""

    assay: str
    by: LookupBy
    indices: tuple[int, ...]
    ids: tuple[str, ...]
    names: tuple[str, ...]
    label: str
    reduction: FeatureReduction | None
    raw: FeatureRef | str


class _AssayFeatureIndex:
    """Feature names, ids, and lookup maps read once for one assay."""

    __slots__ = ("_by_id", "_by_name", "ids", "n_features", "names")

    def __init__(self, assay: Any) -> None:
        self.n_features = int(assay.feats.N)
        self.names = np.asarray(assay.feats.fetch_all("names"))
        self.ids = np.asarray(assay.feats.fetch_all("ids"))
        self._by_name: CaseInsensitiveIndex | None = None
        self._by_id: dict[str, list[int]] | None = None

    def by_name(self, value: str) -> list[int]:
        if self._by_name is None:
            self._by_name = CaseInsensitiveIndex(self.names)
        return self._by_name.positions(value)

    def by_id(self, value: str) -> list[int]:
        if self._by_id is None:
            by_id: dict[str, list[int]] = {}
            for index, identifier in enumerate(self.ids):
                by_id.setdefault(str(identifier), []).append(index)
            self._by_id = by_id
        return self._by_id.get(value, [])


def resolve_feature_batch(
    store: Any,
    features: Sequence[str | FeatureRef],
    *,
    from_assay: str | None = None,
) -> list[ResolvedFeature]:
    """Resolve several features in input order, reading each assay's index once.

    Lookups follow :func:`resolve_feature`: ids and indices are exact, names are
    case-insensitive, and a name or id matching several features requires an
    explicit reduction.
    """
    indexes: dict[str, _AssayFeatureIndex] = {}
    resolved: list[ResolvedFeature] = []
    for feature in features:
        if isinstance(feature, FeatureRef):
            ref = feature
        else:
            ref = FeatureRef(value=feature, assay=from_assay)

        assay_name = ref.assay or from_assay or store._defaultAssay
        index = indexes.get(assay_name)
        if index is None:
            index = _AssayFeatureIndex(store._get_assay(assay_name))
            indexes[assay_name] = index

        if ref.by == "index":
            # FeatureRef stores an index as a Python int.
            assert isinstance(ref.value, int)
            idx = ref.value
            if idx < 0 or idx >= index.n_features:
                raise KeyError(
                    f"Feature index {idx} out of range for assay {assay_name!r} "
                    f"(N={index.n_features})"
                )
            indices = [idx]
        elif ref.by == "id":
            indices = index.by_id(str(ref.value))
        else:
            indices = index.by_name(str(ref.value))

        if len(indices) == 0:
            raise KeyError(
                f"Feature {ref.value!r} not found in assay {assay_name!r} by {ref.by!r}"
            )
        idx_arr = np.asarray(indices)
        names = tuple(str(x) for x in index.names[idx_arr])
        ids = tuple(str(x) for x in index.ids[idx_arr])
        if len(indices) > 1 and ref.reduction is None:
            raise ValueError(
                f"Feature {ref.value!r} matches {len(indices)} entries in assay "
                f"{assay_name!r} at indices {indices}, with IDs "
                f"{', '.join(map(repr, ids))}. Pass FeatureRef(value=<ID>, "
                "by='id') for one of them, or reduction='mean' or "
                "reduction='sum'."
            )

        label = ref.label or (
            names[0] if len(names) == 1 else f"{ref.value}:{ref.reduction}"
        )
        resolved.append(
            ResolvedFeature(
                assay=assay_name,
                by=ref.by,
                indices=tuple(int(i) for i in indices),
                ids=ids,
                names=names,
                label=label,
                reduction=ref.reduction,
                raw=ref if isinstance(feature, FeatureRef) else feature,
            )
        )
    return resolved


def resolve_feature(
    store: Any,
    feature: str | FeatureRef,
    *,
    from_assay: str | None = None,
) -> ResolvedFeature:
    """Resolve exact feature IDs or case-insensitive names without implicit reduction."""
    return resolve_feature_batch(store, [feature], from_assay=from_assay)[0]


def feature_description(feature: str | FeatureRef) -> str:
    """Name a requested feature in a message: by its name, ID, or index."""
    if not isinstance(feature, FeatureRef):
        return repr(feature)
    if feature.by == "id":
        return f"ID {feature.value!r}"
    if feature.by == "index":
        return f"index {feature.value}"
    return repr(feature.value)


def resolve_feature_rows(
    features: Sequence[str | FeatureRef],
    *,
    assay: str,
    indices: np.ndarray,
    ids: np.ndarray,
    names: np.ndarray,
    where: str,
) -> list[int | None]:
    """Find each feature among the rows of a saved result, in input order.

    The rows are features that a result froze, such as the retained features
    of a trajectory aggregation, given by their index in ``assay``, their ID,
    and their name. Lookups follow :func:`resolve_feature_batch`: a string or a
    ``FeatureRef`` by name matches names case-insensitively, and a
    ``FeatureRef`` by ID or index matches exactly.

    Args:
        features: Feature names or ``FeatureRef`` values.
        assay: The assay of the rows. A ``FeatureRef`` that names an assay
            must name this one.
        indices: The feature index of each row in ``assay``.
        ids: The feature ID of each row.
        names: The feature name of each row.
        where: The rows as messages name them, such as ``"the retained
            features of the aggregation"``.

    Returns:
        The row of each feature, or None for a feature that matches no row,
        which the caller may skip or refuse.

    Raises:
        TypeError: If a feature is neither a string nor a ``FeatureRef``.
        ValueError: If a feature matches several rows, naming their IDs in the
            order of their indices, or a ``FeatureRef`` names another assay or
            a reduction, which would combine rows rather than find one.
    """
    names = np.asarray(names).astype(str)
    ids = np.asarray(ids).astype(str)
    indices = np.asarray(indices)
    by_name = CaseInsensitiveIndex(names)
    rows: list[int | None] = []
    for feature in features:
        if isinstance(feature, str):
            matches = by_name.positions(feature)
        elif isinstance(feature, FeatureRef):
            if feature.assay is not None and feature.assay != assay:
                raise ValueError(
                    f"Feature {feature_description(feature)} names assay "
                    f"{feature.assay!r}, but {where} belong to assay {assay!r}"
                )
            if feature.reduction is not None:
                raise ValueError(
                    f"Feature {feature_description(feature)} sets "
                    f"reduction={feature.reduction!r}, which combines features; "
                    f"pass a FeatureRef without a reduction to find one of {where}"
                )
            if feature.by == "name":
                matches = by_name.positions(feature.value)
            elif feature.by == "id":
                matches = np.flatnonzero(ids == feature.value).tolist()
            else:
                matches = np.flatnonzero(indices == feature.value).tolist()
        else:
            raise TypeError(
                "Features must be feature names or FeatureRef values, not "
                f"{type(feature).__name__}"
            )
        if len(matches) > 1:
            # IDs follow the features' order in the assay, whatever the rows' order.
            matched = np.asarray(matches)[np.argsort(indices[matches], kind="stable")]
            shared = ", ".join(repr(str(identifier)) for identifier in ids[matched])
            raise ValueError(
                f"Feature {feature_description(feature)} matches {len(matches)} "
                f"of {where}, with IDs {shared}. Pass "
                "FeatureRef(value=<ID>, by='id') to choose one."
            )
        rows.append(int(matches[0]) if matches else None)
    return rows


def fetch_normalized_feature_matrix(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    normalization: NormalizationSpec | None = None,
    *,
    unmeasured: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Return assay-native or raw feature values in requested feature order.

    Row blocks are sized so that they fit the memory budget beside the output.
    """
    output = np.empty((len(cell_idx), len(resolved)), dtype=np.float64)
    for slots, start, values in iter_normalized_feature_blocks(
        store,
        resolved,
        cell_idx,
        normalization,
        resident_bytes=output.nbytes,
        unmeasured=unmeasured,
    ):
        output[start : start + len(values), slots] = values
    return output


def normalization_is_cell_local(assay: Any, normalization: NormalizationSpec) -> bool:
    """Return whether each cell's values depend on that cell's counts alone.

    Plots read only the cells they show when this holds. The normalizer policy
    is ``cytearc.assay.normalization.normalization_is_cell_local``; it is
    imported when called, so importing this module loads no assay code.
    """
    from ..assay.normalization import normalization_is_cell_local as is_cell_local

    return is_cell_local(assay, normalization)


def spread_measured_rows(
    blocks: Iterable[np.ndarray],
    measured: np.ndarray,
    n_columns: int,
    piece_rows: int,
    *,
    fill: float = np.nan,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield every requested row in order, ``fill`` where it was not measured.

    ``blocks`` hold the values of the measured rows, in their requested
    order. Each yielded piece covers at most ``piece_rows`` consecutive
    requested rows, so the rows of unmeasured cells between and after the
    measured ones never form a larger block than the stream reads.
    """
    positions = np.flatnonzero(measured)
    total = len(measured)
    emitted = 0
    consumed = 0
    for block in blocks:
        n_rows = len(block)
        if n_rows == 0:
            continue
        block_positions = positions[consumed : consumed + n_rows]
        stop = int(block_positions[-1]) + 1
        while emitted < stop:
            end = min(emitted + piece_rows, stop)
            piece = np.full((end - emitted, n_columns), fill)
            low, high = np.searchsorted(block_positions, (emitted, end))
            piece[block_positions[low:high] - emitted] = block[low:high]
            yield emitted, piece
            emitted = end
        consumed += n_rows
    while emitted < total:
        end = min(emitted + piece_rows, total)
        yield emitted, np.full((end - emitted, n_columns), fill)
        emitted = end


def _record_of_features(assay: Any, features: np.ndarray) -> FeatureRecord | None:
    """Return an assay's record of which features each source measured, over ``features``, None when every source measured every one of them.

    Features that every source measured need no cell's source.
    """
    record = measured_features_by_source(assay.feats, features)
    if record is None or bool(record.measured.all()):
        return None
    return record


class _CellSources:
    """The source of each requested cell, with which features each source measured.

    ``codes`` gives the source of each requested cell that the assay
    measured, in their order, and ``present`` the sources among them.
    """

    __slots__ = ("codes", "measured", "present", "record")

    def __init__(
        self,
        assay: Any,
        record: FeatureRecord,
        cell_idx: np.ndarray,
        measured: np.ndarray | None,
    ) -> None:
        self.record = record
        self.measured = measured
        rows = cell_idx if measured is None else cell_idx[measured]
        self.codes = cell_sources(
            assay.cells, assay.feats, rows, record, assay=assay.name
        )
        self.present = present_sources(self.codes, len(record.sources))

    def measuring(self, features: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return whether each source measured every one of ``features``."""
        return self.record.measured_of(features).all(axis=1)

    def cells(self, sources: np.ndarray, n_cells: int) -> np.ndarray | None:
        """Return whether each requested cell's source is one of ``sources``, None for all of them.

        ``sources`` marks the sources as :meth:`measuring` does. A cell that
        the assay did not measure has none of them.
        """
        cells = sources[self.codes]
        if self.measured is not None:
            within = np.zeros(n_cells, dtype=bool)
            within[self.measured] = cells
            cells = within
        return None if bool(cells.all()) else cells


def _measured_cells_of_features(
    assay: Any, features: Sequence[int] | np.ndarray, cell_idx: np.ndarray
) -> np.ndarray | None:
    """Return whether each cell at ``cell_idx`` measured every one of ``features``, None for all of them.

    A cell measured a feature when its assay measured the cell
    (:func:`~cytearc.metadata.membership.measured_rows`) and, when the assay
    records which features each source of a merge measured, the cell's source
    measured the feature.
    """
    measured = measured_rows(assay.cells, assay.name, cell_idx)
    indices = np.asarray(features, dtype=np.int64)
    record = _record_of_features(assay, indices)
    if record is None:
        return measured
    sources = _CellSources(assay, record, np.asarray(cell_idx), measured)
    return sources.cells(sources.measuring(indices), len(cell_idx))


def _slot_groups(
    assay: Any,
    resolved: Sequence[ResolvedFeature],
    slots: list[int],
    cell_idx: np.ndarray,
    measured: np.ndarray | None,
) -> list[tuple[list[int], np.ndarray | None]]:
    """Group ``slots`` by the cells that measured every feature that each slot reads.

    Each group pairs its slots with those cells among ``cell_idx``, None for
    all of them. When the assay records no source that lacks a feature of the
    slots, the cells are those that the assay measured, ``measured``.
    """
    record = _record_of_features(
        assay,
        np.concatenate(
            [np.asarray(resolved[slot].indices, dtype=np.int64) for slot in slots]
        ),
    )
    if record is None:
        return [(slots, measured)]
    sources = _CellSources(assay, record, cell_idx, measured)
    groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for slot in slots:
        measuring = sources.measuring(resolved[slot].indices)
        # Slots that the same sources among the requested cells measured are
        # measured by the same cells, so they are read together.
        key = measuring[sources.present].tobytes()
        groups.setdefault(key, (measuring, []))[1].append(slot)
    return [
        (group_slots, sources.cells(measuring, len(cell_idx)))
        for measuring, group_slots in groups.values()
    ]


def _slot_blocks(
    store: Any,
    assay: Any,
    resolved: Sequence[ResolvedFeature],
    slots: list[int],
    cell_idx: np.ndarray,
    measured: np.ndarray | None,
    normalization: NormalizationSpec,
    resident_bytes: int,
) -> Iterator[tuple[list[int], int, np.ndarray]]:
    """Yield the normalized values of ``slots`` over ``cell_idx``, NaN where ``measured`` is False.

    Only the measured cells are read and normalized.
    """
    from ..assay.normalization import feature_value_blocks, streamed_block_rows

    physical_indices = np.unique(
        np.concatenate(
            [np.asarray(resolved[slot].indices, dtype=np.int64) for slot in slots]
        )
    )
    local_indices = [
        np.searchsorted(physical_indices, resolved[slot].indices) for slot in slots
    ]
    log_raw = normalization.source == "raw" and normalization.transform == "log1p"

    def reduce_block(block: Any) -> np.ndarray:
        normalized = np.asarray(block, dtype=np.float64)
        if normalized.ndim == 1:
            normalized = normalized.reshape(-1, 1)
        if log_raw:
            normalized = np.log1p(normalized)
        output = np.empty((len(normalized), len(slots)), dtype=np.float64)
        for column, (slot, indices) in enumerate(
            zip(slots, local_indices, strict=True)
        ):
            selected = normalized[:, indices]
            if selected.shape[1] == 1:
                output[:, column] = selected[:, 0]
            elif resolved[slot].reduction == "sum":
                output[:, column] = selected.sum(axis=1)
            else:
                output[:, column] = selected.mean(axis=1)
        return output

    # ``reduce_block`` holds two float64 copies of a block and its output.
    float_bytes = np.dtype(np.float64).itemsize
    scratch_bytes_per_row = (2 * len(physical_indices) + len(slots)) * float_bytes
    if measured is None:
        _, blocks = feature_value_blocks(
            assay,
            physical_indices,
            cell_idx,
            normalization,
            nthreads=store.nthreads,
            resident_bytes=resident_bytes,
            scratch_bytes_per_row=scratch_bytes_per_row,
        )
        start = 0
        for block in blocks:
            output = reduce_block(block)
            yield slots, start, output
            start += len(output)
        return
    read_idx = np.asarray(cell_idx)[measured]
    if len(read_idx) == 0:
        # No requested cell was measured, so nothing is normalized.
        measured_blocks: Iterable[np.ndarray] = ()
        piece_rows = streamed_block_rows(assay.rawData)
    else:
        # The mask of measured cells and their positions are held beside the
        # stream, which charges the rows that it reads.
        held = measured.nbytes + read_idx.nbytes
        piece_rows, raw_blocks = feature_value_blocks(
            assay,
            physical_indices,
            read_idx,
            normalization,
            nthreads=store.nthreads,
            resident_bytes=resident_bytes + held,
            scratch_bytes_per_row=scratch_bytes_per_row,
        )
        measured_blocks = (reduce_block(block) for block in raw_blocks)
    for start, output in spread_measured_rows(
        measured_blocks, measured, len(slots), piece_rows
    ):
        yield slots, start, output


def iter_normalized_feature_blocks(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    normalization: NormalizationSpec | None = None,
    *,
    resident_bytes: int = 0,
    unmeasured: dict[str, np.ndarray] | None = None,
) -> Iterator[tuple[list[int], int, np.ndarray]]:
    """Yield feature slots, selected-row offsets, and normalized value blocks.

    A value is NaN in a cell that did not measure the feature: a cell that
    its assay did not measure, which ``unmeasured`` receives by assay, or,
    when the assay records which features each source of a merge measured, a
    cell whose source did not measure it. Each slot's values are read and
    normalized over the cells that measured every feature it reads.
    """
    normalization = normalization or NormalizationSpec()
    assay_slots: dict[str, list[int]] = {}
    for slot, feat in enumerate(resolved):
        assay_slots.setdefault(feat.assay, []).append(slot)

    for assay_name, slots in assay_slots.items():
        assay = store._get_assay(assay_name)
        # Membership comes from the assay's own cell table, which a run's
        # frozen cell view does not replace.
        measured = measured_rows(assay.cells, assay_name, cell_idx)
        if measured is not None and unmeasured is not None:
            unmeasured[assay_name] = ~measured
        groups = _slot_groups(assay, resolved, slots, np.asarray(cell_idx), measured)
        # Each group's read also holds the cells of the groups beside it.
        masks = sum(cells.nbytes for _slots, cells in groups if cells is not None)
        for group_slots, cells in groups:
            yield from _slot_blocks(
                store,
                assay,
                resolved,
                group_slots,
                cell_idx,
                cells,
                normalization,
                resident_bytes + masks - (0 if cells is None else cells.nbytes),
            )


def measured_feature_means(
    assay: Any,
    feature_indices: np.ndarray,
    cell_idx: np.ndarray,
    *,
    nthreads: int,
) -> np.ndarray:
    """Return the mean normalized value of features in each cell.

    Cells that did not measure every one of the features are NaN
    (:func:`_measured_cells_of_features`), and only the others are read and
    normalized.
    """
    from ..assay.normalization import feature_value_blocks

    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    measured = _measured_cells_of_features(assay, feature_indices, cell_idx)
    read_idx = cell_idx if measured is None else cell_idx[measured]
    if len(read_idx) == 0:
        return np.full(len(cell_idx), np.nan)
    means = np.full(len(cell_idx), np.nan)
    positions = None
    held = means.nbytes
    if measured is not None:
        positions = np.flatnonzero(measured)
        held += positions.nbytes + measured.nbytes + read_idx.nbytes
    _, blocks = feature_value_blocks(
        assay,
        np.asarray(feature_indices),
        read_idx,
        NormalizationSpec(),
        nthreads=nthreads,
        resident_bytes=held,
        # One float64 mean per row.
        scratch_bytes_per_row=np.dtype(np.float64).itemsize,
    )
    start = 0
    for block in blocks:
        stop = start + len(block)
        values = block.mean(axis=1, dtype=np.promote_types(block.dtype, np.float64))
        if positions is None:
            means[start:stop] = values
        else:
            means[positions[start:stop]] = values
        start = stop
    return means
