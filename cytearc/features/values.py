"""Resolve assay features and fetch their values without presentation dependencies."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metadata.membership import measured_rows
from ..metadata.selection import (
    FeatureReduction,
    FeatureRef,
    LookupBy,
    NormalizationSpec,
)
from ..metadata.table import CaseInsensitiveIndex

__all__ = [
    "ResolvedFeature",
    "fetch_normalized_feature_matrix",
    "measured_feature_means",
    "resolve_feature",
    "resolve_feature_batch",
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
        if len(indices) > 1 and ref.reduction is None:
            raise ValueError(
                f"Feature {ref.value!r} matches {len(indices)} entries in assay "
                f"{assay_name!r} at indices {indices}. Pass reduction='mean' or "
                f"reduction='sum', or look up by id/index."
            )

        idx_arr = np.asarray(indices)
        names = tuple(str(x) for x in index.names[idx_arr])
        ids = tuple(str(x) for x in index.ids[idx_arr])
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


def iter_normalized_feature_blocks(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    normalization: NormalizationSpec | None = None,
    *,
    resident_bytes: int = 0,
    unmeasured: dict[str, np.ndarray] | None = None,
) -> Iterator[tuple[list[int], int, np.ndarray]]:
    """Yield feature slots, selected-row offsets, and normalized value blocks."""
    from ..assay.normalization import feature_value_blocks, streamed_block_rows

    normalization = normalization or NormalizationSpec()
    assay_slots: dict[str, list[int]] = {}
    for slot, feat in enumerate(resolved):
        assay_slots.setdefault(feat.assay, []).append(slot)

    for assay_name, slots in assay_slots.items():
        assay = store._get_assay(assay_name)
        physical_indices = np.unique(
            np.concatenate(
                [np.asarray(resolved[slot].indices, dtype=np.int64) for slot in slots]
            )
        )
        local_indices = [
            np.searchsorted(physical_indices, resolved[slot].indices) for slot in slots
        ]
        log_raw = normalization.source == "raw" and normalization.transform == "log1p"

        def reduce_block(
            block: Any,
            slots: list[int] = slots,
            local: list[np.ndarray] = local_indices,
        ) -> np.ndarray:
            normalized = np.asarray(block, dtype=np.float64)
            if normalized.ndim == 1:
                normalized = normalized.reshape(-1, 1)
            if log_raw:
                normalized = np.log1p(normalized)
            output = np.empty((len(normalized), len(slots)), dtype=np.float64)
            for column, (slot, indices) in enumerate(zip(slots, local, strict=True)):
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
        # Membership comes from the assay's own cell table, which a run's
        # frozen cell view does not replace.
        measured = measured_rows(assay.cells, assay_name, cell_idx)
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
            continue
        if unmeasured is not None:
            unmeasured[assay_name] = ~measured
        read_idx = np.asarray(cell_idx)[measured]
        if len(read_idx) == 0:
            # No requested cell was measured, so nothing is normalized.
            blocks = ()
            piece_rows = streamed_block_rows(assay.rawData)
        else:
            # The membership mask and the positions of the measured cells are
            # held beside the stream, which charges the rows that it reads.
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
            blocks = (reduce_block(block) for block in raw_blocks)
        for start, output in spread_measured_rows(
            blocks, measured, len(slots), piece_rows
        ):
            yield slots, start, output


def measured_feature_means(
    assay: Any,
    feature_indices: np.ndarray,
    cell_idx: np.ndarray,
    *,
    nthreads: int,
) -> np.ndarray:
    """Return the mean normalized value of features in each cell.

    Cells that the assay did not measure are NaN.
    """
    from ..assay.normalization import feature_value_blocks

    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    measured = measured_rows(assay.cells, assay.name, cell_idx)
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
