"""Rows of a cell-major count matrix read from its stored feature-major copy.

A read of a few features over many cells touches many chunks of ``counts``,
one per row band, but few of ``countsT``, which holds the same values with the
features as rows. Read band by band, every ``countsT`` chunk is decoded once.
These helpers decide when that is cheaper and stream the raw rows from
``countsT`` in the row blocks that a ``counts`` read would give, so callers
transform them exactly as they transform rows read from ``counts``.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import zarr

from .budget import ResourceBudget
from .geometry import array_geometry
from .io_policy import StorageIoPolicy


class CellBandAssembler:
    """Gather each cell band's feature groups into one cells-by-features block.

    It is the ``process`` of an ordered, cell-major ``map_feature_cell_bands``
    stream, which hands over every feature group of a band before the next
    band and the bands in cell order, so a band is complete when its last
    group arrives. ``destinations`` maps each feature row of the stream's array
    to its output column, or -1, and ``metrics`` is the stream's metrics,
    which name the number of feature groups in every band.
    """

    def __init__(
        self,
        destinations: np.ndarray,
        n_features: int,
        dtype: Any,
        metrics: dict[str, Any],
    ) -> None:
        self._destinations = destinations
        self._n_features = n_features
        self._dtype = dtype
        self._metrics = metrics
        self._raw: np.ndarray | None = None
        self._groups = 0

    @property
    def pending(self) -> bool:
        """Whether a band is still missing feature groups."""
        return self._raw is not None or self._groups > 0

    def __call__(self, band: Any) -> np.ndarray | None:
        from .feature_stream import selected_feature_values

        target = self._destinations[band.featStart + band.featureRows()]
        keep = target >= 0
        if not np.any(keep):
            raise RuntimeError(
                "A countsT feature group contained no requested features"
            )
        rows = np.asarray(band.selectedDestinations, dtype=np.int64)
        start = int(rows[0])
        if not np.array_equal(rows, np.arange(start, start + len(rows))):
            raise ValueError("countsT reads require contiguous selected-cell bands")
        if self._raw is None:
            self._raw = np.empty((len(rows), self._n_features), dtype=self._dtype)
        values = selected_feature_values(band.values, keep)
        self._raw[:, target[keep]] = values[:, band.selectedLocal].T
        self._groups += 1
        if self._groups != int(self._metrics["featureGroupCount"]):
            return None
        result = self._raw
        self._raw = None
        self._groups = 0
        return result


def feature_major_row_scratch(
    counts: zarr.Array,
    counts_t: zarr.Array,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    block_rows: int,
    held_bytes: int,
    memory_bytes: int,
) -> int | None:
    """Return the scratch of streaming ``counts[rows][:, columns]`` from ``counts_t``.

    The scratch is what the stream's caller holds beside its band reads. It is
    None when the row blocks of ``counts`` touch no more chunk bytes than the
    bands of ``counts_t``, or when one band read does not fit ``memory_bytes``
    beside the stream's buffers.

    Args:
        counts: Cell-major counts.
        counts_t: Its feature-major copy.
        rows: Strictly increasing rows of ``counts`` to read.
        columns: Distinct columns of ``counts`` to read, in output order.
        block_rows: Rows of each yielded block.
        held_bytes: What the caller holds while the rows stream, including
            what it creates from each block.
        memory_bytes: The memory budget of the stream.

    Returns:
        The scratch, or None to read ``counts``.
    """
    from .feature_stream import persisted_read_group

    counts_geometry = array_geometry(counts)
    transposed = array_geometry(counts_t)
    if counts_geometry is None or transposed is None:
        return None
    feature_chunks = np.unique(columns // counts_geometry.chunks[1]).size
    row_chunks = sum(
        np.unique(rows[start : start + block_rows] // counts_geometry.chunks[0]).size
        for start in range(0, len(rows), block_rows)
    )
    row_bytes = row_chunks * feature_chunks * counts_geometry.nominalChunkBytes()
    column_bytes = (
        np.unique(rows // transposed.chunks[1]).size
        * np.unique(columns // transposed.chunks[0]).size
        * transposed.nominalChunkBytes()
    )
    if column_bytes >= row_bytes:
        return None
    width, _ = persisted_read_group(counts_t)
    band_rows = min(transposed.chunks[1], len(rows))
    raw_bytes = len(columns) * transposed.itemsize
    # The assembled band and its selected copy, the row block being filled
    # and the one handed on, and the feature destinations with their mask.
    scratch = (
        held_bytes
        + 2 * (band_rows + block_rows) * raw_bytes
        + transposed.shape[0] * (np.dtype(np.int64).itemsize + 1)
        + columns.nbytes
    )
    read_bytes = transposed.readBytes(
        width * min(transposed.chunks[1], transposed.shape[1]) * transposed.itemsize,
        -(-width // transposed.chunks[0]),
    )
    # The stream adds its cell destinations and sparse feature row indices.
    minimum = scratch + read_bytes + 2 * rows.nbytes + columns.nbytes
    return int(scratch) if minimum <= memory_bytes else None


def iter_feature_major_rows(
    counts_t: zarr.Array,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    block_rows: int,
    resources: ResourceBudget,
    io: StorageIoPolicy | None,
    scratch_bytes: int,
    progress: str | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``counts[rows][:, columns]`` in row blocks, read from ``counts_t``.

    Each block is ``(start, values)``: the block's first position in ``rows``
    and its raw values, ``block_rows`` rows except the last. ``rows`` must be
    strictly increasing and ``columns`` distinct; ``scratch_bytes`` comes from
    ``feature_major_row_scratch``. ``progress`` names the stream's progress bar.

    Raises:
        RuntimeError: If the bands do not cover every requested row.
    """
    from .feature_stream import map_feature_cell_bands

    block_rows = max(1, min(block_rows, len(rows)))
    destinations = np.full(counts_t.shape[0], -1, dtype=np.int64)
    destinations[columns] = np.arange(len(columns), dtype=np.int64)
    metrics: dict[str, Any] = {}
    assemble = CellBandAssembler(destinations, len(columns), counts_t.dtype, metrics)
    buffer = np.empty((block_rows, len(columns)), dtype=counts_t.dtype)
    filled = 0
    emitted = 0
    for band in map_feature_cell_bands(
        counts_t,
        assemble,
        cell_idx=rows,
        feat_idx=columns,
        resources=resources,
        progress=progress,
        io=io,
        metrics=metrics,
        scratchBytes=scratch_bytes,
        cellMajorOrder=True,
        orderedCompute=True,
    ):
        if band is None:
            continue
        start = 0
        while start < len(band):
            take = min(block_rows - filled, len(band) - start)
            buffer[filled : filled + take] = band[start : start + take]
            filled += take
            start += take
            if filled == block_rows:
                yield emitted, buffer
                emitted += filled
                buffer = np.empty_like(buffer)
                filled = 0
        del band
    if filled:
        yield emitted, buffer[:filled]
        emitted += filled
    if assemble.pending or emitted != len(rows):
        raise RuntimeError("countsT reads did not cover every requested cell")
