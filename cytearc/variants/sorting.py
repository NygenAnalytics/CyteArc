"""Rows keyed by cell that arrive in any order, returned in bands of cells.

Files list their rows in their own order: mgatk groups rows by cell in the
order it read the cells, and Matrix Market files list them by variant. A
stored long table is sorted by cell, so a reader can take a band of cells at
a time. ``CellRowSorter`` holds rows in memory up to a budget and writes each
batch beyond it, sorted, to a run on local disk; ``bands`` then reads the
rows of each band of cells from every run and sorts them again. The input is
read once, whatever its order, and memory stays within the budget.
"""

import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from ..utils.logging import logger

__all__ = ["CellRowSorter", "sort_rows"]

# Rows of a run's chunk on disk: the unit that a band decompresses.
_RUN_CHUNK_ROWS = 1 << 18
_CODECS = (BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle"),)


def sort_rows(rows: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    """Return ``rows`` sorted by the fields ``keys``, the first most significant.

    Rows with equal keys keep their order.
    """
    if len(rows) < 2:
        return rows
    if tuple(keys) == ("cell", "key"):
        # One stable sort of a 64-bit key is faster than a lexical sort.
        combined = (rows["cell"].astype(np.int64) << 32) | rows["key"].astype(np.int64)
        order = np.argsort(combined, kind="stable")
        del combined
    else:
        order = np.lexsort(tuple(rows[name] for name in reversed(keys)))
    return rows[order]


@dataclass(slots=True)
class _Run:
    """A sorted run: on disk (``group``) or in memory (``rows``)."""

    cells: np.ndarray
    starts: np.ndarray
    rows: np.ndarray | None = None
    group: zarr.Group | None = None
    names: tuple[str, ...] = ()
    dtype: np.dtype[Any] | None = None

    def span(self, first: int, stop: int) -> tuple[int, int]:
        """The rows of cells ``[first, stop)``."""
        lower = int(np.searchsorted(self.cells, first, side="left"))
        upper = int(np.searchsorted(self.cells, stop, side="left"))
        return int(self.starts[lower]), int(self.starts[upper])

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.rows is not None:
            return self.rows[start:stop]
        assert self.group is not None and self.dtype is not None
        out = np.empty(stop - start, dtype=self.dtype)
        for name in self.names:
            array = self.group[name]
            assert isinstance(array, zarr.Array)
            out[name] = array[start:stop]
        return out


def _run_index(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct cells of sorted rows, which are not empty, and the start
    of each, with the end."""
    breaks = np.flatnonzero(cells[1:] != cells[:-1]) + 1
    starts = np.concatenate([[0], breaks, [len(cells)]]).astype(np.int64)
    return cells[starts[:-1]].astype(np.int64), starts


@dataclass(slots=True)
class _Stream:
    dtype: np.dtype[Any]
    keys: tuple[str, ...]
    rows_per_cell: np.ndarray
    buffered: list[np.ndarray] = field(default_factory=list)
    buffered_bytes: int = 0
    runs: list[_Run] = field(default_factory=list)


class CellRowSorter:
    """Gather rows keyed by cell in any order and return them by bands of cells.

    Each stream holds rows of one structured dtype whose ``cell`` field is a
    cell's position, from 0 to ``n_cells - 1``. Rows are held in memory until
    the streams' buffers together exceed ``buffer_bytes``; then each stream's
    buffer is sorted and written to a run in a temporary folder, compressed.
    :meth:`bands` returns every stream's rows of each band of cells, sorted
    by the stream's keys, after the last :meth:`add`. A spill holds about
    three times the buffer at its peak, and the last buffer stays in memory
    while bands are read.

    Args:
        streams: Each stream's name, row dtype and sort keys, the first of
            which is ``"cell"``.
        n_cells: The number of cells.
        buffer_bytes: The bytes of rows held in memory before a spill.
        temp_dir: The folder of the temporary runs, or None for the system's.

    Raises:
        ValueError: If a stream's dtype lacks ``cell`` or a key, or its keys
            do not start with ``"cell"``.
    """

    def __init__(
        self,
        streams: Mapping[str, tuple[np.dtype[Any], Sequence[str]]],
        *,
        n_cells: int,
        buffer_bytes: int,
        temp_dir: str | Path | None = None,
    ) -> None:
        self._streams: dict[str, _Stream] = {}
        for name, (dtype, keys) in streams.items():
            keys = tuple(keys)
            names = set(dtype.names or ())
            if not keys or keys[0] != "cell" or not set(keys) <= names:
                raise ValueError(
                    f"The stream {name!r} must sort by 'cell' first and by fields "
                    "of its rows"
                )
            self._streams[name] = _Stream(
                dtype=np.dtype(dtype),
                keys=keys,
                rows_per_cell=np.zeros(n_cells, dtype=np.int64),
            )
        self._n_cells = int(n_cells)
        self._buffer_bytes = int(buffer_bytes)
        self._temp_dir = temp_dir
        self._folder: Path | None = None
        self._group: zarr.Group | None = None
        self._n_runs = 0
        self._finished = False

    def __enter__(self) -> "CellRowSorter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def n_runs(self) -> int:
        """The runs written to disk so far."""
        return self._n_runs

    def rows_per_cell(self, stream: str) -> np.ndarray:
        """Each cell's rows of ``stream`` so far."""
        return self._streams[stream].rows_per_cell

    def add(self, stream: str, rows: np.ndarray) -> None:
        """Add rows to a stream, spilling the buffers when they outgrow the budget.

        Raises:
            ValueError: If the rows have another dtype, a cell is out of range,
                or the sorter has started returning bands.
        """
        if self._finished:
            raise ValueError("Rows cannot be added once bands are read")
        state = self._streams[stream]
        if rows.dtype != state.dtype:
            raise ValueError(f"Rows of {stream!r} must have the dtype {state.dtype}")
        if not len(rows):
            return
        cells = rows["cell"]
        if int(cells.min()) < 0 or int(cells.max()) >= self._n_cells:
            raise ValueError(f"Rows of {stream!r} name a cell out of range")
        state.rows_per_cell += np.bincount(cells, minlength=self._n_cells)
        state.buffered.append(rows)
        state.buffered_bytes += rows.nbytes
        if sum(other.buffered_bytes for other in self._streams.values()) > (
            self._buffer_bytes
        ):
            for other in self._streams.values():
                self._spill(other)

    def _runs_group(self) -> zarr.Group:
        if self._group is None:
            self._folder = Path(
                tempfile.mkdtemp(prefix="cytearc-sorted-runs-", dir=self._temp_dir)
            )
            self._group = zarr.open_group(
                str(self._folder), mode="w", use_consolidated=False
            )
        return self._group

    def _spill(self, state: _Stream) -> None:
        if not state.buffered:
            return
        rows = self._take_buffer(state)
        group = self._runs_group().create_group(str(self._n_runs))
        self._n_runs += 1
        names = tuple(state.dtype.names or ())
        chunk = min(len(rows), _RUN_CHUNK_ROWS)
        for name in names:
            values = np.ascontiguousarray(rows[name])
            array = group.create_array(
                name,
                shape=values.shape,
                chunks=(chunk,),
                dtype=values.dtype,
                compressors=_CODECS,
            )
            array[:] = values
        cells, starts = _run_index(rows["cell"])
        state.runs.append(
            _Run(
                cells=cells, starts=starts, group=group, names=names, dtype=state.dtype
            )
        )

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        for state in self._streams.values():
            if not state.buffered:
                continue
            rows = self._take_buffer(state)
            cells, starts = _run_index(rows["cell"])
            state.runs.append(_Run(cells=cells, starts=starts, rows=rows))

    @staticmethod
    def _take_buffer(state: _Stream) -> np.ndarray:
        """A stream's buffered rows, sorted, with the buffer emptied first.

        Its pieces are dropped once joined, so the sort holds about twice
        the buffer and 16 bytes per row of keys at its peak.
        """
        rows = np.concatenate(state.buffered)
        state.buffered = []
        state.buffered_bytes = 0
        return sort_rows(rows, state.keys)

    def band_plan(self, max_rows: int) -> np.ndarray:
        """The first cell of each band and the end, so that a band holds at most
        ``max_rows`` rows of all streams together, or one cell.
        """
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        totals = np.zeros(self._n_cells, dtype=np.int64)
        for state in self._streams.values():
            totals += state.rows_per_cell
        cumulative = np.concatenate([[0], np.cumsum(totals)])
        bounds = [0]
        while bounds[-1] < self._n_cells:
            start = bounds[-1]
            stop = int(
                np.searchsorted(cumulative, cumulative[start] + max_rows, side="right")
                - 1
            )
            bounds.append(min(self._n_cells, max(start + 1, stop)))
        return np.asarray(bounds, dtype=np.int64)

    def bands(self, max_rows: int) -> Iterator[tuple[int, int, dict[str, np.ndarray]]]:
        """Yield each band's first cell, end and rows of every stream, sorted.

        A band holds at most ``max_rows`` rows of all streams together,
        unless one cell alone holds more.
        """
        self._finish()
        bounds = self.band_plan(max_rows)
        for first, stop in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
            rows: dict[str, np.ndarray] = {}
            for name, state in self._streams.items():
                pieces = []
                for run in state.runs:
                    start, end = run.span(first, stop)
                    if end > start:
                        pieces.append(run.read(start, end))
                if not pieces:
                    rows[name] = np.empty(0, dtype=state.dtype)
                elif len(pieces) == 1:
                    rows[name] = pieces[0]
                else:
                    rows[name] = sort_rows(np.concatenate(pieces), state.keys)
            yield first, stop, rows

    def close(self) -> None:
        """Remove the temporary runs; a failure to remove them is logged."""
        self._group = None
        if self._folder is None:
            return
        folder, self._folder = self._folder, None
        try:
            shutil.rmtree(folder)
        except OSError as error:
            logger.warning(f"Could not remove the temporary rows in {folder}: {error}")
