import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metadata.rows import read_metadata_rows_chunkwise
from ..storage.cell_order import (
    SHUFFLE_BYTES_PER_CELL,
    shuffle_seed_argument,
    shuffled_order,
)
from ..storage.partition import checked_indices
from ..utils.digest import mix64_array
from ..utils.memory_budget import memory_refusal

# The text between a source's name and the source's own cell ID in each cell
# ID of a merged store.
MERGED_ID_SEPARATOR = "__"
# Why a merge cannot take a stored table of merged cells as its own.
CELL_ORDER_MISMATCH = (
    "its cell IDs or their order differ from this merge's, as when an earlier "
    "CyteArc release wrote it in blocks of one source or a source has changed"
)
# What a window of merged rows holds for each of its rows while it is in use:
# the positions and source rows of its parts, and the sort that groups them.
WINDOW_BYTES_PER_ROW = 24
# SplitMix64's increment, which spaces the counters of a stream.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
# "MrgStart" in ASCII, mixed into the merge's seed, so the first cell of each
# source takes a stream of its own beside the order of the sources' turns.
_START_SALT = np.uint64(0x4D72675374617274)


@dataclass(frozen=True, slots=True)
class RowPlan:
    """The order of the cells that a merge or composite takes from its sources.

    Merged row ``r`` holds the next cell of source ``owners[r]``. The
    sources take turns in the order that ``shuffled_order(nCells, seed)``
    gives their concatenated cells, so ``owners`` is a uniformly random
    arrangement of the sources' cells. Each source gives its cells in their
    stored order from the one at ``firstCells`` on, a uniformly random cell
    that ``seed`` also draws, and then from its first cell up to that one.
    When the sources list their cells in independent uniformly random
    orders, so does the merge. Sources that an import shuffled with one seed
    list each file row at about the same place, which the random first cells
    set apart.
    """

    # The source of each merged row, as the smallest unsigned integers that
    # hold every source.
    owners: np.ndarray
    # The rows that each source contributes, increasing, or None for every row.
    selectedRows: tuple[np.ndarray | None, ...]
    # The cells that each source contributes.
    sourceCells: tuple[int, ...]
    # The cell, among those that each source contributes in stored order,
    # that the source gives first; 0 for a source without cells.
    firstCells: tuple[int, ...]
    nCells: int
    seed: int
    sourceNames: tuple[str, ...]

    def resident_bytes(self) -> int:
        return int(self.owners.nbytes) + sum(
            int(rows.nbytes) for rows in self.selectedRows if rows is not None
        )

    def source_rows(
        self, source: int, start: int = 0, stop: int | None = None
    ) -> np.ndarray:
        """Return the rows of ``source`` that the merge takes, from its ``start``-th to before its ``stop``-th in stored order, increasing."""
        stop = self.sourceCells[source] if stop is None else stop
        selected = self.selectedRows[source]
        if selected is None:
            return np.arange(start, stop, dtype=np.int64)
        return selected[start:stop]

    def most_source_cells(self, window_rows: int) -> int:
        """Return the most cells that a window of ``window_rows`` merged rows takes from one source, of the windows that :func:`iter_row_plan_windows` cuts without boundaries."""
        n_sources = len(self.sourceNames)
        return max(
            (
                int(
                    np.bincount(
                        self.owners[start : start + int(window_rows)],
                        minlength=n_sources,
                    ).max()
                )
                for start in range(0, self.nCells, max(1, int(window_rows)))
            ),
            default=0,
        )

    def destination_rows(self, source: int) -> np.ndarray:
        """Return the merged row of each cell that the merge takes from ``source``, in the source's stored order.

        The rows increase from the source's first cell on, and again from
        its stored first cell up to that one. Beside the result, this holds a
        flag of every merged row and the merged rows of the source's cells.
        """
        rows = np.flatnonzero(self.owners == source)
        return np.roll(rows, self.firstCells[source])


@dataclass(frozen=True, slots=True)
class RowPlanPart:
    """The cells of one source in a window of merged rows.

    ``positions`` are their rows in the window and ``localRows`` their rows in
    the source, both increasing.
    """

    sourceIdx: int
    positions: np.ndarray
    localRows: np.ndarray


@dataclass(frozen=True, slots=True)
class RowPlanWindow:
    """Merged rows ``destStart`` to ``destStart + nRows`` and the cells of each source in them."""

    destStart: int
    nRows: int
    parts: tuple[RowPlanPart, ...]


def build_row_plan(
    n_cells_per_source: list[int],
    source_names: list[str],
    seed: int | None = 42,
    *,
    selected_rows: list[np.ndarray | None] | None = None,
    memory_bytes: int | None = None,
) -> RowPlan:
    """Interleave the sources' cells in the random order that ``seed`` draws.

    ``selected_rows`` gives, for each source, the rows that it contributes,
    in any order, or None for every row. Each source gives its cells in
    their stored order from a random first cell, wrapping round, as
    :class:`RowPlan` describes. ``seed`` None draws a new order. Building
    holds what the shuffle that arranges the sources' turns holds at its
    peak, ``SHUFFLE_BYTES_PER_CELL`` bytes for each merged cell, beside the
    selected rows, which must fit ``memory_bytes`` when given.

    Raises:
        MemoryBudgetError: If building needs more than ``memory_bytes``.
    """
    if len(n_cells_per_source) != len(source_names):
        raise ValueError("Source names must match the number of sources")
    if len(source_names) != len(set(source_names)):
        raise ValueError("A unique name must be provided for each source DataStore")
    if selected_rows is not None and len(selected_rows) != len(source_names):
        raise ValueError("Selected rows must match the number of sources")
    seed = secrets.randbits(64) if seed is None else shuffle_seed_argument(seed, "seed")

    chosen: list[np.ndarray | None] = []
    cells: list[int] = []
    for index, n_cells in enumerate(n_cells_per_source):
        selected = None if selected_rows is None else selected_rows[index]
        if selected is None:
            chosen.append(None)
            cells.append(int(n_cells))
            continue
        rows = np.sort(
            checked_indices(selected, limit=int(n_cells), name="selected_rows")
        )
        chosen.append(rows)
        cells.append(int(rows.size))
    n_total = sum(cells)
    needed = n_total * SHUFFLE_BYTES_PER_CELL + sum(
        int(rows.nbytes) for rows in chosen if rows is not None
    )
    if memory_bytes is not None and needed > int(memory_bytes):
        raise memory_refusal(
            f"Ordering {n_total} merged cells",
            "the shuffle that arranges the sources' turns",
            needed,
            int(memory_bytes),
        )
    # The concatenated cells of source s are the integers from the cells of
    # the sources before it up to its own last; a uniform permutation of them
    # gives the turn of every merged row to the source of its cell. Finding
    # them holds less than the shuffle: the order and the turns, 16 bytes a
    # cell, and then the turns and their sources.
    owners = np.searchsorted(
        np.cumsum(cells), shuffled_order(n_total, seed), side="right"
    ).astype(np.min_scalar_type(max(0, len(cells) - 1)), copy=False)
    return RowPlan(
        owners=owners,
        selectedRows=tuple(chosen),
        sourceCells=tuple(cells),
        firstCells=_first_cells(cells, seed),
        nCells=n_total,
        seed=seed,
        sourceNames=tuple(source_names),
    )


def _first_cells(cells: list[int], seed: int) -> tuple[int, ...]:
    """Return the cell that each source gives first: uniformly random among its ``cells``, from a stream of ``seed``."""
    key = mix64_array(np.array([seed], dtype=np.uint64) ^ _START_SALT)
    hashes = mix64_array(
        key ^ mix64_array(np.arange(len(cells), dtype=np.uint64) + _GOLDEN)
    )
    return tuple(
        int(value % np.uint64(count)) if count else 0
        for value, count in zip(hashes.tolist(), cells, strict=True)
    )


def iter_row_plan_windows(
    row_plan: RowPlan,
    window_rows: int,
    *,
    boundary_rows: int | None = None,
) -> Iterator[RowPlanWindow]:
    """Yield the merged rows in order, in windows of at most ``window_rows`` rows.

    No window crosses a multiple of ``boundary_rows``, such as the edge of a
    stored chunk band. A source whose cells wrap round from its last to its
    first within a window has a part for each side. A window in use holds
    ``WINDOW_BYTES_PER_ROW`` bytes for each of its rows.
    """
    if int(window_rows) < 1:
        raise ValueError("window_rows must be positive")
    if boundary_rows is not None and int(boundary_rows) < 1:
        raise ValueError("boundary_rows must be positive")
    n_sources = len(row_plan.sourceNames)
    taken = np.zeros(n_sources, dtype=np.int64)
    start = 0
    while start < row_plan.nCells:
        stop = min(row_plan.nCells, start + int(window_rows))
        if boundary_rows is not None:
            stop = min(stop, (start // int(boundary_rows) + 1) * int(boundary_rows))
        owners = row_plan.owners[start:stop]
        # A stable sort keeps each source's positions increasing.
        grouped = np.argsort(owners, kind="stable")
        counts = np.bincount(owners, minlength=n_sources)
        parts = []
        offset = 0
        for source in np.flatnonzero(counts).tolist():
            count = int(counts[source])
            positions = grouped[offset : offset + count]
            cells = row_plan.sourceCells[source]
            first = (int(taken[source]) + row_plan.firstCells[source]) % cells
            # The cells from ``first`` up to the source's last, then from its
            # first on.
            before_wrap = min(count, cells - first)
            parts.append(
                RowPlanPart(
                    sourceIdx=source,
                    positions=positions[:before_wrap],
                    localRows=row_plan.source_rows(source, first, first + before_wrap),
                )
            )
            if before_wrap < count:
                parts.append(
                    RowPlanPart(
                        sourceIdx=source,
                        positions=positions[before_wrap:],
                        localRows=row_plan.source_rows(source, 0, count - before_wrap),
                    )
                )
            offset += count
        taken += counts
        yield RowPlanWindow(destStart=start, nRows=stop - start, parts=tuple(parts))
        start = stop


def prefixed_cell_ids(name: str, source_ids: np.ndarray, dtype: Any) -> np.ndarray:
    """Return source cell IDs prefixed with ``{name}__`` in the merged dtype."""
    prefix = f"{name}{MERGED_ID_SEPARATOR}"
    if source_ids.dtype.kind in "UT":
        # Text concatenates in one pass, as each value formats itself.
        return np.asarray(np.char.add(prefix, source_ids), dtype=np.dtype(dtype))
    merged = np.empty(source_ids.size, dtype=np.dtype(dtype))
    for index, value in enumerate(source_ids):
        merged[index] = f"{prefix}{value}"
    return merged


def iter_merged_cell_ids(
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    dtype: Any,
    block_rows: int = 100_000,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield the merged cell IDs in merged row order, ``block_rows`` at a time.

    Each yield is ``(start, ids)``, where ``ids`` are the IDs of the merged
    rows from ``start``, already prefixed with ``{source_name}__``, so one
    generator can both write and verify cell identity.
    """
    if len(source_cell_tables) != len(row_plan.sourceNames):
        raise ValueError("Source cell tables must match the row plan")
    for window in iter_row_plan_windows(row_plan, block_rows):
        merged = np.empty(window.nRows, dtype=np.dtype(dtype))
        for part in window.parts:
            source_ids = read_metadata_rows_chunkwise(
                source_cell_tables[part.sourceIdx], "ids", part.localRows
            )
            merged[part.positions] = prefixed_cell_ids(
                row_plan.sourceNames[part.sourceIdx], source_ids, dtype
            )
            del source_ids
        yield window.destStart, merged


def verify_merged_cell_ids(
    stored_ids: Any,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    block_rows: int = 100_000,
) -> None:
    """Compare stored cell ids against the row-plan identity generator.

    Stored ids are read one whole chunk band at a time. Windows arrive in
    merged row order, so each band is read once.
    """
    n_cells = int(stored_ids.shape[0])
    if n_cells != row_plan.nCells:
        raise ValueError(CELL_ORDER_MISMATCH)
    chunk_rows = max(1, int(stored_ids.chunks[0]))
    band_start = 0
    band = np.asarray(stored_ids[0:0])
    for start, expected in iter_merged_cell_ids(
        row_plan,
        source_cell_tables,
        dtype=stored_ids.dtype,
        block_rows=block_rows,
    ):
        offset = 0
        while offset < expected.size:
            row = start + offset
            if row >= band_start + band.size:
                band_start = row - row % chunk_rows
                band = np.asarray(
                    stored_ids[band_start : min(band_start + chunk_rows, n_cells)]
                )
            width = min(expected.size - offset, band_start + band.size - row)
            actual = band[row - band_start : row - band_start + width]
            if not np.array_equal(actual, expected[offset : offset + width]):
                raise ValueError(CELL_ORDER_MISMATCH)
            offset += width
