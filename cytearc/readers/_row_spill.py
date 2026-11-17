"""Send the rows of an import that shuffles cells to their destination rows on local disk.

An import that shuffles cells reads its rows in the file's order and needs
them in shuffled order. Each import appends the rows, as it reads them, to a
row bucket store (:class:`~cytearc.readers._sparse.SparseRowStore`) of whole
rows, tagged with their destination rows, and its counts write reads them
back in destination order. :class:`RowSpillPlan` sizes the store's buckets
and what its writers hold within the import's budget, the same way for every
format: from the rows that a value scan reads, as H5AD, 10x HDF5 and Tapestri
imports send them, or from batches of rows that an import reads in the
file's order, as Matrix Market, CSV and Seurat imports send them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ..utils.count_values import scan_window_bytes, scan_window_values
from ..utils.memory_budget import MemoryBudgetError
from ._sparse import ROW_BYTES, SparseRowStore, encoded_entry_bytes

__all__ = [
    "RowSpillPlan",
    "checked_destinations",
    "dense_sample_rows",
    "dense_spill_estimate",
    "row_spill_consumer",
    "row_spill_plan",
    "sample_rows",
    "spill_rows",
]

# What appending a block of rows to the row spill of a shuffle holds for each
# value of the block: the rows gathered by bucket, their values cast to the
# source dtype, a dense block's sparse copy, and the row and bucket numbers of
# each row, which a block holds no more of than values.
_SPILL_BYTES_PER_VALUE = 96
# The most rows that a spill writer keeps pending; more gives longer blocks.
_SPILL_BUFFER_BYTES = 1 << 30
# The least pending rows for which a shuffle's scan starts another process.
_SPILL_PROCESS_BUFFER_BYTES = 64 << 20
# Buckets of shuffled rows are sized to read in this much, or in a
# thirty-second of the budget when that is less, and split beyond twice it.
_SPILL_READ_BYTES = 256 << 20
# The entries of the first rows whose encoding estimates the spill's disk use.
_SPILL_SAMPLE_ENTRIES = 1 << 20


def checked_destinations(destinationRows: Any, n_rows: int) -> np.ndarray:
    """Return the destination row of each of ``n_rows`` rows as int64, checked to be a permutation.

    Raises:
        ValueError: If the rows are not one integer for each row, or not a
            permutation of the rows.
    """
    destinations = np.asarray(destinationRows)
    if destinations.shape != (n_rows,) or destinations.dtype.kind not in "iu":
        raise ValueError(
            f"destinationRows must hold one integer row for each of the "
            f"{n_rows} cells; found dtype {destinations.dtype} and "
            f"shape {destinations.shape}"
        )
    destinations = destinations.astype(np.int64, copy=False)
    if n_rows and (
        int(destinations.min()) < 0
        or not np.array_equal(
            np.bincount(destinations, minlength=n_rows),
            np.ones(n_rows, dtype=np.int64),
        )
    ):
        raise ValueError(
            "destinationRows must be a permutation of the rows, each row from 0 once"
        )
    return destinations


def sample_rows(pointers: np.ndarray) -> int:
    """Return the first rows of row ``pointers`` that hold about a million entries.

    Their encoding, as :func:`~cytearc.readers._sparse.encoded_entry_bytes`
    measures it, estimates what the spill takes on disk. At least one row,
    when there is one.
    """
    n_rows = int(pointers.size) - 1
    if n_rows <= 0:
        return 0
    first = int(
        np.searchsorted(pointers, pointers[0] + _SPILL_SAMPLE_ENTRIES, side="right")
    )
    return min(n_rows, max(1, first - 1))


def dense_sample_rows(n_rows: int, n_columns: int) -> int:
    """Return the first of ``n_rows`` dense rows of ``n_columns`` values that hold about a million values.

    At least one row, when there is one.
    """
    if n_rows <= 0:
        return 0
    return min(int(n_rows), max(1, _SPILL_SAMPLE_ENTRIES // max(1, int(n_columns))))


def dense_spill_estimate(first_rows: Any, n_rows: int) -> tuple[int, float | None]:
    """Estimate what ``n_rows`` dense rows like ``first_rows`` take in a shuffle's temporary files.

    A dense row is spilled as a sparse row, its zeros left out. Pass the
    first rows, of about a million values, as :func:`dense_sample_rows`
    counts them.

    Returns:
        The values that the rows hold beside zeros, at the share of them
        among the first rows, and the bytes that each takes on disk, as
        :func:`~cytearc.readers._sparse.encoded_entry_bytes` measures it on
        the first rows, or None without first rows.
    """
    sample = csr_matrix(first_rows)
    rows = int(sample.shape[0])
    if rows == 0:
        return 0, None
    return -(-int(sample.nnz) * int(n_rows) // rows), encoded_entry_bytes(sample)


@dataclass(frozen=True)
class RowSpillPlan:
    """The row bucket store of a shuffle, and what its scan and writers hold.

    The owner holds the store's row counts. Each writer holds a count and
    mark for every row of the store. A scan window's writer also holds the
    destinations of its own rows, the scan window and the rows it appends,
    and its pending rows, which take what is left of the window's share, up
    to 1 GiB. A writer fed batches of rows holds what appending a batch
    holds beside the batch, and its pending rows.
    """

    destinations: np.ndarray
    shape: tuple[int, int]
    dtype: np.dtype[Any]
    bucketRows: int
    readBytes: int
    entries: int | None
    longest: int
    maxBytes: int

    @property
    def ownerBytes(self) -> int:
        return (self.shape[0] + 1) * ROW_BYTES

    def _fixed_bytes(self, rows: int) -> int:
        """Return what a writer of ``rows`` rows holds beside its scan and buffer."""
        return SparseRowStore.writer_bytes(self.shape[0], self.bucketRows) + 8 * rows

    def _window_values(self, share: int, rows: int) -> int:
        available = share - self._fixed_bytes(rows)
        per_value = scan_window_bytes(1) + _SPILL_BYTES_PER_VALUE
        return max(
            max(1, self.longest),
            min(scan_window_values(self.longest), available // per_value),
        )

    def scan_bytes(self, share: int, start: int, stop: int) -> int:
        """Return the scan's budget for the window ``[start, stop)`` within ``share``."""
        return scan_window_bytes(self._window_values(share, stop - start))

    def buffer_bytes(self, share: int, start: int, stop: int) -> int:
        """Return the pending rows of the window ``[start, stop)``'s writer within ``share``."""
        values = self._window_values(share, stop - start)
        left = (
            share
            - self._fixed_bytes(stop - start)
            - values * (scan_window_bytes(1) + _SPILL_BYTES_PER_VALUE)
        )
        return int(max(0, min(_SPILL_BUFFER_BYTES, left)))

    def process_extra_bytes(self) -> int:
        """Return what a scan process holds beside its scan when it may start."""
        return (
            self._fixed_bytes(self.shape[0])
            + _SPILL_BYTES_PER_VALUE * scan_window_values(self.longest)
            + _SPILL_PROCESS_BUFFER_BYTES
        )

    def least_bytes(self) -> int:
        """Return the least budget that scans and spills the longest row in one process."""
        return (
            self.ownerBytes
            + self._fixed_bytes(self.shape[0])
            + max(1, self.longest) * (scan_window_bytes(1) + _SPILL_BYTES_PER_VALUE)
        )

    def batch_buffer_bytes(self, share: int, values: int) -> int:
        """Return the pending rows of a writer fed batches of up to ``values`` values.

        ``share`` is what the writer may hold beside the batches themselves.
        """
        left = share - self._fixed_bytes(0) - int(values) * _SPILL_BYTES_PER_VALUE
        return int(max(0, min(_SPILL_BUFFER_BYTES, left)))

    def batch_bytes(self, values: int) -> int:
        """Return what the owner and a writer hold to append a batch of ``values`` values.

        The writer's pending rows come on top of these.
        """
        return (
            self.ownerBytes
            + self._fixed_bytes(0)
            + int(values) * _SPILL_BYTES_PER_VALUE
        )

    def bucket_bytes(self) -> int:
        """Return the most that reading one bucket may hold: a quarter of the budget."""
        return max(2 * self.readBytes, self.maxBytes // 4)

    def new_store(
        self,
        *,
        operation: str,
        entry_bytes: float | None,
        temp_dir: str | Path | None,
        entries: int | None = None,
    ) -> SparseRowStore:
        """Create the store, its own part empty, after checking its disk.

        ``entry_bytes``, as :func:`~cytearc.readers._sparse.encoded_entry_bytes`
        measures them on the first rows, estimates the disk the rows take;
        None bounds each entry by its bytes before compression. ``entries``
        are the values that the rows hold beside zeros: the plan's own by
        default, and for dense rows, whose plan counts none, the estimate of
        :func:`dense_spill_estimate`.

        Raises:
            OSError: If the temporary disk lacks room for the rows.
        """
        return SparseRowStore(
            self.shape,
            self.dtype,
            operation=operation,
            bucket_rows=self.bucketRows,
            whole_rows=True,
            buffer_bytes=0,
            bucket_bytes=self.bucket_bytes(),
            split_bytes=2 * self.readBytes,
            entries=self.entries if entries is None else entries,
            entry_bytes=entry_bytes,
            sum_duplicates=False,
            temp_dir=temp_dir,
        )


def row_spill_plan(
    destinations: np.ndarray,
    shape: tuple[int, int],
    dtype: Any,
    *,
    entries: int | None,
    longest: int,
    maxBytes: int,
    least: Callable[[RowSpillPlan], int] | None = None,
) -> RowSpillPlan:
    """Plan the row bucket store of a shuffle and check that ``maxBytes`` holds it.

    Buckets are sized to read in 256 MiB, or a thirty-second of ``maxBytes``
    when that is less, from ``entries``, the stored values of the rows, or
    for dense rows of ``shape[1]`` values when None. ``longest`` is the most
    values of one row.

    Args:
        destinations: The destination row of each row, checked to be a
            permutation.
        shape: Rows and columns of the matrix.
        dtype: The dtype of the values that the rows hold.
        entries: The stored values of the rows, or None for dense rows.
        longest: The most values of one row.
        maxBytes: Memory available to the spill.
        least: The least budget of the spill under the plan; by default, a
            scan that spills the longest row in one process.

    Raises:
        MemoryBudgetError: If ``maxBytes`` holds less than the least budget.
    """
    itemsize = np.dtype(dtype).itemsize
    read_bytes = min(_SPILL_READ_BYTES, int(maxBytes) // 32)
    n_rows = int(shape[0])
    plan = RowSpillPlan(
        destinations=destinations,
        shape=(n_rows, int(shape[1])),
        dtype=np.dtype(dtype),
        bucketRows=SparseRowStore.whole_row_bucket_rows(
            n_rows,
            entries if entries is not None else n_rows * int(shape[1]),
            read_bytes,
            itemsize,
        ),
        readBytes=read_bytes,
        entries=entries,
        longest=int(longest),
        maxBytes=int(maxBytes),
    )
    needed = plan.least_bytes() if least is None else int(least(plan))
    if int(maxBytes) < needed:
        raise MemoryBudgetError(
            "Shuffling cells exceeds the memory limit. Reading the rows and "
            f"sending them to their buckets needs at least {needed} bytes.",
            needed_bytes=needed,
            budget_bytes=int(maxBytes),
            item="the row spill of a cell shuffle",
        )
    return plan


def row_spill_consumer(
    store: SparseRowStore, destinations: np.ndarray, *, dense: bool
) -> Callable[[int, Any], None]:
    """Return a consumer that appends each block of rows to ``store``.

    ``destinations`` holds the destination row of each row that the
    consumer receives, by the row's position among them. A dense block is
    appended as sparse rows, its zeros left out.
    """

    def consume(first: int, block: Any) -> None:
        rows = block.shape[0]
        store.append_rows(
            destinations[first : first + rows],
            csr_matrix(block) if dense else block,
        )

    return consume


def spill_rows[T](
    plan: RowSpillPlan,
    fill: Callable[[Callable[[int, Any], None]], T],
    *,
    operation: str,
    entry_bytes: float | None,
    buffer_bytes: int,
    dense: bool,
    temp_dir: str | Path | None,
    entries: int | None = None,
) -> tuple[SparseRowStore, T]:
    """Spill rows in this process into a new store and finalize it.

    ``fill`` receives a consumer, as :func:`row_spill_consumer` returns it,
    sends every row to it once, and returns what the caller keeps, such as
    the value ranges of a scan. A writer of its own part holds up to
    ``buffer_bytes`` of pending rows. ``entry_bytes`` and ``entries`` check
    the free disk, as :meth:`RowSpillPlan.new_store` describes. A failure
    removes the store.

    Returns:
        The finalized store and what ``fill`` returned.
    """
    store = plan.new_store(
        operation=operation,
        entry_bytes=entry_bytes,
        temp_dir=temp_dir,
        entries=entries,
    )
    try:
        writer = SparseRowStore.open(
            store.directory, part=1, buffer_bytes=int(buffer_bytes)
        )
        result = fill(row_spill_consumer(writer, plan.destinations, dense=dense))
        writer.finish()
        del writer
        store.finalize()
    except BaseException:
        store.close()
        raise
    return store, result
