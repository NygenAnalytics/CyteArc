"""UMI count histograms of the guides of a pooled CRISPR screen.

Guide calling fits a model to the UMI counts of each guide, separately in each
sample. A fit needs only how many cells hold each count, so one streamed pass
over the counts builds a histogram for every sample and guide, and the fits
read the histograms instead of the cells. Their cost then follows the number of
distinct counts, which grows slowly with the number of cells.

The histograms are exact, so a fit on a histogram equals the fit on every cell.
Counts up to :data:`DENSE_COUNTS`, which hold the ambient molecules that most
cells carry, go to a dense array of samples x guides x :data:`DENSE_COUNTS`
cells. The larger counts, mostly those of the cells that carry a guide, are
kept as exact (count, cells) pairs. A buffer of fixed size gathers them, and
they join the pairs whenever it fills, so the memory follows the distinct
counts that the histograms hold, which only the counts tell.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from cytearc.perturbation.guide_calls import checked_count_blocks
from cytearc.utils.arguments import integer_argument
from cytearc.utils.numba import njit

__all__ = [
    "DENSE_COUNTS",
    "MAX_COUNT",
    "GuideCountHistograms",
    "HistogramMemoryError",
    "checked_sample_codes",
    "guide_count_histograms",
    "histogram_bytes",
]

#: Counts from 1 to this value each get a bin in the dense part of a histogram.
DENSE_COUNTS = 32
#: The largest UMI count of one guide in one cell, the uint32 maximum.
MAX_COUNT = 0xFFFFFFFF

# A larger count is stored as one uint64 key: the histogram's position in the
# high 32 bits and the count, which never exceeds MAX_COUNT, in the low.
_COUNT_BITS = np.uint64(32)
_COUNT_MASK = np.uint64(MAX_COUNT)
# The buffer of larger counts holds two keys per cell and at most this many
# keys, 8 MiB, and never fewer than two rows of keys; the counts of a screen
# whose cells carry more fill it again.
_KEYS_PER_CELL = 2
_MOST_KEYS = 1 << 20
# The bytes of Python objects and of the sort's scratch that a call holds;
# about 4 kB in Python 3.13 with NumPy 2.5.
_CALL_BYTES = 1 << 14


@dataclass(frozen=True, slots=True)
class GuideCountHistograms:
    """How many cells of each sample hold each UMI count of each guide.

    The histogram of guide ``g`` in sample ``s`` is entry
    ``s * n_guides + g``: the slice ``offsets[entry]`` to ``offsets[entry + 1]``
    of ``counts`` and ``cells`` lists the guide's distinct nonzero UMI counts in
    ascending order and how many cells hold each. Cells with no UMI of the
    guide are not listed; :meth:`zero_cells` gives their number.

    Attributes:
        n_cells: The number of cells of each sample.
        n_guides: The number of guides.
        offsets: Where each histogram starts in ``counts`` and ``cells``, with
            one final entry for the end.
        counts: The distinct nonzero UMI counts of every histogram.
        cells: The number of cells that hold each count.
    """

    n_cells: np.ndarray
    n_guides: int
    offsets: np.ndarray
    counts: np.ndarray
    cells: np.ndarray

    @property
    def n_samples(self) -> int:
        """The number of samples."""
        return len(self.n_cells)

    def nonzero_cells(self) -> np.ndarray:
        """Return the cells with at least one UMI of each guide, samples x guides."""
        cumulative = np.zeros(len(self.cells) + 1, dtype=np.int64)
        np.cumsum(self.cells, out=cumulative[1:])
        totals: np.ndarray = (
            cumulative[self.offsets[1:]] - cumulative[self.offsets[:-1]]
        )
        return totals.reshape(self.n_samples, self.n_guides)

    def zero_cells(self) -> np.ndarray:
        """Return the cells with no UMI of each guide, samples x guides."""
        zeros: np.ndarray = self.n_cells[:, None] - self.nonzero_cells()
        return zeros

    def n_calls(self, thresholds: np.ndarray) -> int:
        """Return how many calls a table of smallest called counts makes.

        ``thresholds`` holds the smallest called count of each guide in each
        sample, samples x guides, 0 where a guide calls nothing; each cell
        whose count of a guide reaches it is one call. The count makes no
        array as long as the histograms.
        """
        limits = np.asarray(thresholds, dtype=np.int64).reshape(-1)
        return int(_called_cells(self.offsets, self.counts, self.cells, limits))

    def distinct_counts(self) -> np.ndarray:
        """Return the number of distinct nonzero counts of each guide, samples x guides."""
        return np.diff(self.offsets).reshape(self.n_samples, self.n_guides)

    def max_counts(self) -> np.ndarray:
        """Return the largest count of each guide, 0 without UMIs, samples x guides."""
        stops = self.offsets[1:]
        filled = stops > self.offsets[:-1]
        largest = np.zeros(len(stops), dtype=np.int64)
        largest[filled] = self.counts[stops[filled] - 1]
        return largest.reshape(self.n_samples, self.n_guides)


@njit(cache=True, nogil=True)
def _called_cells(
    offsets: np.ndarray, counts: np.ndarray, cells: np.ndarray, limits: np.ndarray
) -> int:
    """Return the cells whose count of each histogram reaches its limit, 0 for none.

    Each histogram lists its counts in ascending order, so the counts that
    reach the limit are the histogram's last ones.
    """
    total = 0
    for entry in range(limits.shape[0]):
        limit = limits[entry]
        if limit <= 0:
            continue
        stop = offsets[entry + 1]
        first = offsets[entry] + np.searchsorted(counts[offsets[entry] : stop], limit)
        for index in range(first, stop):
            total += cells[index]
    return total


def histogram_bytes(
    n_samples: int, n_guides: int, n_cells: int, *, n_distinct: int
) -> int:
    """Return the memory that :func:`guide_count_histograms` holds besides its blocks.

    The memory grows with ``n_distinct``, the distinct nonzero counts of all
    histograms together, the length of their ``counts``, which only reading
    the counts tells; with 0 it is the least memory of any call. It covers
    the dense array, the buffer that gathers the counts above
    :data:`DENSE_COUNTS`, two keys per cell and at most 8 MiB but at least two
    per guide, 16 KiB of Python objects and sorting, and 32 bytes per
    distinct count: a merge holds the merged larger counts beside the new
    ones, 16 bytes each, and assembling the histograms holds them beside the
    16 bytes of each count of the histograms. The sample codes, as int64,
    belong to the caller.

    Args:
        n_samples: The number of samples.
        n_guides: The number of guides.
        n_cells: The number of cells.
        n_distinct: The distinct nonzero counts of all histograms together.

    Returns:
        The most bytes held at once.
    """
    n_entries = n_samples * n_guides
    dense = n_entries * DENSE_COUNTS * np.dtype(np.uint32).itemsize
    buffer = 8 * _buffer_keys(n_cells, n_guides)
    # The offsets of the histograms and the cells of each sample.
    assembled = 8 * (n_entries + 1) + 16 * n_samples
    return dense + buffer + assembled + _CALL_BYTES + 32 * n_distinct


def _buffer_keys(n_cells: int, n_guides: int) -> int:
    """Return how many keys the buffer of larger counts holds."""
    return max(2 * n_guides, min(_KEYS_PER_CELL * n_cells + 64, _MOST_KEYS))


@njit(cache=True, nogil=True)
def _add_counts(
    block: np.ndarray,
    sample_codes: np.ndarray,
    n_guides: int,
    dense: np.ndarray,
    keys: np.ndarray,
    n_keys: int,
    first_row: int,
) -> tuple[int, int]:
    """Add the counts of one row block, from row ``first_row`` on.

    Counts up to the dense width are added to ``dense``, and the key of each
    larger count is written to ``keys`` after the ``n_keys`` that it holds.
    The pass ends before a row whose counts might not all fit in ``keys``.
    Returns the row where the pass ended, which is the block's number of rows
    once every row is added and -1 when a count does not fit in uint32, and
    the number of keys that ``keys`` then holds.
    """
    width = dense.shape[2]
    room = keys.shape[0]
    for row in range(first_row, block.shape[0]):
        if room - n_keys < block.shape[1]:
            return row, n_keys
        sample = sample_codes[row]
        base = sample * n_guides
        for guide in range(block.shape[1]):
            count = int(block[row, guide])
            if count == 0:
                continue
            # A uint64 count above the int64 range wraps to a negative value.
            if count < 0 or count > MAX_COUNT:
                return -1, n_keys
            if count <= width:
                dense[sample, guide, count - 1] += 1
                continue
            keys[n_keys] = (np.uint64(base + guide) << _COUNT_BITS) | (np.uint64(count))
            n_keys += 1
    return block.shape[0], n_keys


@njit(cache=True, nogil=True)
def _merged_size(keys: np.ndarray, run: np.ndarray) -> int:
    """Return how many distinct keys sorted distinct ``keys`` and a sorted ``run`` hold."""
    size = int(keys.shape[0])
    held = 0
    position = 0
    while position < run.shape[0]:
        key = run[position]
        while held < keys.shape[0] and keys[held] < key:
            held += 1
        if held == keys.shape[0] or keys[held] != key:
            size += 1
        while position < run.shape[0] and run[position] == key:
            position += 1
    return size


@njit(cache=True, nogil=True)
def _merge_run(
    keys: np.ndarray,
    cells: np.ndarray,
    run: np.ndarray,
    merged_keys: np.ndarray,
    merged_cells: np.ndarray,
) -> None:
    """Merge a sorted ``run`` of keys, one cell each, into distinct keys and their cells.

    ``keys`` are sorted and distinct, ``cells`` hold their cells, and the
    merged arrays have the length that :func:`_merged_size` returns.
    """
    held = 0
    position = 0
    written = 0
    while position < run.shape[0]:
        key = run[position]
        while held < keys.shape[0] and keys[held] < key:
            merged_keys[written] = keys[held]
            merged_cells[written] = cells[held]
            written += 1
            held += 1
        total = 0
        if held < keys.shape[0] and keys[held] == key:
            total = cells[held]
            held += 1
        while position < run.shape[0] and run[position] == key:
            total += 1
            position += 1
        merged_keys[written] = key
        merged_cells[written] = total
        written += 1
    while held < keys.shape[0]:
        merged_keys[written] = keys[held]
        merged_cells[written] = cells[held]
        written += 1
        held += 1


@njit(cache=True, nogil=True)
def _assemble(
    dense: np.ndarray,
    keys: np.ndarray,
    key_cells: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    cells: np.ndarray,
) -> None:
    """Write every histogram: its dense counts in ascending order, then its larger ones.

    The keys of the larger counts are sorted by histogram and count, and every
    one exceeds the dense range, so each histogram comes out in ascending order.
    """
    position = 0
    larger = 0
    for entry in range(dense.shape[0]):
        offsets[entry] = position
        for bin_index in range(dense.shape[1]):
            if dense[entry, bin_index] > 0:
                counts[position] = bin_index + 1
                cells[position] = dense[entry, bin_index]
                position += 1
        while larger < keys.shape[0] and np.int64(keys[larger] >> _COUNT_BITS) == entry:
            counts[position] = np.int64(keys[larger] & _COUNT_MASK)
            cells[position] = key_cells[larger]
            position += 1
            larger += 1
    offsets[dense.shape[0]] = position


class HistogramMemoryError(MemoryError):
    """The guide count histograms need more memory than they may hold.

    Attributes:
        needed: The bytes that the histograms need at the point where they
            stopped, within their memory limit.
        memory_bytes: The bytes that the histograms may hold.
        n_distinct: The distinct counts that they held then.
        rows_read: The rows of counts read when they stopped.
        n_rows: The rows of counts, one per cell.
    """

    def __init__(
        self,
        needed: int,
        memory_bytes: int,
        n_distinct: int,
        *,
        rows_read: int,
        n_rows: int,
    ) -> None:
        self.needed = needed
        self.memory_bytes = memory_bytes
        self.n_distinct = n_distinct
        self.rows_read = rows_read
        self.n_rows = n_rows
        remain = (
            f", and {n_rows - rows_read} of {n_rows} rows of counts remain to be "
            "read, so they may need more"
            if self.counts_remain
            else ""
        )
        super().__init__(
            f"The guide count histograms need about {needed} bytes for "
            f"{n_distinct} distinct counts, more than the {memory_bytes} bytes "
            f"they may hold{remain}"
        )

    @property
    def counts_remain(self) -> bool:
        """Whether counts remained to be read, so that the histograms may need more."""
        return self.rows_read < self.n_rows

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            _restore_histogram_memory_error,
            (
                type(self),
                self.needed,
                self.memory_bytes,
                self.n_distinct,
                self.rows_read,
                self.n_rows,
            ),
            self.__dict__,
        )


def _restore_histogram_memory_error(
    error_type: type[HistogramMemoryError],
    needed: int,
    memory_bytes: int,
    n_distinct: int,
    rows_read: int,
    n_rows: int,
) -> HistogramMemoryError:
    """Rebuild a pickled HistogramMemoryError, whose row counts are keywords."""
    return error_type(
        needed, memory_bytes, n_distinct, rows_read=rows_read, n_rows=n_rows
    )


def _require_memory(
    needed: int,
    memory_bytes: int | None,
    n_distinct: int,
    *,
    rows_read: int,
    n_rows: int,
) -> None:
    """Raise HistogramMemoryError when the histograms need more than ``memory_bytes``."""
    if memory_bytes is not None and needed > memory_bytes:
        raise HistogramMemoryError(
            needed, memory_bytes, n_distinct, rows_read=rows_read, n_rows=n_rows
        )


def _assembly_bytes(held: int, dense: np.ndarray, n_larger: int) -> int:
    """Return what assembling the histograms holds.

    ``held`` covers what the caller holds and the dense array; beside it,
    the ``n_larger`` merged larger counts and their cells, the offsets, the
    cells of each sample, and the counts and cells of the histograms.
    """
    n_samples, n_guides = int(dense.shape[0]), int(dense.shape[1])
    total = int(np.count_nonzero(dense)) + n_larger
    return (
        held
        + 16 * n_larger
        + 8 * (n_samples * n_guides + 1)
        + 16 * n_samples
        + 16 * total
    )


def _larger_counts(
    blocks: Iterable[np.ndarray],
    codes: np.ndarray,
    n_guides: int,
    dense: np.ndarray,
    held: int,
    memory_bytes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add every row block of counts to the histograms, and merge the larger counts.

    Counts up to the dense width are added to ``dense``. Returns the distinct
    keys of the larger counts, sorted, and the number of cells that hold
    each. ``held`` counts the bytes that the caller holds beside them; the
    buffer is released when this returns.
    """
    buffer = np.empty(_buffer_keys(len(codes), n_guides), dtype=np.uint64)
    keys = np.zeros(0, dtype=np.uint64)
    cells = np.zeros(0, dtype=np.int64)

    n_rows = len(codes)

    def merged(n_keys: int, rows_read: int) -> tuple[np.ndarray, np.ndarray]:
        run = buffer[:n_keys]
        run.sort()
        size = int(_merged_size(keys, run))
        needed = held + buffer.nbytes + keys.nbytes + cells.nbytes + 16 * size
        if rows_read == n_rows:
            # The last merge also checks the assembly that follows, so that
            # one refusal names what both need.
            needed = max(needed, _assembly_bytes(held, dense, size))
        _require_memory(needed, memory_bytes, size, rows_read=rows_read, n_rows=n_rows)
        merged_keys = np.empty(size, dtype=np.uint64)
        merged_cells = np.empty(size, dtype=np.int64)
        _merge_run(keys, cells, run, merged_keys, merged_cells)
        return merged_keys, merged_cells

    n_keys = 0
    for start, block in checked_count_blocks(blocks, n_cells=n_rows, n_guides=n_guides):
        block_codes = codes[start : start + block.shape[0]]
        row = 0
        while row < block.shape[0]:
            row, n_keys = _add_counts(
                block, block_codes, n_guides, dense, buffer, n_keys, row
            )
            if row < 0:
                raise ValueError(f"Guide UMI counts must be at most {MAX_COUNT}")
            if row < block.shape[0]:
                # The next row might not fit, so the buffer's keys join the
                # merged ones; the rows from the next on remain to be read.
                keys, cells = merged(n_keys, start + row)
                n_keys = 0
    if n_keys:
        keys, cells = merged(n_keys, n_rows)
    return keys, cells


def checked_sample_codes(sample_codes: object, n_samples: int) -> np.ndarray:
    """Return the sample position of each cell after checking its range.

    Args:
        sample_codes: The sample position of each cell.
        n_samples: The number of samples.

    Returns:
        The positions as int64.

    Raises:
        ValueError: If the codes are not a one-dimensional integer array with
            every value from 0 to ``n_samples - 1``.
    """
    codes = np.asarray(sample_codes)
    if codes.ndim != 1:
        raise ValueError("Sample codes must be one-dimensional, one per cell")
    if codes.dtype.kind not in "iu":
        raise ValueError(f"Sample codes must be integers, got {codes.dtype}")
    if len(codes) and (codes.min() < 0 or codes.max() >= n_samples):
        raise ValueError(f"Sample codes must lie in [0, {n_samples})")
    return codes.astype(np.int64, copy=False)


def guide_count_histograms(
    blocks: Iterable[np.ndarray],
    sample_codes: object,
    *,
    n_samples: int,
    n_guides: int,
    memory_bytes: int | None = None,
) -> GuideCountHistograms:
    """Count the cells that hold each UMI count of each guide, per sample.

    The memory grows with the distinct counts that the histograms hold, as
    :func:`histogram_bytes` counts it, and what the histograms would hold is
    checked against ``memory_bytes`` before each array is allocated.

    Args:
        blocks: Consecutive row blocks of the cells x guides counts, which
            together hold one row per entry of ``sample_codes``.
        sample_codes: The sample position of each cell, from 0 to
            ``n_samples - 1``. Without samples, every cell has code 0.
        n_samples: The number of samples.
        n_guides: The number of guides, the width of every block.
        memory_bytes: The most bytes that the histograms may hold besides the
            blocks and the caller's sample codes, or None for no limit.

    Returns:
        The exact histogram of every guide in every sample.

    Raises:
        TypeError: If a block holds counts that are not unsigned integers, or
            ``memory_bytes`` is not an integer.
        ValueError: If there is no sample or no guide, a sample code is out of
            range, a block has the wrong width, the blocks hold more or fewer
            rows than there are cells, a count exceeds the uint32 range, or
            ``memory_bytes`` is negative.
        HistogramMemoryError: If the histograms need more than
            ``memory_bytes``, a MemoryError that holds the bytes they need at
            that point and the rows of counts read by then.
    """
    if n_samples < 1 or n_guides < 1:
        raise ValueError("Histograms need at least one sample and one guide")
    if n_samples * n_guides >= 2**31:
        raise ValueError(
            f"{n_samples} samples of {n_guides} guides exceed 2**31 histograms"
        )
    if memory_bytes is not None:
        memory_bytes = integer_argument(memory_bytes, "memory_bytes", minimum=0)
    codes = checked_sample_codes(sample_codes, n_samples)
    # Codes that are not int64 already are held as an int64 copy.
    copied = 0 if codes is sample_codes else codes.nbytes
    n_entries = n_samples * n_guides
    _require_memory(
        copied + histogram_bytes(n_samples, n_guides, len(codes), n_distinct=0),
        memory_bytes,
        0,
        rows_read=0,
        n_rows=len(codes),
    )
    held = copied + _CALL_BYTES
    dense = np.zeros((n_samples, n_guides, DENSE_COUNTS), dtype=np.uint32)
    keys, key_cells = _larger_counts(
        blocks, codes, n_guides, dense, held + dense.nbytes, memory_bytes
    )

    flat = dense.reshape(n_entries, DENSE_COUNTS)
    total = int(np.count_nonzero(flat)) + len(keys)
    _require_memory(
        _assembly_bytes(held + dense.nbytes, dense, len(keys)),
        memory_bytes,
        total,
        rows_read=len(codes),
        n_rows=len(codes),
    )
    offsets = np.empty(n_entries + 1, dtype=np.int64)
    counts = np.empty(total, dtype=np.int64)
    cells = np.empty(total, dtype=np.int64)
    _assemble(flat, keys, key_cells, offsets, counts, cells)
    return GuideCountHistograms(
        n_cells=np.bincount(codes, minlength=n_samples).astype(np.int64),
        n_guides=n_guides,
        offsets=offsets,
        counts=counts,
        cells=cells,
    )
