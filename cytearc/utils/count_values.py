"""Value ranges of count matrices, which decide their storage dtype.

Readers and writers summarize the canonical (duplicate-summed) values of a
count matrix into a :class:`CountValueRange`, one for each group of features
that an import stores as one assay;
:func:`cytearc.storage.count_dtype.count_storage_dtype` turns a range into the
storage dtype of its counts. Count matrices hold finite values, so the scans
read every value and reject NaN and infinity.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike

from .arrays import canonicalize_sparse
from .memory_budget import MemoryBudgetError

# A scan window holds its values and indices, a 64-bit copy of narrow integers,
# the sort and sum buffers of its canonical values, the values of one group,
# and the integrality checks, within 64 bytes per value. A block of up to one
# pointer per window value adds 8 more.
_SCAN_BYTES_PER_VALUE = 72
_SCAN_WINDOW_VALUES = 1 << 20
# A matrix with fewer stored values than this is scanned in one process.
# Starting scan processes takes about half a second. One process scans this
# many gzip-compressed values in 2 to 3 s, which four processes cut to about
# 1 s, and uncompressed ones in about 0.4 s, which processes cannot improve.
_SCAN_PROCESS_VALUES = 1 << 26


def is_real_count_dtype(dtype: DTypeLike) -> bool:
    """Return whether ``dtype`` holds real count values."""
    return np.dtype(dtype).kind in "biuf"


@dataclass(slots=True)
class CountValueRange:
    """Running summary of canonical count values.

    A summary does not depend on how the values were split into blocks, nor
    on their order, so the summaries of disjoint parts of a matrix merge into
    the summary of the whole.

    Attributes:
        integral: Whether every value seen is a non-negative integer.
        maximum: The largest value seen while ``integral`` holds, and 0 once
            it does not.
        negative: Whether any value seen is negative.
        largest: The largest value seen, which bounds the values that undoing
            a log gives.
    """

    integral: bool = True
    maximum: int = 0
    negative: bool = False
    largest: float = field(default=0.0, compare=False)

    def update(self, values: Any) -> None:
        """Add a block of canonical values, in any shape.

        A fractional or negative value settles ``integral``; later blocks are
        checked only for NaN, infinity, and negative values.

        Raises:
            TypeError: If the values are not real numbers.
            ValueError: If a value is NaN or infinite.
        """
        block = np.asarray(values)
        if block.size == 0:
            return
        if not is_real_count_dtype(block.dtype):
            raise TypeError(f"Count values must be real numbers, not {block.dtype}")
        kind = block.dtype.kind
        if kind == "f" and not np.isfinite(block).all():
            raise ValueError(
                "Count matrices must hold finite values; found NaN or infinity"
            )
        self.largest = max(self.largest, float(block.max()))
        if not self.integral:
            if kind in "if" and not self.negative and block.min() < 0:
                self.negative = True
            return
        if kind in "if" and block.min() < 0:
            self.integral = False
            self.negative = True
            self.maximum = 0
            return
        if kind == "f" and not np.equal(block, np.trunc(block)).all():
            self.integral = False
            self.maximum = 0
            return
        self.maximum = max(self.maximum, int(block.max()))

    def merge(self, other: "CountValueRange") -> None:
        """Add the values that ``other`` summarized, in any order."""
        self.integral = self.integral and other.integral
        self.maximum = max(self.maximum, other.maximum) if self.integral else 0
        self.negative = self.negative or other.negative
        self.largest = max(self.largest, other.largest)


@dataclass(slots=True)
class IndexOrder:
    """Running record of the index order of the compressed vectors scanned.

    Attributes:
        canonical: Whether every vector seen holds its indices sorted without
            repeats, as SciPy's ``has_canonical_format`` reports. Row batches
            of a CSR matrix whose rows all are need no sorting or summing.
    """

    canonical: bool = True


@dataclass(slots=True)
class CountScan:
    """What a scan of the compressed vectors ``[start, stop)`` of a matrix found.

    The vectors are the rows of a dense or CSR matrix and the columns of a
    CSC matrix.

    Attributes:
        start: The first vector scanned.
        stop: The vector after the last one scanned.
        valueRanges: The range of each group of features.
        rowNonzeros: The number of nonzero values of each row of a dense
            matrix, or None for a sparse one.
        canonical: Whether every vector holds its indices sorted without
            repeats; True for a dense matrix.
    """

    start: int
    stop: int
    valueRanges: list[CountValueRange]
    rowNonzeros: np.ndarray | None = None
    canonical: bool = True


def merge_count_scans(scans: list[CountScan]) -> CountScan:
    """Return the scan of the vectors that the scans of disjoint windows cover.

    Ranges merge exactly in any order, the index order of every vector
    holds when it holds in each window, and the row counts of dense windows
    take their positions, so the result equals one scan of all the vectors.

    Raises:
        ValueError: If the windows do not follow one another without gaps,
            or record ranges of different groups.
    """
    ordered = sorted(scans, key=lambda scan: scan.start)
    for before, after in zip(ordered, ordered[1:]):
        if before.stop != after.start:
            raise ValueError(
                f"Scanned windows must follow one another; {before.start}:"
                f"{before.stop} is followed by {after.start}:{after.stop}"
            )
    groups = {len(scan.valueRanges) for scan in ordered}
    if len(groups) != 1:
        raise ValueError("Scanned windows record ranges of different groups")
    value_ranges = [CountValueRange() for _ in range(groups.pop())]
    for scan in ordered:
        for value_range, window_range in zip(
            value_ranges, scan.valueRanges, strict=True
        ):
            value_range.merge(window_range)
    counts = [scan.rowNonzeros for scan in ordered]
    return CountScan(
        start=ordered[0].start,
        stop=ordered[-1].stop,
        valueRanges=value_ranges,
        rowNonzeros=(
            None
            if any(count is None for count in counts)
            else np.concatenate(cast(list[np.ndarray], counts))
        ),
        canonical=all(scan.canonical for scan in ordered),
    )


def scan_processes(values: int, longest: int, maxBytes: int, processes: int) -> int:
    """Return how many processes scan a matrix, each over its own vectors.

    A matrix of fewer than about 67 million stored values
    (``_SCAN_PROCESS_VALUES``) is scanned in one process, faster than
    starting more would let it finish. Otherwise at most ``processes``
    processes share ``maxBytes``, each with room for a whole scan window and
    for the ``longest`` vector, so that each reads as one scan with all of
    ``maxBytes`` would.

    Args:
        values: Stored values of the matrix.
        longest: Values of its longest compressed vector.
        maxBytes: Memory available to the scan.
        processes: Most processes that may scan.
    """
    if int(processes) <= 1 or int(values) < _SCAN_PROCESS_VALUES:
        return 1
    process_bytes = _SCAN_BYTES_PER_VALUE * max(_SCAN_WINDOW_VALUES, int(longest))
    return max(1, min(int(processes), max(0, int(maxBytes)) // process_bytes))


def value_windows(pointers: np.ndarray, parts: int) -> list[tuple[int, int]]:
    """Split compressed vectors into at most ``parts`` windows of about equal values.

    Args:
        pointers: Pointers of the vectors into their values, one more than
            there are vectors.
        parts: Most windows.

    Returns:
        Windows ``(start, stop)`` of whole vectors that follow one another
        and cover every vector, each holding at least one.
    """
    n_vectors = len(pointers) - 1
    if n_vectors < 1:
        return []
    first, last = int(pointers[0]), int(pointers[-1])
    targets = [first + (last - first) * part // parts for part in range(1, parts)]
    cuts = np.searchsorted(pointers, targets, side="left").clip(1, n_vectors - 1)
    bounds = np.unique(np.concatenate(([0], cuts, [n_vectors]))).tolist()
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _summable(values: np.ndarray) -> np.ndarray:
    """Return values in the dtype in which their duplicates are summed.

    Narrow integers widen to 64 bits so that sums cannot wrap. float16, which
    SciPy sparse matrices cannot hold, is read as float32, as readers do.
    """
    kind = values.dtype.kind
    if kind in "bu":
        return values.astype(np.uint64, copy=False)
    if kind == "i":
        return values.astype(np.int64, copy=False)
    if kind == "f":
        dtype = values.dtype.newbyteorder("=")
        return values.astype(np.float32 if dtype == np.float16 else dtype, copy=False)
    raise TypeError(f"Count values must be real numbers, not {values.dtype}")


def _canonical(block: Any) -> Any:
    """Return a CSR window with its duplicate coordinates summed as writers sum them."""
    if block.dtype.kind == "f":
        # Float sums depend on the order of their terms, so they are summed
        # by the writers' function, which leaves the caller's arrays unchanged.
        return canonicalize_sparse(block.tocoo(copy=False)).tocsr()
    # Integer sums are exact in 64 bits. SciPy sorts and sums in place, and
    # the window may be a view of the caller's arrays, so it works on a copy.
    block = block.copy()
    block.sum_duplicates()
    return block


def new_count_ranges(groups: np.ndarray | None) -> list[CountValueRange]:
    """Return one range that has seen no value for each group in ``groups``.

    Groups are numbered from zero; None stands for one group.
    """
    count = 1 if groups is None else int(np.max(groups, initial=0)) + 1
    return [CountValueRange() for _ in range(count)]


def compressed_count_ranges(
    indptr: Any,
    indices: Any,
    data: Any,
    *,
    minorSize: int,
    maxBytes: int,
    groups: np.ndarray | None = None,
    groupAxis: int = 1,
    valueRanges: list[CountValueRange] | None = None,
    indexOrder: IndexOrder | None = None,
) -> list[CountValueRange]:
    """Return the ranges of the canonical values of a compressed sparse matrix.

    The arrays are those of a CSR or CSC matrix, or objects that slice like
    them, such as HDF5 datasets; they are not modified. The scan reads every
    value, in windows of whole compressed vectors, and sums duplicate
    coordinates as writers store them: integers exactly in 64 bits and floats
    with :func:`~cytearc.utils.arrays.canonicalize_sparse`, in source order. The
    ranges therefore depend neither on the index order nor on ``maxBytes``.

    Args:
        indptr: Compressed vector pointers into ``indices`` and ``data``. A
            slice of a matrix's pointers scans only those vectors.
        indices: Minor-axis index of each stored value.
        data: Stored values.
        minorSize: Length of the minor axis.
        maxBytes: Memory available to the scan. A window holds at most about
            ``maxBytes // 72`` values.
        groups: Group, numbered from zero, of each minor index when
            ``groupAxis`` is 1, or of each compressed vector of ``indptr``
            when it is 0. None puts every value in one group.
        groupAxis: The axis that ``groups`` indexes.
        valueRanges: The range of each group that the scan extends, so that a
            matrix can be scanned in parts. None starts new ranges.
        indexOrder: The record in which the scan clears ``canonical`` when a
            vector it reads holds unsorted or repeated indices. None records
            nothing.

    Returns:
        The range of each group.

    Raises:
        MemoryError: If one compressed vector holds more values than fit in
            ``maxBytes``.
        ValueError: If a value is NaN or infinite.
    """
    from scipy.sparse import csr_matrix

    value_ranges = new_count_ranges(groups) if valueRanges is None else valueRanges
    limit = max(0, int(maxBytes)) // _SCAN_BYTES_PER_VALUE
    window = min(_SCAN_WINDOW_VALUES, limit)
    n_vectors = int(indptr.shape[0]) - 1
    start = 0
    while start < n_vectors:
        # Each pointer is read once; windows of whole vectors walk the block.
        pointers = np.asarray(
            indptr[start : min(n_vectors, start + max(1, window)) + 1],
            dtype=np.int64,
        )
        position = 0
        while position < len(pointers) - 1:
            target = pointers[position] + window
            end = int(np.searchsorted(pointers, target, side="right")) - 1
            if end <= position:
                # A vector longer than a window is read whole when it fits.
                length = int(pointers[position + 1] - pointers[position])
                if length > limit:
                    # The longest vector whose pointers are read names the
                    # budget that scans them all, not this one alone.
                    length = int(np.diff(pointers[position:]).max())
                    raise MemoryBudgetError(
                        f"One compressed vector holds {length} values; scanning "
                        f"it needs about {length * _SCAN_BYTES_PER_VALUE} bytes, "
                        f"more than the {max(0, int(maxBytes))}-byte limit. "
                        "Increase mem_budget.",
                        needed_bytes=length * _SCAN_BYTES_PER_VALUE,
                        budget_bytes=max(0, int(maxBytes)),
                        item="scanning one compressed vector",
                    )
                end = position + 1
            first, last = int(pointers[position]), int(pointers[end])
            block = csr_matrix(
                (
                    _summable(np.asarray(data[first:last])),
                    np.asarray(indices[first:last]),
                    pointers[position : end + 1] - first,
                ),
                shape=(end - position, int(minorSize)),
            )
            if not block.has_canonical_format:
                if indexOrder is not None:
                    indexOrder.canonical = False
                block = _canonical(block)
            if groups is None:
                value_ranges[0].update(block.data)
            else:
                codes = (
                    groups[block.indices]
                    if groupAxis == 1
                    else np.repeat(
                        groups[start + position : start + end], np.diff(block.indptr)
                    )
                )
                for code, value_range in enumerate(value_ranges):
                    value_range.update(block.data[codes == code])
            position = end
        start += position
    return value_ranges


def dense_count_ranges(
    read: Callable[[int, int], np.ndarray],
    nRows: int,
    nColumns: int,
    *,
    maxBytes: int,
    groups: np.ndarray | None = None,
    rowNonzeros: np.ndarray | None = None,
) -> list[CountValueRange]:
    """Return the ranges of the values of a dense matrix read in row blocks.

    The scan reads every row.

    Args:
        read: Returns rows ``[start, stop)`` as a NumPy array.
        nRows: Number of rows.
        nColumns: Number of columns.
        maxBytes: Memory available to the scan.
        groups: Group, numbered from zero, of each column. None puts every
            value in one group.
        rowNonzeros: An array of ``nRows`` integers in which the scan records
            the number of nonzero values of each row, from which an import
            plans its sparse batches. None records nothing.

    Returns:
        The range of each group.

    Raises:
        MemoryError: If one row does not fit in ``maxBytes``.
        ValueError: If a value is NaN or infinite, or ``rowNonzeros`` does
            not hold one integer for each row.
    """
    if rowNonzeros is not None and (
        rowNonzeros.shape != (int(nRows),) or rowNonzeros.dtype.kind not in "iu"
    ):
        raise ValueError(
            f"rowNonzeros must hold one integer for each of the {int(nRows)} "
            f"rows; got shape {rowNonzeros.shape} and dtype {rowNonzeros.dtype}"
        )
    row_values = max(1, int(nColumns))
    rows = min(
        max(1, _SCAN_WINDOW_VALUES // row_values),
        max(0, int(maxBytes)) // (_SCAN_BYTES_PER_VALUE * row_values),
    )
    value_ranges = new_count_ranges(groups)
    if int(nRows) and rows < 1:
        raise MemoryBudgetError(
            f"One dense row holds {row_values} values; scanning it needs about "
            f"{row_values * _SCAN_BYTES_PER_VALUE} bytes, more than the "
            f"{max(0, int(maxBytes))}-byte limit. Increase mem_budget.",
            needed_bytes=row_values * _SCAN_BYTES_PER_VALUE,
            budget_bytes=max(0, int(maxBytes)),
            item="scanning one dense row",
        )
    columns = (
        None
        if groups is None
        else [np.flatnonzero(groups == code) for code in range(len(value_ranges))]
    )
    for start in range(0, int(nRows), max(1, rows)):
        stop = min(int(nRows), start + rows)
        block = read(start, stop)
        if rowNonzeros is not None:
            rowNonzeros[start:stop] = np.count_nonzero(block, axis=1)
        if columns is None:
            value_ranges[0].update(block)
        else:
            for value_range, selected in zip(value_ranges, columns, strict=True):
                value_range.update(block[:, selected])
    return value_ranges


def dense_row_nonzeros(
    read: Callable[[int, int], np.ndarray], nRows: int, nColumns: int
) -> np.ndarray:
    """Return the number of nonzero values of each row of a dense matrix.

    The count reads every row, in blocks of about a million values, or one
    row at a time when a row holds more. :func:`dense_count_ranges` records
    the same counts in its scan; this reads them for a matrix not scanned.

    Args:
        read: Returns rows ``[start, stop)`` as a NumPy array.
        nRows: Number of rows.
        nColumns: Number of columns.

    Returns:
        The count of each row, as int64.
    """
    rows = max(1, _SCAN_WINDOW_VALUES // max(1, int(nColumns)))
    counts = np.empty(int(nRows), dtype=np.int64)
    for start in range(0, int(nRows), rows):
        stop = min(int(nRows), start + rows)
        counts[start:stop] = np.count_nonzero(read(start, stop), axis=1)
    return counts
