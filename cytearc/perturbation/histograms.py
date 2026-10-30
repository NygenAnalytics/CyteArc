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
kept as exact (count, cells) pairs.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from cytearc.perturbation.guide_calls import checked_count_blocks
from cytearc.utils.numba import njit

__all__ = [
    "DENSE_COUNTS",
    "MAX_COUNT",
    "GuideCountHistograms",
    "checked_sample_codes",
    "guide_count_histograms",
    "histogram_bytes",
]

#: Counts from 1 to this value each get a bin in the dense part of a histogram.
DENSE_COUNTS = 32
#: The largest UMI count of one guide in one cell, the uint32 maximum.
MAX_COUNT = 0xFFFFFFFF

# A sparse count is stored as one uint64 key: the histogram's position in the
# high 32 bits and the count, which never exceeds MAX_COUNT, in the low.
_COUNT_BITS = np.uint64(32)
_COUNT_MASK = np.uint64(MAX_COUNT)
# Each block reserves room for this many sparse counts per row before a second
# pass is needed; cells rarely carry more than two guides above DENSE_COUNTS.
_SPARSE_PER_ROW = 2
# Sparse counts merge into distinct keys once this many wait to be merged.
_MERGE_ENTRIES = 1 << 22
# The distinct counts up to DENSE_COUNTS that histogram_bytes expects for one
# guide in one sample; ambient counts rarely reach ten.
_EXPECTED_DENSE_DISTINCT = 8


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

    def histogram(self, sample: int, guide: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the distinct nonzero counts of one guide in one sample.

        Args:
            sample: The position of the sample.
            guide: The position of the guide.

        Returns:
            The counts in ascending order and the number of cells holding each.
        """
        entry = sample * self.n_guides + guide
        start, stop = self.offsets[entry], self.offsets[entry + 1]
        return self.counts[start:stop], self.cells[start:stop]

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


def histogram_bytes(n_samples: int, n_guides: int, n_cells: int) -> int:
    """Estimate the memory that :func:`guide_count_histograms` holds besides its blocks.

    The estimate counts the dense array, two counts above :data:`DENSE_COUNTS`
    per cell at 16 bytes each, which a merge holds three times over, and the
    histograms it returns, with eight distinct counts up to
    :data:`DENSE_COUNTS` for each guide in each sample. Each block also
    reserves 16 bytes per row while it is counted.

    Args:
        n_samples: The number of samples.
        n_guides: The number of guides.
        n_cells: The number of cells.

    Returns:
        The estimated peak in bytes.
    """
    n_entries = n_samples * n_guides
    dense = n_entries * DENSE_COUNTS * np.dtype(np.uint32).itemsize
    sparse = _SPARSE_PER_ROW * n_cells
    merging = 16 * sparse + 32 * min(sparse, _MERGE_ENTRIES)
    result = 8 * (n_entries + 1) + 16 * (n_entries * _EXPECTED_DENSE_DISTINCT + sparse)
    return dense + merging + result


@njit(cache=True, nogil=True)
def _add_block(
    block: np.ndarray,
    sample_codes: np.ndarray,
    n_guides: int,
    dense: np.ndarray,
    sparse_keys: np.ndarray,
) -> int:
    """Add one row block of counts to the histograms.

    Counts up to the dense width are added to ``dense``. The keys of larger
    counts are written to ``sparse_keys`` while it has room. Returns the number
    of larger counts, which may exceed the room, or -1 when a count does not
    fit in uint32.
    """
    width = dense.shape[2]
    room = sparse_keys.shape[0]
    n_sparse = 0
    for row in range(block.shape[0]):
        sample = sample_codes[row]
        base = sample * n_guides
        for guide in range(block.shape[1]):
            count = int(block[row, guide])
            if count == 0:
                continue
            # A uint64 count above the int64 range wraps to a negative value.
            if count < 0 or count > MAX_COUNT:
                return -1
            if count <= width:
                dense[sample, guide, count - 1] += 1
                continue
            if n_sparse < room:
                sparse_keys[n_sparse] = (np.uint64(base + guide) << _COUNT_BITS) | (
                    np.uint64(count)
                )
            n_sparse += 1
    return n_sparse


@njit(cache=True, nogil=True)
def _assemble(
    dense: np.ndarray,
    sparse_entries: np.ndarray,
    sparse_counts: np.ndarray,
    sparse_cells: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    cells: np.ndarray,
) -> None:
    """Write every histogram: its dense counts in ascending order, then its sparse ones.

    The sparse counts are sorted by histogram and count, and every one exceeds
    the dense range, so each histogram comes out in ascending order.
    """
    position = 0
    sparse = 0
    for entry in range(dense.shape[0]):
        offsets[entry] = position
        for bin_index in range(dense.shape[1]):
            if dense[entry, bin_index] > 0:
                counts[position] = bin_index + 1
                cells[position] = dense[entry, bin_index]
                position += 1
        while sparse < sparse_entries.shape[0] and sparse_entries[sparse] == entry:
            counts[position] = sparse_counts[sparse]
            cells[position] = sparse_cells[sparse]
            position += 1
            sparse += 1
    offsets[dense.shape[0]] = position


def _merged(
    keys: list[np.ndarray], cells: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Merge keyed cell counts into one sorted array of distinct keys."""
    all_keys = np.concatenate(keys)
    all_cells = np.concatenate(cells)
    if not len(all_keys):
        return all_keys, all_cells
    order = np.argsort(all_keys, kind="stable")
    all_keys = all_keys[order]
    all_cells = all_cells[order]
    starts = np.flatnonzero(np.r_[True, all_keys[1:] != all_keys[:-1]])
    return all_keys[starts], np.add.reduceat(all_cells, starts)


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
) -> GuideCountHistograms:
    """Count the cells that hold each UMI count of each guide, per sample.

    Args:
        blocks: Consecutive row blocks of the cells x guides counts, which
            together hold one row per entry of ``sample_codes``.
        sample_codes: The sample position of each cell, from 0 to
            ``n_samples - 1``. Without samples, every cell has code 0.
        n_samples: The number of samples.
        n_guides: The number of guides, the width of every block.

    Returns:
        The exact histogram of every guide in every sample.

    Raises:
        TypeError: If a block holds counts that are not unsigned integers.
        ValueError: If there is no sample or no guide, a sample code is out of
            range, a block has the wrong width, the blocks hold more or fewer
            rows than there are cells, or a count exceeds the uint32 range.
    """
    if n_samples < 1 or n_guides < 1:
        raise ValueError("Histograms need at least one sample and one guide")
    if n_samples * n_guides >= 2**31:
        raise ValueError(
            f"{n_samples} samples of {n_guides} guides exceed 2**31 histograms"
        )
    codes = checked_sample_codes(sample_codes, n_samples)
    n_cells = len(codes)
    dense = np.zeros((n_samples, n_guides, DENSE_COUNTS), dtype=np.uint32)
    merged_keys = np.zeros(0, dtype=np.uint64)
    merged_cells = np.zeros(0, dtype=np.int64)
    pending_keys: list[np.ndarray] = []
    pending_cells: list[np.ndarray] = []
    n_pending = 0
    for start, block in checked_count_blocks(
        blocks, n_cells=n_cells, n_guides=n_guides
    ):
        stop = start + block.shape[0]
        room = np.empty(_SPARSE_PER_ROW * block.shape[0] + 64, dtype=np.uint64)
        n_sparse = _add_block(block, codes[start:stop], n_guides, dense, room)
        if n_sparse < 0:
            raise ValueError(f"Guide UMI counts must be at most {MAX_COUNT}")
        if n_sparse > len(room):
            rows, guides = np.nonzero(block > DENSE_COUNTS)
            groups = (codes[start + rows] * n_guides + guides).astype(np.uint64)
            values = block[rows, guides].astype(np.uint64)
            keys = (groups << _COUNT_BITS) | values
        else:
            keys = room[:n_sparse]
        if len(keys):
            distinct, multiplicity = np.unique(keys, return_counts=True)
            pending_keys.append(distinct)
            pending_cells.append(multiplicity.astype(np.int64))
            n_pending += len(distinct)
        if n_pending > _MERGE_ENTRIES:
            merged_keys, merged_cells = _merged(
                [merged_keys, *pending_keys], [merged_cells, *pending_cells]
            )
            pending_keys, pending_cells, n_pending = [], [], 0
    merged_keys, merged_cells = _merged(
        [merged_keys, *pending_keys], [merged_cells, *pending_cells]
    )

    n_entries = n_samples * n_guides
    flat = dense.reshape(n_entries, DENSE_COUNTS)
    total = int(np.count_nonzero(flat)) + len(merged_keys)
    offsets = np.empty(n_entries + 1, dtype=np.int64)
    counts = np.empty(total, dtype=np.int64)
    cells = np.empty(total, dtype=np.int64)
    _assemble(
        flat,
        (merged_keys >> _COUNT_BITS).astype(np.int64),
        (merged_keys & _COUNT_MASK).astype(np.int64),
        merged_cells,
        offsets,
        counts,
        cells,
    )
    return GuideCountHistograms(
        n_cells=np.bincount(codes, minlength=n_samples).astype(np.int64),
        n_guides=n_guides,
        offsets=offsets,
        counts=counts,
        cells=cells,
    )
