"""Guide calls of pooled single-cell CRISPR screens.

A pooled CRISPR screen gives each cell one or more guide RNAs, and each guide
leads Cas9, or a Cas9 fused to a repressor or an activator, to one target gene.
The guides are captured beside the transcriptome, so each cell has a count of
UMIs for every guide of the library. Every cell also holds a few ambient guide
molecules, so a guide is called in a cell only when a calling method, such as
Cell Ranger's or CyteArc's own, decides that the cell carries it.

These functions turn calls into the arrays of a ``guide_calls`` artifact and
summarise each cell's guide counts. They read count blocks that a caller
streams and never touch storage.

- A cell with one called guide is labelled with the guide's ID, a cell with
  none with ``"Negative"``, and a cell with two or more with ``"Multiple"``.
- A cell's called guides are ordered by UMIs, most first, and guides with equal
  UMIs by their position in the assay, so the order never depends on the
  method that made the calls.
- The top guide of a cell is the guide with the most UMIs, called or not, and
  the guide that comes first in the assay among guides with equal UMIs.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "GUIDE_ASSAY_TYPE",
    "MULTIPLE",
    "NEGATIVE",
    "RESERVED_LABELS",
    "GuideCallArrays",
    "GuideCalls",
    "GuideCountSummary",
    "checked_count_blocks",
    "guide_call_arrays",
    "guide_calls_frames",
    "summarize_guide_counts",
    "validated_guide_ids",
]

#: The declared type of an assay of guide counts, which guides are called from.
GUIDE_ASSAY_TYPE = "CRISPR"
#: The label of a cell with no called guide.
NEGATIVE = "Negative"
#: The label of a cell with two or more called guides.
MULTIPLE = "Multiple"
#: Labels that no guide ID may take, since they label cells without one guide.
RESERVED_LABELS = frozenset({NEGATIVE, MULTIPLE})

# Called UMIs are stored as uint32, which holds any count of one guide in one
# cell that a sequencing run produces.
_UMI_DTYPE = np.dtype(np.uint32)


def validated_guide_ids(guide_ids: Any) -> np.ndarray:
    """Return guide IDs as text after checking that they can label cells.

    Args:
        guide_ids: The ID of each guide, in assay order.

    Returns:
        The IDs as a one-dimensional array of Python strings.

    Raises:
        ValueError: If there is no guide, an ID is empty or repeated, or an ID
            equals a label reserved for cells without one guide.
    """
    ids = np.asarray(guide_ids, dtype=object).reshape(-1)
    if not len(ids):
        raise ValueError("A guide assay needs at least one guide")
    text = np.asarray([str(value) for value in ids], dtype=object)
    blank = [index for index, value in enumerate(text) if not value.strip()]
    if blank:
        raise ValueError(
            f"Guide IDs must not be empty, but {len(blank)} are, the first at "
            f"position {blank[0]}"
        )
    values, counts = np.unique(text.astype(str), return_counts=True)
    repeated = values[counts > 1].tolist()
    if repeated:
        raise ValueError(
            "Guide IDs must be unique, but these repeat: "
            + ", ".join(repr(value) for value in repeated[:5])
        )
    reserved = sorted(RESERVED_LABELS.intersection(values.tolist()))
    if reserved:
        raise ValueError(
            "Guide IDs conflict with labels reserved for cells without one guide: "
            + ", ".join(repr(value) for value in reserved)
        )
    return text


@dataclass(frozen=True, slots=True)
class GuideCountSummary:
    """The top guide and the guide UMIs of each cell.

    Attributes:
        top_code: The position of each cell's top guide in the assay, or -1 for
            a cell with no guide UMIs.
        top_umi: The UMIs of each cell's top guide, 0 for a cell with none.
        total_umi: All guide UMIs of each cell.
    """

    top_code: np.ndarray
    top_umi: np.ndarray
    total_umi: np.ndarray

    @property
    def top_fraction(self) -> np.ndarray:
        """The top guide's share of each cell's guide UMIs, NaN without UMIs."""
        fraction = np.full(len(self.total_umi), np.nan, dtype=np.float64)
        counted = self.total_umi > 0
        fraction[counted] = self.top_umi[counted] / self.total_umi[counted]
        return fraction


def checked_count_blocks(
    blocks: Iterable[np.ndarray],
    *,
    n_cells: int,
    n_guides: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield each row block of guide counts with the position of its first row.

    Args:
        blocks: Consecutive row blocks of the cells x guides counts, which
            together hold ``n_cells`` rows.
        n_cells: The number of cells.
        n_guides: The number of guides, the width of every block.

    Yields:
        The position of the block's first cell and the block as an array.

    Raises:
        TypeError: If a block holds counts that are not unsigned integers.
        ValueError: If a block has the wrong width, or the blocks hold more or
            fewer than ``n_cells`` rows.
    """
    start = 0
    for raw in blocks:
        block = np.asarray(raw)
        if block.ndim != 2 or block.shape[1] != n_guides:
            raise ValueError(
                f"Guide count blocks must have {n_guides} columns, got shape "
                f"{block.shape}"
            )
        if block.dtype.kind != "u":
            raise TypeError(
                f"Guide counts must be unsigned integers, got {block.dtype} counts"
            )
        stop = start + block.shape[0]
        if stop > n_cells:
            raise ValueError(f"The guide count blocks hold more than {n_cells} rows")
        yield start, block
        start = stop
    if start != n_cells:
        raise ValueError(
            f"The guide count blocks hold {start} rows, but there are {n_cells} cells"
        )


def summarize_guide_counts(
    blocks: Iterable[np.ndarray],
    n_cells: int,
    n_guides: int,
) -> GuideCountSummary:
    """Find each cell's top guide and total guide UMIs from streamed counts.

    Args:
        blocks: Consecutive row blocks of the cells x guides counts, which
            together hold ``n_cells`` rows.
        n_cells: The number of cells.
        n_guides: The number of guides, the width of every block.

    Returns:
        The top guide, its UMIs, and the total guide UMIs of each cell.

    Raises:
        TypeError: If a block holds counts that are not unsigned integers.
        ValueError: If a block has the wrong width, or the blocks hold more or
            fewer than ``n_cells`` rows.
    """
    top_code = np.full(n_cells, -1, dtype=np.int32)
    top_umi = np.zeros(n_cells, dtype=_UMI_DTYPE)
    total_umi = np.zeros(n_cells, dtype=np.uint64)
    for start, block in checked_count_blocks(
        blocks, n_cells=n_cells, n_guides=n_guides
    ):
        stop = start + block.shape[0]
        if block.shape[0]:
            totals = block.sum(axis=1, dtype=np.uint64)
            # argmax returns the first maximum, the lowest guide position.
            codes = np.argmax(block, axis=1)
            tops = block[np.arange(block.shape[0]), codes]
            counted = totals > 0
            top_code[start:stop] = np.where(counted, codes, -1)
            top_umi[start:stop] = np.where(counted, tops, 0)
            total_umi[start:stop] = totals
    return GuideCountSummary(top_code=top_code, top_umi=top_umi, total_umi=total_umi)


@dataclass(frozen=True, slots=True)
class GuideCallArrays:
    """The called guides of each cell, padded to one width.

    Attributes:
        values: Each cell's label: the ID of its one called guide,
            ``"Negative"``, or ``"Multiple"``.
        n_guides: The number of called guides of each cell.
        guide_codes: The positions of each cell's called guides in the assay,
            most UMIs first, padded with -1.
        guide_umis: The UMIs of each called guide, padded with 0.
    """

    values: np.ndarray
    n_guides: np.ndarray
    guide_codes: np.ndarray
    guide_umis: np.ndarray


def guide_call_arrays(
    cells: Any,
    codes: Any,
    umis: Any,
    *,
    n_cells: int,
    guide_ids: np.ndarray,
) -> GuideCallArrays:
    """Turn calls into the per-cell arrays of a ``guide_calls`` artifact.

    The width of the padded arrays is the largest number of guides called in
    one cell, and at least 1, so no call is dropped.

    Args:
        cells: The cell position of each call.
        codes: The guide position of each call in ``guide_ids``.
        umis: The UMIs of each call.
        n_cells: The number of cells.
        guide_ids: The guide IDs in assay order, from
            :func:`validated_guide_ids`.

    Returns:
        The label, the number of called guides, and the padded guide positions
        and UMIs of each cell.

    Raises:
        ValueError: If the three call arrays differ in length, a position is out
            of range, a UMI count is negative, not whole, or too large for
            uint32, or a cell has the same guide called twice.
    """
    cell_positions = np.asarray(cells).reshape(-1)
    guide_positions = np.asarray(codes).reshape(-1)
    counts = np.asarray(umis).reshape(-1)
    n_calls = len(cell_positions)
    if len(guide_positions) != n_calls or len(counts) != n_calls:
        raise ValueError("Calls need one cell, one guide, and one UMI count each")
    n_guide_ids = len(guide_ids)
    if n_calls:
        if cell_positions.dtype.kind not in "iu" or guide_positions.dtype.kind not in (
            "iu"
        ):
            raise ValueError("Cell and guide positions must be integers")
        if cell_positions.min() < 0 or cell_positions.max() >= n_cells:
            raise ValueError(f"Cell positions must lie in [0, {n_cells})")
        if guide_positions.min() < 0 or guide_positions.max() >= n_guide_ids:
            raise ValueError(f"Guide positions must lie in [0, {n_guide_ids})")
        if counts.dtype.kind not in "iu":
            whole = np.isfinite(counts) & (counts == np.floor(counts))
            if not whole.all():
                raise ValueError("UMI counts of calls must be whole numbers")
        if counts.min() < 0:
            raise ValueError("UMI counts of calls must not be negative")
        if counts.max() > np.iinfo(_UMI_DTYPE).max:
            raise ValueError(
                f"UMI counts of calls must be at most {np.iinfo(_UMI_DTYPE).max}"
            )
    cell_positions = cell_positions.astype(np.int64, copy=False)
    guide_positions = guide_positions.astype(np.int64, copy=False)
    counts = counts.astype(np.int64)
    pairs = cell_positions * n_guide_ids + guide_positions
    if len(np.unique(pairs)) != n_calls:
        raise ValueError("A guide is called more than once in one cell")
    # Order by cell, then by UMIs, most first, then by guide position.
    order = np.lexsort((guide_positions, -counts, cell_positions))
    cell_positions = cell_positions[order]
    guide_positions = guide_positions[order]
    counts = counts[order]
    n_guides = np.bincount(cell_positions, minlength=n_cells).astype(np.int32)
    width = max(1, int(n_guides.max(initial=0)))
    starts = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(n_guides, out=starts[1:])
    slots = np.arange(n_calls, dtype=np.int64) - starts[cell_positions]
    guide_codes = np.full((n_cells, width), -1, dtype=np.int32)
    guide_umis = np.zeros((n_cells, width), dtype=_UMI_DTYPE)
    guide_codes[cell_positions, slots] = guide_positions
    guide_umis[cell_positions, slots] = counts
    values = np.full(n_cells, MULTIPLE, dtype=object)
    values[n_guides == 0] = NEGATIVE
    single = n_guides == 1
    values[single] = np.asarray(guide_ids, dtype=object)[guide_codes[single, 0]]
    return GuideCallArrays(
        values=values,
        n_guides=n_guides,
        guide_codes=guide_codes,
        guide_umis=guide_umis,
    )


@dataclass(frozen=True, slots=True)
class GuideCalls:
    """The guide calls of a set of cells, read from a ``guide_calls`` artifact.

    Attributes:
        cells: One row per cell, indexed by cell ``ids``, with ``call`` (the ID
            of a cell's one called guide, ``"Negative"``, or ``"Multiple"``),
            ``n_guides``, ``top_guide`` (an empty string for a cell with no
            guide UMIs), ``top_umi``, ``top_fraction`` (NaN for a cell with no
            guide UMIs) and ``total_umi``.
        calls: One row per called guide of a cell, with the cell's ``ids``, the
            ``guide`` and its ``umis``, most UMIs first within each cell.
        guides: The ID of every guide of the assay, in assay order, including
            guides called in no cell.
        source: The operation that made the calls: ``"import_guide_calls"``
            for Cell Ranger's calls and ``"call_guides"`` for CyteArc's.
    """

    cells: pd.DataFrame = field(repr=False)
    calls: pd.DataFrame = field(repr=False)
    guides: pd.Index = field(repr=False)
    source: str


def guide_calls_frames(
    cell_ids: np.ndarray,
    *,
    values: np.ndarray,
    n_guides: np.ndarray,
    top_guide: np.ndarray,
    top_umi: np.ndarray,
    top_fraction: np.ndarray,
    total_umi: np.ndarray,
    guide_codes: np.ndarray,
    guide_umis: np.ndarray,
    guide_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the per-cell and per-call tables of stored guide calls."""
    index = pd.Index(np.asarray(cell_ids), name="ids")
    cells = pd.DataFrame(
        {
            "call": np.asarray(values, dtype=object),
            "n_guides": np.asarray(n_guides),
            "top_guide": np.asarray(top_guide, dtype=object),
            "top_umi": np.asarray(top_umi),
            "top_fraction": np.asarray(top_fraction, dtype=np.float64),
            "total_umi": np.asarray(total_umi),
        },
        index=index,
    )
    rows, slots = np.nonzero(np.asarray(guide_codes) >= 0)
    order = np.lexsort((slots, rows))
    rows, slots = rows[order], slots[order]
    calls = pd.DataFrame(
        {
            "ids": np.asarray(cell_ids)[rows],
            "guide": np.asarray(guide_ids, dtype=object)[
                np.asarray(guide_codes)[rows, slots]
            ],
            "umis": np.asarray(guide_umis)[rows, slots],
        }
    )
    return cells, calls
