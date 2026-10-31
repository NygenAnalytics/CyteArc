"""Clone assignments: which clone each cell belongs to.

A clone is the set of cells that descend from one ancestor cell. Lineage
barcodes, immune receptors, recorder alleles, phylogenies and somatic
variants each define clones their own way, and each source labels a cell with
one clone or none. ``CloneAssigner`` gathers those labels chunk by chunk,
refuses or drops a cell given two clones, and ranks the clones by size, so
every source saves clones alike: a ``clones`` artifact whose canonical
``values`` hold each cell's clone label and whose ``codes`` number the clones
by rank. These functions never touch storage.
"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.arrays import sort_categories

__all__ = [
    "MULTIPLE_POLICIES",
    "MultiplePolicy",
    "CloneAssigner",
    "CloneAssignment",
    "Clones",
    "clone_frames",
    "clone_label_texts",
    "ranked_clones",
]

type MultiplePolicy = Literal["error", "drop"]

#: The values of ``multiple``: what to do with a cell given two clones.
MULTIPLE_POLICIES: tuple[str, ...] = ("error", "drop")
# Cells given two clones named in the error.
_CONFLICT_EXAMPLES = 5
# Labels of at most this many ASCII digits sort as integers at once; floats
# hold them exactly, so ``sort_categories`` orders them alike.
_INTEGER_DIGITS = 15


def clone_label_texts(values: Any, missing: Any | None = None) -> np.ndarray:
    """Return clone labels as text, None where a cell has no clone.

    Text is stripped, and blank text means no clone, as do missing values
    and the values that ``missing`` marks. Integers and booleans read as
    their text, and floating-point whole numbers as integers, so a label
    column that pandas stored as float because of missing values gives
    ``"3"`` rather than ``"3.0"``.

    Raises:
        TypeError: If a label is a floating-point number that is not whole,
            or neither text nor a number.
        ValueError: If ``missing`` does not align with the values.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("clone labels must be one-dimensional")
    labels = np.full(len(array), None, dtype=object)
    kind = array.dtype.kind
    if kind == "b":
        labels[:] = np.where(array, "True", "False").astype(object)
    elif kind in "iu":
        labels[:] = array.astype(np.int64).astype(str).astype(object)
    elif kind == "f":
        present = np.isfinite(array)
        whole = present & (array == np.floor(array))
        if np.any(present & ~whole):
            example = array[present & ~whole][0]
            raise TypeError(
                f"clone labels must be text or whole numbers, not {float(example)!r}"
            )
        labels[whole] = array[whole].astype(np.int64).astype(str).astype(object)
    elif kind in "OUS":
        for position, value in enumerate(array.tolist()):
            labels[position] = _label_text(value)
    else:
        raise TypeError(f"clone labels must be text or numbers, not {array.dtype}")
    if missing is not None:
        mask = np.asarray(missing, dtype=bool)
        if mask.shape != labels.shape:
            raise ValueError("missing mask must align with the clone labels")
        labels[mask] = None
    return labels


def _label_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes | np.bytes_):
        value = value.decode()
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        if np.isnan(value):
            return None
        if value != np.floor(value):
            raise TypeError(
                f"clone labels must be text or whole numbers, not {float(value)!r}"
            )
        return str(int(value))
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"clone labels must be text or numbers, not {type(value).__name__}")


def _label_order(labels: list[str]) -> np.ndarray:
    """Rank labels in natural order: integers by value, then other text.

    Labels of at most 15 ASCII digits, such as TREX's clone numbers, sort as
    integers without the per-label key of ``sort_categories``, which orders
    them the same way. The labels stay Python strings, so a long label holds
    only its own text.
    """
    if not labels:
        return np.empty(0, dtype=np.int64)
    if all(
        len(label) <= _INTEGER_DIGITS and label.isascii() and label.isdigit()
        for label in labels
    ):
        values = np.fromiter(map(int, labels), dtype=np.int64, count=len(labels))
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        if not np.any(ordered[1:] == ordered[:-1]):
            return order.astype(np.int64)
        # Equal values, such as "07" and "7", follow their text.
        by_text = np.argsort(np.asarray(labels, dtype=object), kind="stable")
        text_rank = np.empty(len(labels), dtype=np.int64)
        text_rank[by_text] = np.arange(len(labels))
        return np.lexsort((text_rank, values)).astype(np.int64)
    rank = {label: position for position, label in enumerate(sort_categories(labels))}
    return np.argsort(
        np.fromiter(
            (rank[label] for label in labels), dtype=np.int64, count=len(labels)
        ),
        kind="stable",
    )


def ranked_clones(
    codes: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Number clones by rank: largest first, then by label in natural order.

    Args:
        codes: Each cell's clone as a position in ``labels``, -1 for none.
        labels: The clone labels.

    Returns:
        Each cell's clone by rank, as int64, -1 for none; the labels of the
        clones that hold a cell, by rank; and their sizes, as int64. A label
        that no cell holds is left out.
    """
    codes = np.asarray(codes, dtype=np.int64)
    sizes = np.bincount(codes[codes >= 0], minlength=len(labels)).astype(np.int64)
    held = np.flatnonzero(sizes > 0)
    by_label = _label_order([labels[position] for position in held.tolist()])
    label_rank = np.empty(len(held), dtype=np.int64)
    label_rank[by_label] = np.arange(len(held))
    order = held[np.lexsort((label_rank, -sizes[held]))]
    rank = np.full(len(labels), -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    ranked = np.full(len(codes), -1, dtype=np.int64)
    assigned = codes >= 0
    ranked[assigned] = rank[codes[assigned]]
    clone_ids = np.asarray(
        [labels[position] for position in order.tolist()], dtype=object
    )
    return ranked, clone_ids, sizes[order]


@dataclass(frozen=True, slots=True)
class CloneAssignment:
    """Each cell's clone, with the clones numbered by rank.

    Attributes:
        codes: Each cell's clone, a position in ``clone_ids``, -1 for none.
        clone_ids: The clone labels, largest clone first.
        clone_sizes: The cells of each clone.
        probabilities: Each cell's probability of its clone, NaN where the
            source gives none, or None when no source gave any.
        n_cells_in_several_clones: Cells left without a clone because their
            source gave them several.
    """

    codes: np.ndarray
    clone_ids: np.ndarray
    clone_sizes: np.ndarray
    probabilities: np.ndarray | None
    n_cells_in_several_clones: int

    @property
    def values(self) -> np.ndarray:
        """Each cell's clone label, an empty string for none."""
        values = np.full(len(self.codes), "", dtype=object)
        assigned = self.codes >= 0
        values[assigned] = self.clone_ids[self.codes[assigned]]
        return values

    @property
    def clone_size(self) -> np.ndarray:
        """The cells of each cell's clone, 0 for none, as int64."""
        sizes = np.zeros(len(self.codes), dtype=np.int64)
        assigned = self.codes >= 0
        sizes[assigned] = self.clone_sizes[self.codes[assigned]]
        return sizes


class CloneAssigner:
    """Gather each cell's clone from rows given chunk by chunk.

    A row names a cell, as its position among the cells, a clone label, and
    optionally the probability of that clone. Rows of a cell that name its
    clone again change nothing. A cell given two different clones raises
    ``ValueError`` with ``multiple="error"``, and with ``multiple="drop"`` it
    ends without a clone and is counted. Each label belongs to one source,
    such as one file, so equal labels of two sources raise ``ValueError``
    rather than merge two clones.

    Args:
        cell_ids: The IDs of the cells, which errors name.
        multiple: ``"error"`` or ``"drop"``.
    """

    def __init__(self, cell_ids: np.ndarray, *, multiple: MultiplePolicy) -> None:
        if multiple not in MULTIPLE_POLICIES:
            raise ValueError(
                f"multiple must be one of {', '.join(MULTIPLE_POLICIES)}, not "
                f"{multiple!r}"
            )
        self._cell_ids = np.asarray(cell_ids)
        self._multiple = multiple
        n_cells = len(self._cell_ids)
        self._codes = np.full(n_cells, -1, dtype=np.int64)
        self._several = np.zeros(n_cells, dtype=bool)
        self._probabilities: np.ndarray | None = None
        self._labels: dict[str, int] = {}
        self._label_list: list[str] = []
        self._label_sources: list[int] = []
        self._label_bytes = 0

    @property
    def n_labels(self) -> int:
        """The distinct clone labels given so far."""
        return len(self._label_list)

    @property
    def label_bytes(self) -> int:
        """The bytes of the distinct labels given so far, as Python strings."""
        return self._label_bytes

    def _label_codes(self, labels: np.ndarray, source: int) -> np.ndarray:
        """Return each label's code, numbering labels not seen before."""
        positions, distinct = pd.factorize(labels)
        mapped = np.empty(len(distinct), dtype=np.int64)
        for index, label in enumerate(distinct.tolist()):
            code = self._labels.get(label)
            if code is None:
                code = len(self._label_list)
                self._labels[label] = code
                self._label_list.append(label)
                self._label_sources.append(source)
                self._label_bytes += sys.getsizeof(label)
            elif self._label_sources[code] != source:
                raise ValueError(
                    f"The clone label {label!r} comes from two sources, so two "
                    "clones would merge; give each file its own barcode_prefix"
                )
            mapped[index] = code
        return mapped[np.asarray(positions, dtype=np.int64)]

    def add(
        self,
        cells: np.ndarray,
        labels: np.ndarray,
        *,
        probabilities: np.ndarray | None = None,
        source: int = 0,
    ) -> None:
        """Assign the cells of some rows to the clones that the rows name.

        Args:
            cells: Each row's cell, -1 for a row that names no cell.
            labels: Each row's clone label, None for a row without a clone.
            probabilities: Each row's probability of its clone, NaN for none,
                or None when the rows give no probabilities.
            source: The source of the rows, such as the position of a file.

        Raises:
            ValueError: With ``multiple="error"``, if a cell is given two
                clones; if a label comes from two sources; or if a cell's
                rows give its clone two probabilities.
        """
        cells = np.asarray(cells, dtype=np.int64)
        labels = np.asarray(labels, dtype=object)
        if len(labels) != len(cells):
            raise ValueError("Clone rows need one cell and one label each")
        keep = (cells >= 0) & pd.notna(labels)
        if not keep.any():
            return
        rows = cells[keep]
        codes = self._label_codes(labels[keep], source)
        # Rows of one cell, in cell order, so that a cell's rows are adjacent.
        order = np.argsort(rows, kind="stable")
        rows, codes = rows[order], codes[order]
        starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        first = codes[starts]
        lengths = np.diff(np.r_[starts, len(rows)])
        # A cell is given several clones when one of its rows names another.
        chunk_several = np.logical_or.reduceat(
            codes != np.repeat(first, lengths), starts
        )
        group_cells = rows[starts]
        held = self._codes[group_cells]
        several = chunk_several | ((held >= 0) & (held != first))
        if several.any():
            self._record_several(group_cells, held, first, several, rows, codes)
        settled = ~several & ~self._several[group_cells]
        self._codes[group_cells[settled]] = first[settled]
        if probabilities is not None:
            values = np.asarray(probabilities, dtype=np.float64)
            if len(values) != len(cells):
                raise ValueError("Clone rows need one probability each")
            self._add_probabilities(values[keep][order], rows, starts, settled)

    def _record_several(
        self,
        group_cells: np.ndarray,
        held: np.ndarray,
        first: np.ndarray,
        several: np.ndarray,
        rows: np.ndarray,
        codes: np.ndarray,
    ) -> None:
        if self._multiple == "error":
            examples = []
            for cell in group_cells[several][:_CONFLICT_EXAMPLES].tolist():
                given = set(codes[rows == cell].tolist())
                if self._codes[cell] >= 0:
                    given.add(int(self._codes[cell]))
                names = sorted(self._label_list[code] for code in given)
                examples.append(
                    f"{str(self._cell_ids[cell])!r} in "
                    + " and ".join(repr(name) for name in names)
                )
            n_several = int(np.count_nonzero(several))
            raise ValueError(
                f"{n_several} cells are listed in two different clones, for "
                f"example {'; '.join(examples)}. Pass multiple='drop' to leave "
                "such cells without a clone"
            )
        dropped = group_cells[several]
        self._several[dropped] = True
        self._codes[dropped] = -1

    def _add_probabilities(
        self,
        values: np.ndarray,
        rows: np.ndarray,
        starts: np.ndarray,
        settled: np.ndarray,
    ) -> None:
        """Keep each settled cell's probability from its rows, in cell order.

        A row's NaN gives no probability, in this chunk as in later ones, so
        only two different numbers for one cell conflict.
        """
        if self._probabilities is None:
            self._probabilities = np.full(len(self._codes), np.nan)
        low = np.fmin.reduceat(values, starts)
        high = np.fmax.reduceat(values, starts)
        differ = settled & ~np.isnan(low) & (low != high)
        if differ.any():
            cell = rows[starts][np.flatnonzero(differ)[0]]
            raise ValueError(
                f"Cell {str(self._cell_ids[cell])!r} is listed with two "
                "probabilities of its clone"
            )
        cells = rows[starts][settled]
        stored = self._probabilities[cells]
        new = low[settled]
        clash = ~np.isnan(stored) & ~np.isnan(new) & (stored != new)
        if clash.any():
            raise ValueError(
                f"Cell {str(self._cell_ids[cells[clash][0]])!r} is listed with two "
                "probabilities of its clone"
            )
        fill = np.isnan(stored)
        self._probabilities[cells[fill]] = new[fill]

    def finish(self) -> CloneAssignment:
        """Rank the clones and return each cell's clone."""
        ranked, clone_ids, clone_sizes = ranked_clones(self._codes, self._label_list)
        probabilities = None
        if self._probabilities is not None:
            probabilities = np.where(ranked >= 0, self._probabilities, np.nan)
        return CloneAssignment(
            codes=ranked,
            clone_ids=clone_ids,
            clone_sizes=clone_sizes,
            probabilities=probabilities,
            n_cells_in_several_clones=int(np.count_nonzero(self._several)),
        )


@dataclass(frozen=True, slots=True)
class Clones:
    """The clones of a set of cells, read from a ``clones`` artifact.

    Attributes:
        cells: One row per cell, indexed by cell ``ids``: ``clone`` (its
            label, as a categorical of the labels in rank order, NA without
            one), ``clone_size`` (the cells of its clone, 0 without one) and,
            when the source gave them, ``clone_probability`` (the probability
            of its clone, NaN without one).
        clones: One row per clone, largest first: ``clone`` and ``n_cells``.
        source: The operation that saved the clones, such as
            ``"import_clones"``.
    """

    cells: pd.DataFrame = field(repr=False)
    clones: pd.DataFrame = field(repr=False)
    source: str


def clone_frames(
    cell_ids: np.ndarray,
    *,
    codes: np.ndarray,
    clone_size: np.ndarray,
    clone_ids: np.ndarray,
    clone_sizes: np.ndarray,
    extra: Mapping[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the tables of cells and clones of stored clones.

    Args:
        cell_ids: The ``ids`` of the cells.
        codes: Each cell's clone, a position in ``clone_ids``, -1 without one.
        clone_size: Each cell's clone size.
        clone_ids: The clone labels, by rank.
        clone_sizes: The cells of each clone.
        extra: Further per-cell columns, such as ``clone_probability``.
    """
    # The labels are categories coded by the stored codes, ordered as the
    # clones rank, so the table holds no text per cell.
    columns: dict[str, Any] = {
        "clone": pd.Categorical.from_codes(
            np.asarray(codes, dtype=np.int64),
            categories=pd.Index(np.asarray(clone_ids, dtype=object)),
        ),
        "clone_size": np.asarray(clone_size, dtype=np.int64),
    }
    for name, values in (extra or {}).items():
        columns[name] = np.asarray(values)
    cells = pd.DataFrame(columns, index=pd.Index(np.asarray(cell_ids), name="ids"))
    table = pd.DataFrame(
        {
            "clone": np.asarray(clone_ids, dtype=object),
            "n_cells": np.asarray(clone_sizes, dtype=np.int64),
        }
    )
    return cells, table
