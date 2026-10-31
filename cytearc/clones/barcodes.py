"""Expressed lineage barcodes: the molecules and barcode counts of each cell.

Lineage barcodes such as LARRY's, CellTag's and TREX's are random sequences
written into a cell's genome; the cell and its descendants transcribe them,
so single-cell sequencing reads each cell's barcodes beside its
transcriptome, and cells that share barcodes share an ancestor. Tools that
extract the barcodes write them at three levels:

- reads: one row per read, naming its cell, its UMI and its barcode;
- molecules: one row per cell, UMI and barcode, with the molecule's reads or
  without them;
- pairs: one row per cell and barcode, with its UMIs or its reads.

An import keeps every cell's barcodes as records of one long table
(:mod:`cytearc.metadata.records`) ordered by cell, then library, barcode and
UMI. A record is a molecule, which has a UMI, or a pair, which has none:

- ``barcode`` and ``umi`` are text, stored once per distinct value; a pair's
  ``umi`` is missing.
- ``umis`` is 1 for a molecule, and a pair's UMIs, -1 where its library
  counts none.
- ``reads`` is the reads of a molecule or a pair, -1 where its library
  counts none.

Rows that are reads collapse into molecules whose reads are their rows, or,
without UMIs, into pairs. Rows that name one record, such as a molecule that
two files of one library list, merge into one record that sums their reads,
and a pair's UMIs too. A library holds molecules or pairs, never both, and a
record's source is its library. ``BarcodeCounter`` collapses rows chunk by
chunk and merges the chunks as they accumulate, so it holds at most about
twice as many rows as there are records, however many reads the files list.
These functions never touch storage.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..metadata.records import (
    INTEGER,
    TEXT,
    EncodedText,
    FieldKind,
    RecordTable,
    pack_text,
)

__all__ = [
    "FIELDS",
    "LEVELS",
    "BarcodeCounter",
    "BarcodeLevel",
    "BarcodePairs",
    "CountedBarcodes",
    "LineageBarcodes",
    "barcode_pairs",
    "cell_summaries",
    "library_summaries",
    "lineage_barcode_frames",
    "pair_keys",
    "repeated_keys",
]

type BarcodeLevel = Literal["molecules", "pairs"]

#: The fields of each barcode record, with their kinds, in storage order.
FIELDS: Mapping[str, FieldKind] = MappingProxyType(
    {"barcode": TEXT, "umi": TEXT, "umis": INTEGER, "reads": INTEGER}
)
#: What a library's records are: molecules, which have UMIs, or pairs of a
#: cell and a barcode, which do not.
LEVELS: tuple[BarcodeLevel, ...] = ("molecules", "pairs")
# Bytes that a distinct text value takes while it is held: its Python string
# and its references.
_TEXT_BYTES = 96
# Bytes of the arrays that each held row takes: its cell, library, barcode
# and UMI codes, UMIs and reads.
_ROW_BYTES = 36
# Bytes that merging takes beside the held rows, for each row and each
# distinct text value: the rows joined, their order and their sorted copy,
# and the table that codes the text.
_MERGE_ROW_BYTES = 110
_MERGE_TEXT_BYTES = 48
# Merging begins once at least this many rows wait, so that small imports
# merge once.
_MIN_MERGE_ROWS = 250_000


@dataclass(slots=True)
class _Rows:
    """Rows of barcodes, with text coded into their own distinct values."""

    cells: np.ndarray
    libraries: np.ndarray
    barcodes: np.ndarray
    barcode_values: np.ndarray
    umis_text: np.ndarray
    umi_values: np.ndarray
    umis: np.ndarray
    reads: np.ndarray

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def nbytes(self) -> int:
        return _ROW_BYTES * len(self) + _TEXT_BYTES * (
            len(self.barcode_values) + len(self.umi_values)
        )


def _known_sums(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum runs of counts; a run that holds an unknown count, -1, sums to -1."""
    sums = np.add.reduceat(np.maximum(values, 0), starts)
    sums[np.minimum.reduceat(values, starts) < 0] = -1
    return np.asarray(sums, dtype=np.int64)


def _collapsed(rows: _Rows) -> _Rows:
    """Merge rows that name one record: one cell, library, barcode and UMI."""
    if not len(rows):
        return rows
    order = np.lexsort((rows.umis_text, rows.barcodes, rows.libraries, rows.cells))
    cells = rows.cells[order]
    libraries = rows.libraries[order]
    barcodes = rows.barcodes[order]
    umis_text = rows.umis_text[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = (
        (cells[1:] != cells[:-1])
        | (libraries[1:] != libraries[:-1])
        | (barcodes[1:] != barcodes[:-1])
        | (umis_text[1:] != umis_text[:-1])
    )
    starts = np.flatnonzero(new)
    molecule = umis_text[starts] >= 0
    # A molecule is one UMI however many rows name it; a pair sums its UMIs.
    umis = np.where(molecule, 1, _known_sums(rows.umis[order], starts))
    return _Rows(
        cells=cells[starts],
        libraries=libraries[starts],
        barcodes=barcodes[starts],
        barcode_values=rows.barcode_values,
        umis_text=umis_text[starts],
        umi_values=rows.umi_values,
        umis=np.asarray(umis, dtype=np.int64),
        reads=_known_sums(rows.reads[order], starts),
    )


def _shared_codes(
    parts: Sequence[_Rows], codes_of: str, values_of: str
) -> tuple[np.ndarray, np.ndarray]:
    """Code the text of several parts by one set of distinct values.

    Returns:
        Every row's code into the shared values, -1 where missing, and the
        shared values in the order they first appear.
    """
    values = [np.asarray(getattr(part, values_of), dtype=object) for part in parts]
    every = np.concatenate([np.empty(0, dtype=object), *values])
    if len(every):
        recoded, shared = pd.factorize(every)
        shared = np.asarray(shared, dtype=object)
    else:
        recoded, shared = np.empty(0, dtype=np.int64), np.empty(0, dtype=object)
    pieces = []
    start = 0
    for part, part_values in zip(parts, values, strict=True):
        codes = np.asarray(getattr(part, codes_of), dtype=np.int64)
        piece = np.full(len(codes), -1, dtype=np.int32)
        present = codes >= 0
        piece[present] = recoded[start + codes[present]]
        pieces.append(piece)
        start += len(part_values)
    return np.concatenate([np.empty(0, dtype=np.int32), *pieces]), shared


def _merged(parts: Sequence[_Rows]) -> _Rows:
    """Merge parts that code their text alike or not into one collapsed part."""
    barcodes, barcode_values = _shared_codes(parts, "barcodes", "barcode_values")
    umis_text, umi_values = _shared_codes(parts, "umis_text", "umi_values")
    return _collapsed(
        _Rows(
            cells=np.concatenate([part.cells for part in parts]),
            libraries=np.concatenate([part.libraries for part in parts]),
            barcodes=barcodes,
            barcode_values=barcode_values,
            umis_text=umis_text,
            umi_values=umi_values,
            umis=np.concatenate([part.umis for part in parts]),
            reads=np.concatenate([part.reads for part in parts]),
        )
    )


@dataclass(frozen=True, slots=True)
class CountedBarcodes:
    """The records that ``BarcodeCounter`` collapsed the rows of files into.

    Attributes:
        table: The records, ordered by cell, library, barcode and UMI, whose
            sources are their libraries.
        libraries: The name of each library, by its position, sorted.
        levels: Each library's level, ``"molecules"`` or ``"pairs"``.
    """

    table: RecordTable
    libraries: tuple[str, ...]
    levels: tuple[BarcodeLevel, ...]


class BarcodeCounter:
    """Collapse the rows of barcode files into molecules and pairs, chunk by chunk.

    Each chunk is collapsed on its own; once the rows that wait outnumber
    the records held, and ``merge_rows``, the waiting chunks merge into the
    records, so the counter holds at most about twice as many rows as there
    are records.

    Args:
        merge_rows: The fewest waiting rows that merge into the records.

    Raises:
        ValueError: If ``merge_rows`` is not positive.
    """

    def __init__(self, *, merge_rows: int = _MIN_MERGE_ROWS) -> None:
        if int(merge_rows) < 1:
            raise ValueError("merge_rows must be at least 1")
        self._merge_rows = int(merge_rows)
        self._held: _Rows | None = None
        self._waiting: list[_Rows] = []
        self._library_codes: dict[str, int] = {}
        self._library_names: list[str] = []
        # Whether each library's records are molecules, by library code.
        self._library_molecules: list[bool] = []
        self._finished = False

    @property
    def n_rows(self) -> int:
        """The rows held: records, and collapsed chunks that wait to merge."""
        held = 0 if self._held is None else len(self._held)
        return held + sum(len(part) for part in self._waiting)

    @property
    def nbytes(self) -> int:
        """The bytes that the held rows and their distinct text take."""
        held = 0 if self._held is None else self._held.nbytes
        return held + sum(part.nbytes for part in self._waiting)

    @property
    def peak_bytes(self) -> int:
        """The bytes held at the peak of merging the rows held now."""
        parts = ([] if self._held is None else [self._held]) + self._waiting
        n_text = sum(len(part.barcode_values) + len(part.umi_values) for part in parts)
        return self.nbytes + _MERGE_ROW_BYTES * self.n_rows + _MERGE_TEXT_BYTES * n_text

    def add_library(self, name: str, *, molecules: bool) -> None:
        """Name a library before its rows, so that it is listed without records.

        Raises:
            ValueError: If the library holds the other level of records.
        """
        self._library_code(str(name), bool(molecules))

    def _library_code(self, name: str, molecules: bool) -> int:
        code = self._library_codes.get(name)
        if code is None:
            code = len(self._library_names)
            self._library_codes[name] = code
            self._library_names.append(name)
            self._library_molecules.append(molecules)
        elif self._library_molecules[code] != molecules:
            raise ValueError(
                f"The library {name!r} holds molecules in one file and barcode "
                "counts without UMIs in another; give each its own library"
            )
        return code

    def add(
        self,
        cells: Any,
        libraries: Any,
        barcodes: Any,
        *,
        umis: Any | None = None,
        umi_counts: Any | None = None,
        reads: Any | None = None,
        rows_are_reads: bool = False,
    ) -> None:
        """Add rows of one level: reads, molecules or pairs.

        Args:
            cells: Each row's cell, its position among the selected cells.
            libraries: Each row's library name.
            barcodes: Each row's barcode, as text.
            umis: Each row's UMI, as text, or None when the rows have none, so
                that they count pairs.
            umi_counts: Each pair's UMIs, -1 where unknown, or None for
                unknown; not for rows with UMIs.
            reads: Each row's reads, -1 where unknown, or None for unknown;
                not for rows that are reads.
            rows_are_reads: Whether each row is one read.

        Raises:
            ValueError: If the arrays differ in length, a cell is negative, a
                barcode or UMI is missing, a count is below -1, counts are
                given that the level does not have, a library holds
                molecules and pairs, or the counter has finished.
        """
        if self._finished:
            raise RuntimeError("The counter has finished; start a new one")
        cell_positions = np.asarray(cells, dtype=np.int64).reshape(-1)
        n_rows = len(cell_positions)
        texts = np.asarray(barcodes, dtype=object).reshape(-1)
        names = np.asarray(libraries, dtype=object).reshape(-1)
        if len(texts) != n_rows or len(names) != n_rows:
            raise ValueError("Each row needs one cell, one library and one barcode")
        if n_rows and cell_positions.min() < 0:
            raise ValueError("A row's cell must be a position among the cells")
        if umis is not None and umi_counts is not None:
            raise ValueError("Rows with UMIs are molecules, which count no UMIs")
        if rows_are_reads and reads is not None:
            raise ValueError("Rows that are reads count their reads by rows")
        molecules = umis is not None
        barcode_codes, barcode_values = _factorized(texts, "barcode")
        if molecules:
            umi_codes, umi_values = _factorized(umis, "UMI", n_rows)
            unit_umis = np.ones(n_rows, dtype=np.int64)
        else:
            umi_codes = np.full(n_rows, -1, dtype=np.int32)
            umi_values = np.empty(0, dtype=object)
            unit_umis = _counts(umi_counts, n_rows, "UMI counts")
        if rows_are_reads:
            row_reads = np.ones(n_rows, dtype=np.int64)
        else:
            row_reads = _counts(reads, n_rows, "read counts")
        library_codes, library_names = pd.factorize(names)
        if np.any(library_codes < 0):
            raise ValueError("Each row needs a library name")
        mapped = np.asarray(
            [self._library_code(str(name), molecules) for name in library_names],
            dtype=np.int32,
        )
        part = _collapsed(
            _Rows(
                cells=cell_positions,
                libraries=mapped[library_codes],
                barcodes=barcode_codes,
                barcode_values=barcode_values,
                umis_text=umi_codes,
                umi_values=umi_values,
                umis=unit_umis,
                reads=row_reads,
            )
        )
        self._waiting.append(part)
        held = 0 if self._held is None else len(self._held)
        waiting = sum(len(rows) for rows in self._waiting)
        if waiting >= max(held, self._merge_rows):
            self._merge()

    def _merge(self) -> None:
        parts = ([] if self._held is None else [self._held]) + self._waiting
        self._waiting = []
        self._held = None
        if parts:
            self._held = _merged(parts)

    def finish(self) -> CountedBarcodes:
        """Return the records, ordered by cell, library, barcode and UMI.

        Text values are sorted, so the order and the codes do not depend on
        the order of the rows or of the files.

        Raises:
            RuntimeError: If the counter has finished already.
        """
        if self._finished:
            raise RuntimeError("The counter has finished; start a new one")
        self._finished = True
        self._merge()
        held = self._held
        if held is None:
            held = _collapsed(_empty_rows())
        self._held = None
        library_order = np.argsort(np.asarray(self._library_names, dtype=object))
        library_ranks = np.empty(len(library_order), dtype=np.int64)
        library_ranks[library_order] = np.arange(len(library_order))
        libraries = library_ranks[held.libraries]
        barcode = _sorted_text(held.barcodes, held.barcode_values)
        umi = _sorted_text(held.umis_text, held.umi_values)
        order = np.lexsort((umi.codes, barcode.codes, libraries, held.cells))
        table = RecordTable(
            cells=held.cells[order],
            sources=np.asarray(libraries[order], dtype=np.int32),
            kinds=FIELDS,
            text=MappingProxyType(
                {
                    "barcode": EncodedText(
                        codes=barcode.codes[order],
                        data=barcode.data,
                        offsets=barcode.offsets,
                    ),
                    "umi": EncodedText(
                        codes=umi.codes[order], data=umi.data, offsets=umi.offsets
                    ),
                }
            ),
            ragged=MappingProxyType({}),
            values=MappingProxyType(
                {"umis": held.umis[order], "reads": held.reads[order]}
            ),
        )
        names = tuple(
            str(self._library_names[position]) for position in library_order.tolist()
        )
        levels: tuple[BarcodeLevel, ...] = tuple(
            "molecules" if self._library_molecules[position] else "pairs"
            for position in library_order.tolist()
        )
        return CountedBarcodes(table=table, libraries=names, levels=levels)


def _empty_rows() -> _Rows:
    empty = np.empty(0, dtype=np.int64)
    codes = np.empty(0, dtype=np.int32)
    return _Rows(
        cells=empty,
        libraries=codes,
        barcodes=codes,
        barcode_values=np.empty(0, dtype=object),
        umis_text=codes,
        umi_values=np.empty(0, dtype=object),
        umis=empty,
        reads=empty,
    )


def _factorized(
    values: Any, what: str, n_rows: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Code text by its distinct values, refusing missing values."""
    text = np.asarray(values, dtype=object).reshape(-1)
    if n_rows is not None and len(text) != n_rows:
        raise ValueError(f"Each row needs one {what}")
    codes, distinct = pd.factorize(text)
    if np.any(codes < 0):
        raise ValueError(f"A row's {what} is missing")
    return np.asarray(codes, dtype=np.int32), np.asarray(distinct, dtype=object)


def _counts(values: Any | None, n_rows: int, what: str) -> np.ndarray:
    """Return counts of at least -1, or -1 for each row without counts."""
    if values is None:
        return np.full(n_rows, -1, dtype=np.int64)
    counts = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(counts) != n_rows:
        raise ValueError(f"Each row needs one of the {what}")
    if n_rows and counts.min() < -1:
        raise ValueError(f"The {what} must be at least 0, or -1 where unknown")
    return counts


def _sorted_text(codes: np.ndarray, values: np.ndarray) -> EncodedText:
    """Recode text so that its distinct values are sorted, -1 kept for missing."""
    values = np.asarray(values, dtype=object)
    # Object text sorts by code point without a copy padded to the longest.
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(len(order), dtype=np.int32)
    recoded = np.full(len(codes), -1, dtype=np.int32)
    present = codes >= 0
    recoded[present] = ranks[codes[present]]
    data, offsets = pack_text(values[order])
    return EncodedText(codes=recoded, data=data, offsets=offsets)


def _cell_sums(cells: np.ndarray, values: np.ndarray, n_cells: int) -> np.ndarray:
    """Sum each cell's counts; a cell with an unknown count, -1, sums to -1."""
    sums = np.bincount(cells, weights=np.maximum(values, 0), minlength=n_cells)
    unknown = np.bincount(cells, weights=values < 0, minlength=n_cells) > 0
    result = np.asarray(sums, dtype=np.int64)
    result[unknown] = -1
    return result


def cell_summaries(table: RecordTable, n_cells: int) -> dict[str, np.ndarray]:
    """Count each cell's records, distinct barcodes, UMIs and reads.

    Returns:
        ``n_barcodes``, the distinct barcodes of each cell whatever library
        holds them, as int32; ``n_umis`` and ``n_reads``, -1 where a record
        of the cell counts none, as int64; and ``n_records``, its records,
        as int32.
    """
    cells = np.asarray(table.cells, dtype=np.int64)
    barcode = table.text["barcode"]
    n_values = max(1, len(barcode.offsets) - 1)
    pairs = np.unique(cells * n_values + np.asarray(barcode.codes, dtype=np.int64))
    return {
        "n_barcodes": np.bincount(pairs // n_values, minlength=n_cells).astype(
            np.int32
        ),
        "n_umis": _cell_sums(cells, np.asarray(table.values["umis"]), n_cells),
        "n_reads": _cell_sums(cells, np.asarray(table.values["reads"]), n_cells),
        "n_records": np.bincount(cells, minlength=n_cells).astype(np.int32),
    }


def library_summaries(counted: CountedBarcodes) -> list[dict[str, Any]]:
    """Describe each library: its level, records, and whether it counts UMIs and reads.

    Returns:
        One mapping per library, by position: ``library``, its name;
        ``level``, ``"molecules"`` or ``"pairs"``; ``records``; ``has_umis``
        and ``has_reads``, whether its records count their UMIs and their
        reads, False for a library without records.
    """
    table = counted.table
    sources = np.asarray(table.sources, dtype=np.int64)
    umis = np.asarray(table.values["umis"])
    reads = np.asarray(table.values["reads"])
    summaries = []
    for position, (name, level) in enumerate(
        zip(counted.libraries, counted.levels, strict=True)
    ):
        held = sources == position
        summaries.append(
            {
                "library": str(name),
                "level": str(level),
                "records": int(np.count_nonzero(held)),
                "has_umis": bool(held.any() and np.all(umis[held] >= 0)),
                "has_reads": bool(held.any() and np.all(reads[held] >= 0)),
            }
        )
    return summaries


@dataclass(frozen=True, slots=True)
class BarcodePairs:
    """The barcodes of each cell and library, with their UMIs and reads.

    Attributes:
        cells: Each pair's cell, as its position among the selected cells.
        sources: Each pair's library, by position.
        barcodes: Each pair's barcode, as its code.
        umis: The pair's distinct UMIs, or its counted UMIs; -1 where unknown.
        reads: The pair's reads, -1 where unknown.
        molecules: The pair's molecule records, 0 for a counted pair.
    """

    cells: np.ndarray
    sources: np.ndarray
    barcodes: np.ndarray
    umis: np.ndarray
    reads: np.ndarray
    molecules: np.ndarray


def _pair_runs(
    cells: np.ndarray,
    sources: np.ndarray,
    barcodes: np.ndarray,
    is_molecule: np.ndarray,
    umis: np.ndarray,
    reads: np.ndarray,
) -> BarcodePairs:
    """Collapse records ordered by cell, library and barcode into pairs."""
    cells = np.asarray(cells, dtype=np.int64)
    sources = np.asarray(sources, dtype=np.int64)
    barcodes = np.asarray(barcodes, dtype=np.int64)
    if not len(cells):
        empty = np.empty(0, dtype=np.int64)
        return BarcodePairs(empty, empty, empty, empty, empty, empty)
    new = np.ones(len(cells), dtype=bool)
    new[1:] = (
        (cells[1:] != cells[:-1])
        | (sources[1:] != sources[:-1])
        | (barcodes[1:] != barcodes[:-1])
    )
    starts = np.flatnonzero(new)
    return BarcodePairs(
        cells=cells[starts],
        sources=sources[starts],
        barcodes=barcodes[starts],
        umis=_known_sums(np.asarray(umis, dtype=np.int64), starts),
        reads=_known_sums(np.asarray(reads, dtype=np.int64), starts),
        molecules=np.add.reduceat(np.asarray(is_molecule, dtype=np.int64), starts),
    )


def barcode_pairs(table: RecordTable) -> BarcodePairs:
    """Collapse records into pairs of a cell and a barcode within each library.

    Records are ordered by cell, library, barcode and UMI, so the molecules
    of one pair are consecutive and their UMIs distinct: a pair's UMIs are
    its molecules, or the UMIs that a counted pair records.
    """
    return _pair_runs(
        table.cells,
        table.sources,
        table.text["barcode"].codes,
        np.asarray(table.text["umi"].codes) >= 0,
        table.values["umis"],
        table.values["reads"],
    )


@dataclass(frozen=True, slots=True)
class LineageBarcodes:
    """The lineage barcodes of a set of cells, from a ``lineage_barcodes`` artifact.

    Attributes:
        cells: One row per selected cell, indexed by cell ``ids``:
            ``n_barcodes``, its distinct barcodes, ``n_umis`` and
            ``n_reads``, -1 where a library of the cell counts none, and
            ``n_records``, its rows of ``records``.
        records: One row per molecule or pair, by cell, library, barcode and
            UMI: ``cell``, ``library``, ``barcode`` and ``umi`` as
            categoricals, ``umi`` missing for a pair, and ``umis`` and
            ``reads`` as ``Int64``, missing where unknown. A molecule's
            ``umis`` is 1.
        libraries: One row per library: ``library``, ``level``
            (``"molecules"`` or ``"pairs"``), ``records``, and ``has_umis``
            and ``has_reads``, whether its records count UMIs and reads.
    """

    cells: pd.DataFrame = field(repr=False)
    records: pd.DataFrame = field(repr=False)
    libraries: pd.DataFrame = field(repr=False)

    def pairs(self) -> pd.DataFrame:
        """Return one row per cell, library and barcode, with its UMIs and reads.

        A pair's ``umis`` are its distinct UMIs, or the UMIs that its library
        counts per barcode, and ``molecules`` its molecules; ``umis`` and
        ``reads`` are missing where its library counts none.
        """
        records = self.records
        pairs = _pair_runs(
            records["cell"].cat.codes.to_numpy(),
            records["library"].cat.codes.to_numpy(),
            records["barcode"].cat.codes.to_numpy(),
            records["umi"].cat.codes.to_numpy() >= 0,
            records["umis"].fillna(-1).to_numpy(dtype=np.int64),
            records["reads"].fillna(-1).to_numpy(dtype=np.int64),
        )
        return pd.DataFrame(
            {
                "cell": pd.Categorical.from_codes(
                    pairs.cells, dtype=records["cell"].dtype
                ),
                "library": pd.Categorical.from_codes(
                    pairs.sources, dtype=records["library"].dtype
                ),
                "barcode": pd.Categorical.from_codes(
                    pairs.barcodes, dtype=records["barcode"].dtype
                ),
                "umis": _nullable_counts(pairs.umis),
                "reads": _nullable_counts(pairs.reads),
                "molecules": pairs.molecules,
            }
        )


def _nullable_counts(values: np.ndarray) -> pd.arrays.IntegerArray:
    counts = np.asarray(values, dtype=np.int64)
    missing = counts < 0
    return pd.arrays.IntegerArray(np.where(missing, 0, counts), missing)


def lineage_barcode_frames(
    cell_ids: Any,
    summaries: Mapping[str, np.ndarray],
    table: RecordTable,
    libraries: Sequence[Mapping[str, Any]],
) -> LineageBarcodes:
    """Build the tables of a ``LineageBarcodes`` record.

    Args:
        cell_ids: The selected cells' IDs, in selection order.
        summaries: The per-cell arrays of :func:`cell_summaries`.
        table: The records.
        libraries: The libraries of :func:`library_summaries`, by position.
    """
    ids = pd.Index(np.asarray(cell_ids, dtype=object), name="ids")
    cells = pd.DataFrame(
        {
            "n_barcodes": np.asarray(summaries["n_barcodes"], dtype=np.int64),
            "n_umis": np.asarray(summaries["n_umis"], dtype=np.int64),
            "n_reads": np.asarray(summaries["n_reads"], dtype=np.int64),
            "n_records": np.asarray(summaries["n_records"], dtype=np.int64),
        },
        index=ids,
    )
    names = [str(library["library"]) for library in libraries]
    barcode = table.text["barcode"]
    umi = table.text["umi"]
    records = pd.DataFrame(
        {
            "cell": pd.Categorical.from_codes(
                np.asarray(table.cells, dtype=np.int64), categories=ids
            ),
            "library": pd.Categorical.from_codes(
                np.asarray(table.sources, dtype=np.int64), categories=names
            ),
            "barcode": pd.Categorical.from_codes(
                np.asarray(barcode.codes, dtype=np.int64),
                categories=pd.Index(barcode.categories, dtype=object),
            ),
            "umi": pd.Categorical.from_codes(
                np.asarray(umi.codes, dtype=np.int64),
                categories=pd.Index(umi.categories, dtype=object),
            ),
            "umis": _nullable_counts(table.values["umis"]),
            "reads": _nullable_counts(table.values["reads"]),
        }
    )
    library_table = pd.DataFrame(
        {
            "library": pd.Series(names, dtype=object),
            "level": pd.Series(
                [str(library["level"]) for library in libraries], dtype=object
            ),
            "records": pd.Series(
                [int(library["records"]) for library in libraries], dtype=np.int64
            ),
            "has_umis": pd.Series(
                [bool(library["has_umis"]) for library in libraries], dtype=bool
            ),
            "has_reads": pd.Series(
                [bool(library["has_reads"]) for library in libraries], dtype=bool
            ),
        }
    )
    return LineageBarcodes(cells=cells, records=records, libraries=library_table)


# Two 64-bit hashes of each text, whose combination keys a pair of texts by
# 128 bits, and the odd constant that mixes the first text into the key.
_HASH_KEYS = ("lineage-barcode1", "lineage-barcode2")
_MIX = np.uint64(0x9E3779B97F4A7C15)


def pair_keys(first: Any, second: Any) -> np.ndarray:
    """Key each pair of texts, such as a cell and a UMI, by 128 bits.

    Two pairs share a key only if their texts are equal, but for a chance
    of about one in 2**128.

    Returns:
        One row per pair and two uint64 columns.
    """
    first = np.asarray(first, dtype=object).reshape(-1)
    second = np.asarray(second, dtype=object).reshape(-1)
    columns = []
    for key in _HASH_KEYS:
        mixed = pd.util.hash_array(first, hash_key=key) * _MIX
        columns.append(np.bitwise_xor(mixed, pd.util.hash_array(second, hash_key=key)))
    return np.stack(columns, axis=1).astype(np.uint64, copy=False)


def repeated_keys(keys: Sequence[np.ndarray]) -> int:
    """Count the keys that repeat an earlier key.

    The keys are joined and sorted in place, so finding the repeats holds a
    copy of the keys and one byte per key beside them.
    """
    joined = np.concatenate([np.empty((0, 2), dtype=np.uint64), *keys])
    pairs = joined.view([("high", np.uint64), ("low", np.uint64)]).reshape(-1)
    pairs.sort()
    return int(np.count_nonzero(pairs[1:] == pairs[:-1]))
