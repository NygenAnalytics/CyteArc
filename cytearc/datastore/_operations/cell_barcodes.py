"""Join the barcodes of imported files to the selected cells of a store.

Files that importers read, such as guide calls or clone tables, name cells by
barcode, while a store names them by cell ID. A barcode names the cell whose ID
is the file's prefix, then the barcode, then the file's suffix; a pattern can
first keep the part of the barcode that its ``barcode`` group matches. Every
importer joins its files through ``CellBarcodeJoin`` and orders and identifies
them through ``barcode_files``, so they share the rules, the diagnostics that
count the barcodes that name no cell of the store and those that name a cell
outside the selection, and the warning about the first. The join charges what
it holds to the importer's memory ledger.
"""

import functools
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...metadata.rows import row_read_bytes
from ...storage.async_execution import zarr_decoding_threads
from ...storage.diagnostics import EXAMPLES, DiagnosticRule
from ...storage.geometry import array_geometry
from ...utils.arrays import union_sorted
from ...utils.digest import file_sha256
from ...utils.memory_budget import (
    MemoryLedger,
    hash_table_bytes,
    object_array_bytes,
    python_text_bytes,
)
from ...utils.arguments import quoted_value

__all__ = [
    "LOOKUP_BYTES_PER_CELL",
    "BarcodeFile",
    "CellBarcodeJoin",
    "barcode_example",
    "barcode_files",
    "barcode_pattern",
    "derived_id_bytes",
    "id_hashing_bytes",
    "join_bytes",
    "lookup_bytes",
    "store_pass_bytes",
    "texts_per_file",
    "unmatched_barcode_rules",
    "unmatched_barcodes_message",
]
#: Bytes of each ID of a pandas index of text IDs that finds some of them:
#: the ID's slot and hash in the index. Clone figures and loaded variant
#: counts count their lookups so; the join counts its own by ``lookup_bytes``.
LOOKUP_BYTES_PER_CELL = 100
# Bytes held for each distinct barcode that matches no selected cell: its
# 64-bit hash and its copy in the hashes that a merge of new ones makes.
_UNMATCHED_BYTES_PER_BARCODE = 16
# Bytes of each barcode while its cell ID is derived, whatever the options:
# the pandas copy of the pointers to the barcodes, and whether it names a cell.
_COPY_BYTES_PER_BARCODE = 16
# Bytes of each copy of a barcode that a prefix or a suffix makes, beside its
# characters: its Python string header and the pointers to it.
_AFFIXED_BYTES_PER_BARCODE = 80
# Bytes of each barcode that a pattern's extraction holds, beside the
# characters of the part it keeps: that part's Python string, the frame and
# series of the extraction, and the mask of the barcodes it matched.
_EXTRACTED_BYTES_PER_BARCODE = 176
# Bytes that matching each barcode holds while ``CellBarcodeJoin.cells`` runs,
# beside the IDs that its options derive: its position, the masks and the
# pandas copies of the barcodes, and for a barcode that matches no cell, its
# hash and the unique arrays that count it. Hashing a barcode also encodes
# its text, a byte for each character.
_MATCH_BYTES_PER_BARCODE = 128
# Bytes of each new hash of a chunk while it is merged: the hash, its sorted
# copy, its place in the merged hashes, and its place among the hashes of the
# store's cell IDs and whether it is one of them.
_NEW_HASH_BYTES = 40
# Bytes of each of the store's cell IDs while a block of them is hashed,
# beside the block as Zarr returns it, the IDs' Python strings and their UTF-8
# characters: each ID's reference, the header of its UTF-8 bytes, the pointer
# to them, their length and their place in the list that hashing keeps, its
# hash and the hash's shifted copy that pandas mixes in.
_HASHED_BYTES_PER_STORE_ID = 8 + 33 + 8 + 8 + 8 + 8 + 8
# The most bytes of a Python string beside its characters, which take up to
# four bytes each.
_STRING_HEADER_BYTES = 60
# The store's cell IDs that become Python strings and are hashed at once; a
# read of a band of chunks holds more of them as fixed-width text, and an
# array without chunks is read this many at a time.
_HASHED_IDS = 1 << 16
# What a refusal of the hashes of barcodes that match no cell suggests.
_UNMATCHED_REMEDY = (
    "check barcode_prefix, barcode_suffix, barcode_pattern and cell_selection"
)
# The store's cell IDs are hashed whatever the selection, so a refusal of them
# suggests no smaller call.
_STORE_REMEDY = ""
# The named group of a barcode pattern that holds the barcode.
_BARCODE_GROUP = "barcode"
# What the options of an import may be, for their errors.
_TEXT_OPTIONS = "text, a sequence of text, or None"
_PATTERN_OPTIONS = "a regular expression, a sequence of regular expressions, or None"


def barcode_pattern(value: Any, name: str = "barcode_pattern") -> str | None:
    """Check a barcode pattern and return it, or None.

    Raises:
        TypeError: If the pattern is not text or None.
        ValueError: If it is not a regular expression with a named group
            ``barcode``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a regular expression or None")
    try:
        compiled = re.compile(value)
    except re.error as error:
        raise ValueError(
            f"{name} {quoted_value(value)} is not a valid regular expression: {error}"
        ) from None
    if _BARCODE_GROUP not in compiled.groupindex:
        raise ValueError(
            f"{name} {quoted_value(value)} has no named group 'barcode'; name the part of "
            "the barcode to keep, such as r'_(?P<barcode>[ACGT]{16})-1$'"
        )
    return value


@dataclass(frozen=True, slots=True)
class BarcodeFile:
    """One file of an import and how its barcodes name cells.

    Attributes:
        path: The file, or the folder of files that one source writes.
        sha256: The SHA-256 of its bytes, or of the files of a folder, which
            identifies it.
        barcode_prefix: Text before each barcode in the cell IDs, or None.
        barcode_suffix: Text after each barcode in the cell IDs, or None.
        barcode_pattern: A regular expression whose ``barcode`` group is the
            part of each barcode that names the cell, or None for the whole
            barcode.
        position: The file's position in the listing of the import's paths,
            which options given per file follow; not part of its identity.
    """

    path: str | os.PathLike[str]
    sha256: str
    barcode_prefix: str | None
    barcode_suffix: str | None
    barcode_pattern: str | None
    position: int = 0

    def record(self) -> dict[str, Any]:
        """Return the file's identity: its bytes and how its barcodes name cells."""
        return {
            "sha256": self.sha256,
            "barcode_prefix": self.barcode_prefix,
            "barcode_suffix": self.barcode_suffix,
            "barcode_pattern": self.barcode_pattern,
        }

    def cell_ids(self, barcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the cell ID that each barcode names, and whether it names one.

        The ID is the prefix, the part of the barcode that the pattern's
        ``barcode`` group matches, or the whole barcode without a pattern,
        and the suffix. A barcode that the pattern does not match names no
        cell; its ID is the whole barcode with the prefix and the suffix, so
        that such barcodes can still be counted.

        Args:
            barcodes: Barcodes as the file holds them, one per row.

        Returns:
            The IDs as an object array of text, and a boolean array that is
            False where a barcode names no cell.
        """
        texts = pd.Series(np.asarray(barcodes, dtype=object), dtype=object)
        named = np.ones(len(texts), dtype=bool)
        core = texts
        if self.barcode_pattern is not None:
            extracted = texts.str.extract(self.barcode_pattern, expand=True)[
                _BARCODE_GROUP
            ]
            named = extracted.notna().to_numpy()
            # A barcode that the pattern does not match is counted whole.
            core = extracted.where(named, texts)
        identifiers = core
        if self.barcode_prefix is not None:
            identifiers = self.barcode_prefix + identifiers
        if self.barcode_suffix is not None:
            identifiers = identifiers + self.barcode_suffix
        return identifiers.to_numpy(dtype=object), named


def derived_id_bytes(file: BarcodeFile, n_rows: int, n_chars: int) -> int:
    """Return the most bytes that :meth:`BarcodeFile.cell_ids` holds for a file's barcodes.

    A prefix, a suffix and a pattern each make a copy of every barcode, so
    a file with all three holds three copies while its IDs are derived, and
    a file with none holds a copy of the pointers alone.

    Args:
        file: The file, whose options derive the IDs.
        n_rows: The barcodes.
        n_chars: The characters of the barcodes, or a bound on them.
    """
    affixes = [
        option for option in (file.barcode_prefix, file.barcode_suffix) if option
    ]
    per_barcode = _COPY_BYTES_PER_BARCODE + sum(
        _AFFIXED_BYTES_PER_BARCODE + len(affix) for affix in affixes
    )
    n_copies = len(affixes)
    if file.barcode_pattern is not None:
        per_barcode += _EXTRACTED_BYTES_PER_BARCODE
        n_copies += 1
    return int(n_rows) * per_barcode + n_copies * int(n_chars)


def join_bytes(file: BarcodeFile, n_rows: int, n_chars: int) -> int:
    """Return the most bytes that :meth:`CellBarcodeJoin.cells` holds for a file's barcodes.

    These are the bytes beside the barcodes themselves and what the join
    charges to its ledger: the IDs that the file's options derive, as
    :func:`derived_id_bytes` counts them, and what matching each barcode
    holds, its encoded text included.

    Args:
        file: The file, whose options derive the IDs.
        n_rows: The barcodes.
        n_chars: The characters of the barcodes, or a bound on them.
    """
    return (
        derived_id_bytes(file, n_rows, n_chars)
        + _MATCH_BYTES_PER_BARCODE * int(n_rows)
        + int(n_chars)
    )


@functools.cache
def _index_copies_references() -> bool:
    """Whether pandas copies an object array's references into an index built on it.

    pandas 3 copies them; pandas 2 keeps the array that it is given.
    """
    values = np.asarray(["a"], dtype=object)
    return not np.shares_memory(pd.Index(values, dtype=object).to_numpy(), values)


def lookup_bytes(n_cells: int) -> int:
    """Return the bytes of the join's lookup of ``n_cells`` cell IDs, beside the IDs' text.

    The lookup is a pandas index of the IDs, which hashes every ID into a
    table on its first use, as
    :func:`~cytearc.utils.memory_budget.hash_table_bytes` counts, and copies
    their references where pandas copies an array into an index, and a flag
    for each cell, whether a file named it.
    """
    copied = object_array_bytes(n_cells) if _index_copies_references() else 0
    return hash_table_bytes(n_cells) + copied + int(n_cells)


def _listed(value: Any, n_files: int, name: str, what: str, expected: str) -> list[Any]:
    """Return one value per file: ``value`` for every file, or its items.

    ``expected`` says what the option may be, for the error.
    """
    if value is None or isinstance(value, str):
        return [value] * n_files
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        items = list(value)
        if len(items) != n_files:
            raise ValueError(
                f"{name} lists {len(items)} values for {n_files} {what}s; pass "
                f"one per {what}, or one for every {what}"
            )
        return items
    raise TypeError(f"{name} must be {expected}")


def texts_per_file(value: Any, n_files: int, name: str, what: str) -> list[str | None]:
    """Return one text or None per file: ``value`` for every file, or its items.

    An empty text is None, as it adds nothing.

    Args:
        value: Text, a sequence of text and None with one item per file, or
            None.
        n_files: The files listed.
        name: The option, named in errors.
        what: What a file is, such as ``"clone file"``, for the errors.

    Raises:
        TypeError: If ``value`` or an item is neither text nor None.
        ValueError: If ``value`` lists a value for another number of files.
    """
    options = _listed(value, n_files, name, what, _TEXT_OPTIONS)
    for option in options:
        if option is not None and not isinstance(option, str):
            raise TypeError(f"{name} must be {_TEXT_OPTIONS}")
    # An empty prefix or suffix adds nothing, as None does, and identifies
    # the file alike.
    return [option or None for option in options]


def barcode_files(
    paths: Any,
    *,
    barcode_prefix: Any = None,
    barcode_suffix: Any = None,
    barcode_pattern: Any = None,
    what: str = "file",
    digest: Callable[[str | os.PathLike[str]], str] = file_sha256,
) -> tuple[BarcodeFile, ...]:
    """Check the files of an import and their options, in canonical order.

    ``barcode_prefix``, ``barcode_suffix`` and ``barcode_pattern`` each take
    one value for every file or one per file. Files are ordered by the
    SHA-256 that identifies them, then by prefix, suffix and pattern, so the
    same files listed in another order give the same import.

    Args:
        paths: A path, or a sequence of paths.
        barcode_prefix: Text before each barcode in the cell IDs.
        barcode_suffix: Text after each barcode in the cell IDs.
        barcode_pattern: A regular expression with a named group ``barcode``.
        what: What a file is, such as ``"clone file"``, for the errors.
        digest: Returns the SHA-256 that identifies a path: by default
            :func:`~cytearc.utils.digest.file_sha256` of a file's bytes; an
            importer whose sources may be folders, such as Zarr stores,
            passes :func:`~cytearc.utils.digest.path_sha256` or one that
            digests every file that a source reads.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If no file is given, an option lists a value for another
            number of files, a pattern has no ``barcode`` group, or a file is
            listed twice with the same prefix and suffix.
        FileNotFoundError: If a file does not exist.
    """
    if isinstance(paths, str | os.PathLike):
        listed: list[Any] = [paths]
    elif isinstance(paths, Sequence) and not isinstance(paths, bytes):
        listed = list(paths)
    else:
        raise TypeError("paths must be a path or a sequence of paths")
    if not listed:
        raise ValueError(f"paths must name at least one {what}")
    for path in listed:
        if not isinstance(path, str | os.PathLike):
            raise TypeError("paths must be a path or a sequence of paths")
    n_files = len(listed)
    prefixes = texts_per_file(barcode_prefix, n_files, "barcode_prefix", what)
    suffixes = texts_per_file(barcode_suffix, n_files, "barcode_suffix", what)
    patterns = [
        _checked_pattern(option)
        for option in _listed(
            barcode_pattern, n_files, "barcode_pattern", what, _PATTERN_OPTIONS
        )
    ]
    files = [
        BarcodeFile(
            path=path,
            sha256=digest(path),
            barcode_prefix=prefix,
            barcode_suffix=suffix,
            barcode_pattern=pattern,
            position=position,
        )
        for position, (path, prefix, suffix, pattern) in enumerate(
            zip(listed, prefixes, suffixes, patterns, strict=True)
        )
    ]
    keys = [
        (file.sha256, file.barcode_prefix or "", file.barcode_suffix or "")
        for file in files
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"A {what} is listed twice with the same barcode_prefix and "
            "barcode_suffix; list each file once, or give each listing its own "
            "barcode_prefix"
        )
    order = sorted(
        range(n_files),
        key=lambda position: (*keys[position], files[position].barcode_pattern or ""),
    )
    return tuple(files[position] for position in order)


def _checked_pattern(value: Any) -> str | None:
    """Check one file's pattern; ``barcode_files`` names its option alike."""
    if value is not None and not isinstance(value, str):
        raise TypeError(f"barcode_pattern must be {_PATTERN_OPTIONS}")
    return barcode_pattern(value)


def barcode_example(barcode: str, cell_id: str, named: bool = True) -> str:
    """Show a file's barcode beside the cell ID it derives, as a diagnostic's example.

    A barcode whose ID is the barcode itself shows once, and a barcode that
    the file's pattern does not match, which names no cell, says so.
    """
    if not named:
        return f"{barcode} -> (barcode_pattern does not match)"
    if cell_id == barcode:
        return barcode
    return f"{barcode} -> {cell_id}"


def _store_block_rows(store_ids: Any) -> int:
    """The store's cell IDs that a block of the pass over them reads: a band of chunks."""
    geometry = array_geometry(store_ids)
    return _HASHED_IDS if geometry is None else geometry.axisChunk(0)


def _store_read_bytes(store_ids: Any, n_rows: int) -> int:
    """What reading ``n_rows`` of the store's cell IDs holds.

    A stored array is read as :func:`~cytearc.metadata.rows.read_rows` reads
    a band of chunks; an array without chunks, such as one given in memory,
    holds the rows alone.
    """
    geometry = array_geometry(store_ids)
    if geometry is None:
        return n_rows * int(np.dtype(store_ids.dtype).itemsize)
    return row_read_bytes(geometry, n_rows)


def id_hashing_bytes(n_ids: int, characters: int, widest: int) -> int:
    """Return what hashing ``n_ids`` of the store's cell IDs holds, beside them as fixed-width text.

    Each ID becomes a Python string, as
    :func:`~cytearc.utils.memory_budget.python_text_bytes` counts it, which
    hashing encodes as UTF-8, up to one byte a character for ASCII text, two
    to U+07FF, three to U+FFFF and four beyond, beside what it keeps for each
    ID.

    Args:
        n_ids: The IDs hashed at once.
        characters: Their characters.
        widest: The widest code point of their characters.
    """
    width = next(
        width
        for limit, width in ((0x7F, 1), (0x7FF, 2), (0xFFFF, 3), (0x10FFFF, 4))
        if widest <= limit
    )
    utf8 = width * characters
    return (
        n_ids * _HASHED_BYTES_PER_STORE_ID
        + python_text_bytes(n_ids, characters, widest=widest)
        + utf8
    )


def store_pass_bytes(store_ids: Any, n_selected: int) -> int:
    """Return the most bytes that the join's pass over the store's cell IDs holds.

    The join charges this when it starts, unless the selection holds every
    cell of the store, and runs the pass once a barcode needs it. It keeps the 64-bit hash of each of the store's cell IDs
    and reads them a band of chunks at a time, as
    :func:`~cytearc.metadata.rows.read_rows` charges a read, then hashes
    65,536 of them at a time beside the band, each as a Python string of up
    to four bytes a character, with what hashing it holds.

    Args:
        store_ids: The store's array of cell IDs, as fixed-width text.
        n_selected: The selected cells.
    """
    n_ids = int(store_ids.shape[0])
    if n_selected == n_ids:
        return 0
    rows = min(n_ids, _store_block_rows(store_ids))
    hashed = min(rows, _HASHED_IDS)
    itemsize = int(np.dtype(store_ids.dtype).itemsize)
    reading = _store_read_bytes(store_ids, rows)
    # A character takes up to four bytes in a Python string and four in UTF-8.
    hashing = hashed * (
        _HASHED_BYTES_PER_STORE_ID + _STRING_HEADER_BYTES + 2 * itemsize
    )
    return 8 * n_ids + max(reading, rows * itemsize + hashing)


def _hashes(values: np.ndarray) -> np.ndarray:
    """The 64-bit hash of each text of an object array, as the join counts barcodes."""
    hashed: np.ndarray = np.asarray(
        pd.util.hash_array(values, categorize=False), dtype=np.uint64
    )
    return hashed


def _first_rows(identifiers: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The rows of ``rows`` that hold the first of each distinct ID, in their order."""
    repeated = pd.Index(identifiers[rows], dtype=object).duplicated()
    first: np.ndarray = rows[~np.asarray(repeated)]
    return first


def _members(hashes: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Whether each hash is one of ``held``, ascending and distinct."""
    if not len(held):
        return np.zeros(len(hashes), dtype=bool)
    at = np.minimum(np.searchsorted(held, hashes), len(held) - 1)
    found: np.ndarray = held[at] == hashes
    return found


class CellBarcodeJoin:
    """Match the barcodes of files to selected cells and count those that match none.

    A barcode that matches no selected cell is either unmatched, naming no
    cell of the store, or unselected, naming a cell of the store outside the
    selection, such as one that QC filtering removed. The join tells them
    apart by the 64-bit hashes of the store's cell IDs, which it reads once,
    the first time a barcode that its file's options name matches no selected
    cell, and charges what that pass holds when it starts, so that what an
    import sizes from its ledger's room afterwards leaves room for it; a
    selection of every cell of the store leaves no cell outside it, so it
    reads none and charges nothing for them. A barcode that a file's
    pattern does not match names no cell, and is unmatched; it is counted
    apart from a barcode that names the same ID, so the counts do not depend
    on the order or the chunks of the rows.

    The join holds a lookup of the selected cells' IDs, whether each cell was
    named, the hashes of the store's cell IDs once read, and the hashes of
    the distinct barcodes of each kind, and charges them to the caller's
    ledger as they grow: the lookup, as :func:`lookup_bytes` counts it, when
    the join starts, with what its pass over the store's IDs will hold, as
    :func:`store_pass_bytes` bounds it, which the pass's own charges of the
    store's hashes and a block of their IDs at a time replace; and each
    chunk's new hashes before they merge into those held. IDs given as an
    object array of text are the caller's to charge; IDs given as fixed-width
    text become the Python strings that the lookup keeps, which the join
    charges. What matching a chunk's barcodes holds for a moment, as
    :func:`join_bytes` states it, is the caller's chunk to charge. Barcodes
    are counted by the cell ID they name, so a barcode that two files name
    with different prefixes counts twice.

    Args:
        cell_ids: The IDs of the selected cells, in selection order.
        ledger: The memory ledger of the import.
        store_ids: The IDs of every cell of the store, as fixed-width text,
            such as the store's stored array of them, read in blocks.

    Raises:
        MemoryBudgetError: If the lookup does not fit the ledger's budget.
        TypeError: If the IDs are neither fixed-width text nor an object
            array, or the store's IDs are not fixed-width text.
        ValueError: If a cell ID repeats.
    """

    def __init__(
        self, cell_ids: np.ndarray, *, ledger: MemoryLedger, store_ids: Any
    ) -> None:
        given = np.asarray(cell_ids)
        n_cells = len(given)
        self._ledger = ledger
        if np.dtype(store_ids.dtype).kind != "U" or len(store_ids.shape) != 1:
            raise TypeError(
                "The store's cell IDs must be one-dimensional fixed-width text, not "
                f"{store_ids.dtype}"
            )
        self._store_ids = store_ids
        if given.dtype.kind == "U":
            # The lookup keeps each ID as the Python string it becomes, which
            # is measured from each ID's length.
            ledger.peak("the selected cells' IDs as text", 8 * n_cells)
            characters = int(np.char.str_len(given).sum()) if n_cells else 0
            widest = int(given.view(np.uint32).max()) if given.size else 0
            ledger.hold(
                "the selected cells' IDs as text",
                object_array_bytes(n_cells)
                + python_text_bytes(n_cells, characters, widest=widest),
                remedy="pass a smaller cell_selection",
            )
        elif given.dtype != object:
            raise TypeError(f"The selected cells' IDs must be text, not {given.dtype}")
        ids = np.asarray(given, dtype=object)
        ledger.hold(
            "the join's lookup of the selected cells' IDs",
            lookup_bytes(n_cells),
            remedy="pass a smaller cell_selection",
        )
        # An object index keeps the IDs as the Python strings counted above;
        # with pyarrow installed, pandas would otherwise copy them to Arrow.
        self._lookup = pd.Index(ids, dtype=object)
        if not self._lookup.is_unique:
            raise ValueError(
                "The selected cells' IDs repeat, so barcodes cannot name them"
            )
        # A selection of every cell of the store leaves none outside it.
        # Otherwise what the pass over the store's IDs will hold is charged
        # now, so that what the import sizes from the ledger's room leaves
        # room for it, and the pass runs once a barcode needs it.
        self._store_hashes: np.ndarray | None = None
        self._reserved = 0
        if n_cells == int(store_ids.shape[0]):
            self._store_hashes = np.empty(0, dtype=np.uint64)
        else:
            self._reserved = ledger.hold(
                "the pass over the store's cell IDs",
                store_pass_bytes(store_ids, n_cells),
                remedy=_STORE_REMEDY,
            )
        self._named = np.zeros(n_cells, dtype=bool)
        self._unmatched = np.empty(0, dtype=np.uint64)
        self._unselected = np.empty(0, dtype=np.uint64)
        self._unmatched_examples: list[str] = []
        self._unselected_examples: list[str] = []
        self.n_rows = 0
        self.n_unmatched_rows = 0

    @property
    def n_file_barcodes(self) -> int:
        """The distinct barcodes of the files, counted by the cell ID they name."""
        return (
            int(np.count_nonzero(self._named))
            + len(self._unmatched)
            + len(self._unselected)
        )

    @property
    def n_unmatched_barcodes(self) -> int:
        """The distinct barcodes that name no cell of the store."""
        return len(self._unmatched)

    @property
    def n_unselected_barcodes(self) -> int:
        """The distinct barcodes that name a cell of the store outside the selection."""
        return len(self._unselected)

    @property
    def unmatched_examples(self) -> list[str]:
        """Up to five barcodes that name no cell of the store, as :func:`barcode_example` shows them."""
        return list(self._unmatched_examples)

    @property
    def unselected_examples(self) -> list[str]:
        """Up to five barcodes of cells outside the selection, as :func:`barcode_example` shows them."""
        return list(self._unselected_examples)

    def _read_store_hashes(self) -> np.ndarray:
        """Read the store's cell IDs a block at a time and return their hashes, ascending."""
        store_ids = self._store_ids
        n_ids = int(store_ids.shape[0])
        # The pass's own charges replace what the join charged for it.
        self._ledger.release(self._reserved)
        self._reserved = 0
        item = "the hashes of the store's cell IDs"
        self._ledger.hold(item, 8 * n_ids, remedy=_STORE_REMEDY)
        hashes = np.empty(n_ids, dtype=np.uint64)
        step = _store_block_rows(store_ids)
        for start in range(0, n_ids, step):
            stop = min(n_ids, start + step)
            self._ledger.peak(
                "a block of the store's cell IDs",
                _store_read_bytes(store_ids, stop - start),
                remedy=_STORE_REMEDY,
            )
            # The read decodes on the ledger's codec threads.
            with zarr_decoding_threads(self._ledger.codec_threads):
                block = np.asarray(store_ids[start:stop])
            for first in range(0, len(block), _HASHED_IDS):
                part = block[first : first + _HASHED_IDS]
                # Each ID becomes a Python string, which hashing encodes as
                # UTF-8, measured from each ID's length.
                self._ledger.peak(
                    "a block of the store's cell IDs",
                    block.nbytes + 8 * len(part),
                    remedy=_STORE_REMEDY,
                )
                characters = int(np.char.str_len(part).sum())
                widest = int(part.view(np.uint32).max()) if part.size else 0
                self._ledger.peak(
                    "a block of the store's cell IDs",
                    block.nbytes + id_hashing_bytes(len(part), characters, widest),
                    remedy=_STORE_REMEDY,
                )
                at = start + first
                hashes[at : at + len(part)] = _hashes(part.astype(object))
                del part
            del block
        hashes.sort()
        return hashes

    def _note_examples(
        self,
        examples: list[str],
        held: np.ndarray,
        kind: np.ndarray,
        hashes: np.ndarray,
        shown: Callable[[int], str],
    ) -> None:
        """Add examples of one kind from a chunk's new barcodes, in file order.

        Each counted barcode gives one example: one counted by an earlier
        chunk has given its own, and a barcode that the pattern does not
        match counts once with a barcode that names its ID among the barcodes
        of no cell.
        """
        if len(examples) == EXAMPLES or not kind.any():
            return
        shown_hashes: set[int] = set()
        for position in np.flatnonzero(kind & ~_members(hashes, held)).tolist():
            if len(examples) == EXAMPLES:
                break
            value = int(hashes[position])
            if value not in shown_hashes:
                shown_hashes.add(value)
                examples.append(shown(position))

    def cells(self, file: BarcodeFile, barcodes: np.ndarray) -> np.ndarray:
        """Return the selected cell that each barcode names, -1 for none.

        Args:
            file: The file that holds the barcodes.
            barcodes: Barcodes as the file holds them, one per row.

        Returns:
            Each row's cell, as its position in the selection, as int64.

        Raises:
            MemoryBudgetError: If the hashes of the store's cell IDs, or of
                the barcodes that matched no selected cell, outgrow the
                ledger's budget.
        """
        texts = np.asarray(barcodes, dtype=object)
        identifiers_array, named = file.cell_ids(texts)
        positions = np.asarray(
            self._lookup.get_indexer(pd.Index(identifiers_array, dtype=object)),
            dtype=np.int64,
        )
        positions[~named] = -1
        matched = positions >= 0
        self._named[positions[matched]] = True
        self.n_rows += len(positions)
        if matched.all():
            return positions
        rows = np.flatnonzero(~matched)
        self.n_unmatched_rows += len(rows)
        # The first row of each distinct ID, in file order, the barcodes that
        # the pattern does not match apart from those that name a cell.
        if file.barcode_pattern is None:
            first = _first_rows(identifiers_array, rows)
        else:
            kinds = named[rows]
            first = np.sort(
                np.concatenate(
                    [
                        _first_rows(identifiers_array, rows[kinds]),
                        _first_rows(identifiers_array, rows[~kinds]),
                    ]
                )
            )
        del rows
        missing = identifiers_array[first]
        # The new hashes merge into those held, which keep room for their
        # copy in the next merge, and are looked up among the store's, before
        # they are made.
        self._ledger.peak(
            "the hashes of the barcodes that match no selected cell",
            _NEW_HASH_BYTES * len(missing),
            remedy=_UNMATCHED_REMEDY,
        )
        # The IDs are distinct already, so they are hashed as they are.
        hashes = _hashes(missing)
        missing_named = named[first]
        unselected = np.zeros(len(missing), dtype=bool)
        if missing_named.any():
            if self._store_hashes is None:
                self._store_hashes = self._read_store_hashes()
            unselected[missing_named] = _members(
                hashes[missing_named], self._store_hashes
            )
        unmatched = ~unselected

        def shown(position: int) -> str:
            return barcode_example(
                str(texts[first[position]]),
                str(missing[position]),
                bool(missing_named[position]),
            )

        self._note_examples(
            self._unmatched_examples, self._unmatched, unmatched, hashes, shown
        )
        self._note_examples(
            self._unselected_examples, self._unselected, unselected, hashes, shown
        )
        held = len(self._unmatched) + len(self._unselected)
        self._unmatched = union_sorted(self._unmatched, hashes[unmatched])
        self._unselected = union_sorted(self._unselected, hashes[unselected])
        self._ledger.hold(
            "the hashes of the barcodes that match no selected cell",
            _UNMATCHED_BYTES_PER_BARCODE
            * (len(self._unmatched) + len(self._unselected) - held),
            remedy=_UNMATCHED_REMEDY,
        )
        return positions

    def diagnostics(self) -> dict[str, Any]:
        """Return the join's diagnostics, as an importer saves them.

        ``fileBarcodes`` counts the files' distinct barcodes by the cell ID
        they name, ``unmatchedBarcodes`` those that name no cell of the store
        and ``unselectedBarcodes`` those that name a cell of the store outside
        the selection, with up to five examples of each, in the order the
        importer joined them.
        """
        return {
            "fileBarcodes": self.n_file_barcodes,
            "unmatchedBarcodes": self.n_unmatched_barcodes,
            "unselectedBarcodes": self.n_unselected_barcodes,
            "unmatchedBarcodeExamples": self.unmatched_examples,
            "unselectedBarcodeExamples": self.unselected_examples,
        }


def unmatched_barcodes_message(diagnostics: Mapping[str, Any], files: str) -> str:
    """Name the barcodes of the files that match no cell of the store.

    Args:
        diagnostics: The join's diagnostics, as an importer saved them.
        files: The files, such as ``"the clone file"``.
    """
    return (
        f"{int(diagnostics['unmatchedBarcodes'])} of "
        f"{int(diagnostics['fileBarcodes'])} barcodes of {files} match no cell of "
        "the store, for example "
        + ", ".join(
            quoted_value(str(value))
            for value in diagnostics["unmatchedBarcodeExamples"]
        )
        + ". Check barcode_prefix, barcode_suffix and barcode_pattern against the "
        "store's cell IDs."
    )


def unmatched_barcode_rules(files: str) -> tuple[DiagnosticRule, ...]:
    """Return the join's warning: barcodes of the files that match no cell of the store.

    Barcodes of cells outside the selection give no warning. ``files`` names
    the files, such as ``"the clone files"``.
    """
    return (
        DiagnosticRule(
            "unmatchedBarcodes",
            applies=lambda diagnostics: int(diagnostics["unmatchedBarcodes"]) > 0,
            message=lambda diagnostics: unmatched_barcodes_message(diagnostics, files),
        ),
    )
