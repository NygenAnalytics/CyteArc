"""Join the barcodes of imported files to the selected cells of a store.

Files that importers read, such as guide calls or clone tables, name cells by
barcode, while a store names them by cell ID. A barcode names the cell whose ID
is the file's prefix, then the barcode, then the file's suffix; a pattern can
first keep the part of the barcode that its ``barcode`` group matches. Every
importer joins its files through ``CellBarcodeJoin`` and orders and identifies
them through ``barcode_files``, so they share the rules, the counts of
barcodes that match no selected cell, and the warning about them. The join
charges what it holds to the importer's memory ledger.
"""

import functools
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...utils.arrays import union_sorted
from ...utils.digest import file_sha256
from ...utils.memory_budget import (
    MemoryLedger,
    hash_table_bytes,
    object_array_bytes,
    python_text_bytes,
)
from ...utils.warnings import warn
from ...utils.arguments import quoted_value

__all__ = [
    "LOOKUP_BYTES_PER_CELL",
    "UNMATCHED_EXAMPLES",
    "BarcodeFile",
    "CellBarcodeJoin",
    "barcode_files",
    "barcode_pattern",
    "derived_id_bytes",
    "join_bytes",
    "lookup_bytes",
    "texts_per_file",
    "unmatched_barcodes_warning",
    "warn_unmatched_barcodes",
]

#: The barcodes that match no selected cell named in an artifact's attributes
#: and in the warning.
UNMATCHED_EXAMPLES = 5
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
# copy and its place in the merged hashes.
_NEW_HASH_BYTES = 24
# What a refusal of the hashes of barcodes that match no cell suggests.
_UNMATCHED_REMEDY = (
    "check barcode_prefix, barcode_suffix, barcode_pattern and cell_selection"
)
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


class CellBarcodeJoin:
    """Match the barcodes of files to selected cells and count those that match none.

    The join holds a lookup of the selected cells' IDs, whether each cell was
    named, and the 64-bit hashes of the distinct barcodes that matched no
    cell, and charges them to the caller's ledger as they grow: the lookup,
    as :func:`lookup_bytes` counts it, when the join starts, and each chunk's
    new hashes before they merge into those held. IDs given as an object
    array of text are the caller's to charge; IDs given as fixed-width text
    become the Python strings that the lookup keeps, which the join charges.
    What matching a chunk's barcodes holds for a moment, as
    :func:`join_bytes` states it, is the caller's chunk to charge. Barcodes
    are counted by the cell ID they name, so a barcode that two files name
    with different prefixes counts twice. A barcode that a file's pattern
    does not match names no cell.

    Args:
        cell_ids: The IDs of the selected cells, in selection order.
        ledger: The memory ledger of the import.

    Raises:
        MemoryBudgetError: If the lookup does not fit the ledger's budget.
        TypeError: If the IDs are neither fixed-width text nor an object array.
        ValueError: If a cell ID repeats.
    """

    def __init__(self, cell_ids: np.ndarray, *, ledger: MemoryLedger) -> None:
        given = np.asarray(cell_ids)
        n_cells = len(given)
        self._ledger = ledger
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
        self._named = np.zeros(n_cells, dtype=bool)
        self._unmatched = np.empty(0, dtype=np.uint64)
        self._examples: list[str] = []
        self.n_rows = 0
        self.n_unmatched_rows = 0

    @property
    def n_file_barcodes(self) -> int:
        """The distinct barcodes of the files, counted by the cell ID they name."""
        return int(np.count_nonzero(self._named)) + len(self._unmatched)

    @property
    def n_unmatched_barcodes(self) -> int:
        """The distinct barcodes that matched no selected cell."""
        return len(self._unmatched)

    @property
    def examples(self) -> list[str]:
        """Up to five barcodes that matched no selected cell, as the files hold them."""
        return list(self._examples)

    def cells(self, file: BarcodeFile, barcodes: np.ndarray) -> np.ndarray:
        """Return the selected cell that each barcode names, -1 for none.

        Args:
            file: The file that holds the barcodes.
            barcodes: Barcodes as the file holds them, one per row.

        Returns:
            Each row's cell, as its position in the selection, as int64.

        Raises:
            MemoryBudgetError: If the hashes of the barcodes that matched no
                cell outgrow the ledger's budget.
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
        if not matched.all():
            unmatched = ~matched
            self.n_unmatched_rows += int(np.count_nonzero(unmatched))
            missing = pd.unique(identifiers_array[unmatched])
            # The new hashes merge into those held, which keep room for their
            # copy in the next merge, before they are made.
            self._ledger.peak(
                "the hashes of the barcodes that match no selected cell",
                _NEW_HASH_BYTES * len(missing),
                remedy=_UNMATCHED_REMEDY,
            )
            # The barcodes are distinct already, so they are hashed as they are.
            hashes = np.asarray(
                pd.util.hash_array(missing, categorize=False), dtype=np.uint64
            )
            held = len(self._unmatched)
            self._unmatched = union_sorted(self._unmatched, hashes)
            self._ledger.hold(
                "the hashes of the barcodes that match no selected cell",
                _UNMATCHED_BYTES_PER_BARCODE * (len(self._unmatched) - held),
                remedy=_UNMATCHED_REMEDY,
            )
            if len(self._examples) < UNMATCHED_EXAMPLES:
                for barcode in pd.unique(texts[unmatched]):
                    if len(self._examples) == UNMATCHED_EXAMPLES:
                        break
                    if str(barcode) not in self._examples:
                        self._examples.append(str(barcode))
        return positions

    def attributes(self) -> dict[str, Any]:
        """Return the counts that an artifact saves and the warning names."""
        return {
            "n_file_barcodes": self.n_file_barcodes,
            "n_unmatched_barcodes": self.n_unmatched_barcodes,
            "unmatched_barcode_examples": self.examples,
        }


def unmatched_barcodes_warning(attributes: Mapping[str, Any], files: str) -> str | None:
    """Name the barcodes of the files that match no selected cell, or return None.

    An import warns with this message, and so does a saved import that is
    reused, from the attributes it saved.

    Args:
        attributes: ``n_file_barcodes``, ``n_unmatched_barcodes`` and
            ``unmatched_barcode_examples``.
        files: The files, such as ``"the call file"``.
    """
    n_unmatched = int(attributes["n_unmatched_barcodes"])
    if not n_unmatched:
        return None
    return (
        f"{n_unmatched} of {int(attributes['n_file_barcodes'])} barcodes of "
        f"{files} match no selected cell, for example "
        + ", ".join(
            quoted_value(str(value))
            for value in attributes["unmatched_barcode_examples"]
        )
        + ". Check barcode_prefix and cell_selection."
    )


def warn_unmatched_barcodes(attributes: Mapping[str, Any], files: str) -> None:
    """Warn about the barcodes of the files that match no selected cell, if any."""
    message = unmatched_barcodes_warning(attributes, files)
    if message is not None:
        warn(message)
