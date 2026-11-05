"""Read cell phylogenies from Newick and TreeData files.

Lineage recorders such as PEtracer and DNA Typewriter publish the trees that
their authors reconstructed: Newick text, plain or gzip-compressed, with one
tree or several per file, or TreeData files, which are AnnData files with
trees in ``obst``. Each tree's leaves name cells.

Newick text is scanned once, without recursion, by a compiled loop that holds
its open parentheses on an explicit stack, so a tree of a million leaves on
one 59 MB line, or a caterpillar a million nodes deep, reads like any other.
The scan skips bracket comments, nested or not, such as ``[&R]`` or NHX and
BEAST annotations, and reads quoted labels with ``''`` for a quote,
polytomies, single-child nodes, missing, zero and negative branch lengths, a
length on the root, and several trees ending in ``;``. Labels are kept as the
file writes them, underscores included; whitespace around a label is dropped.

TreeData stores ``obst`` as one JSON string in ``.h5td`` files and, since
treedata 0.3.0, as one group per tree in ``.zarr`` stores, with each node
attribute in a column; earlier ``.zarr`` stores hold the JSON string. Each
node's ``time`` attribute and each edge's ``length`` attribute are read, and
nothing else.

Each file's nodes come out tree by tree, each tree in preorder, so a parent
comes before its children. These functions never touch storage.
"""

import math
import os
import re
import sys
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.files import GZIP_SIGNATURE, HDF5_SIGNATURE, file_signature, open_binary
from ..utils.numba import njit
from ..utils.arguments import quoted_value
from ..utils.memory_budget import MemoryLedger
from ._treedata import (
    TreeGraph,
    h5td_tree_bytes,
    h5td_tree_graphs,
    is_zarr_store,
    tree_in_preorder,
    zarr_tree_graphs,
    zarr_tree_nodes,
)

__all__ = [
    "FORMATS",
    "FileTrees",
    "TreeFileLayout",
    "read_trees",
    "tree_container",
    "tree_file_layout",
]

#: The values of ``format``.
FORMATS: tuple[str, ...] = ("newick", "treedata")
_BYTE_ORDER_MARK = b"\xef\xbb\xbf"
# Text is counted and read in blocks of this many bytes.
_BLOCK_BYTES = 1 << 20
# Errors quote this many characters on each side of a position.
_CONTEXT_CHARACTERS = 24
# An unquoted label of an internal node that reads as a number is a support
# value, as treedata reads it, rather than a name.
_SUPPORT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

# Flags of a scanned Newick node.
_INTERNAL = 1
_HAS_LABEL = 2
_QUOTED_LABEL = 4
_HAS_LENGTH = 8
_QUOTED_LENGTH = 16

# What the scan of Newick text holds for each node it may find: its parent,
# where its label and length start and end, and its flags, 41 bytes; and for
# each '(' a slot of the stack of open nodes, and for each ';' a tree's end.
_SCAN_BYTES_PER_NODE = 41
_STACK_BYTES_PER_GROUP = 8
# Names and lengths are converted this many at a time.
_BATCH = 1 << 16
# Building a file's trees holds, for each node, its parent, length, time and
# whether it is a leaf, 25 bytes; each length as a Python float in a list
# while the lengths are converted, 32; and the slot of its name, 8.
_TREE_BYTES_PER_NODE = 65
# A name of ASCII text holds a Python string of 49 bytes and one per
# character; other text at most 80 bytes and four per byte of UTF-8.
_ASCII_NAME_BYTES = 49
_TEXT_NAME_BYTES = 80
# Each node of a TreeData tree read from Zarr holds its name, its time and
# length and its place in the lists that order the nodes, besides the text
# of its name.
_ZARR_BYTES_PER_NODE = 400

# Error codes of the scan, with what each means.
_SCAN_ERRORS = {
    1: "a comment '[' that never closes",
    2: "an unexpected ';' inside parentheses or after ':'",
    3: "a quoted label that never closes",
    4: "a second label for one node, or a label after its length",
    5: "no branch length after ':'",
    6: "an unexpected '(' after a label, a length or a closed group",
    7: "an unexpected ',' outside parentheses",
    8: "a ')' without its '('",
    9: "a second ':' for one node",
    10: "a tree whose parentheses never close",
}
# The scan's code for text that holds more nodes, groups or trees than its
# layout counted, which only a file that changed after its layout can.
_CHANGED = 11


@dataclass(frozen=True, slots=True)
class TreeFileLayout:
    """What a tree file holds, found before its trees are read.

    Attributes:
        path: The file, or the directory of a Zarr store.
        format: ``"newick"`` or ``"treedata"``.
        compression: ``"gzip"`` for gzip-compressed Newick text, else None.
        container: ``"hdf5"`` or ``"zarr"`` for TreeData, else None.
        n_text_bytes: Newick: the bytes of its text once decompressed.
            TreeData in HDF5: at most the bytes of its tree JSON, the file's
            bytes outside its fixed-size datasets. Zarr: 0.
        max_nodes: Newick: at most this many nodes, from its ``(``, ``,``
            and ``;``. Zarr with one group per tree: its nodes. Otherwise 0,
            since the nodes are counted once the JSON is read.
        n_groups: Newick: its ``(``, which bound the depth of its trees.
        max_trees: Newick: at most this many trees, from its ``;``.
        ascii_text: Newick: whether its text is ASCII, whose names cost a
            byte a character.
    """

    path: str
    format: str
    compression: str | None
    container: str | None
    n_text_bytes: int
    max_nodes: int
    n_groups: int = 0
    max_trees: int = 0
    ascii_text: bool = True

    def record(self) -> dict[str, Any]:
        """Return how the file is read, for the import's identity."""
        return {"format": self.format}

    @property
    def read_bytes(self) -> int:
        """The bytes that reading the file's trees holds at first.

        Newick text holds its bytes and what the scan holds for each node it
        may find; reading names and lengths is checked again once the scan
        has counted them. TreeData in HDF5 holds its JSON as bytes and as
        text before it is parsed. Zarr's arrays are counted as they are read.
        """
        if self.format == "newick":
            return (
                self.n_text_bytes
                + _SCAN_BYTES_PER_NODE * self.max_nodes
                + _STACK_BYTES_PER_GROUP * (self.n_groups + self.max_trees)
            )
        if self.container == "hdf5":
            return 2 * self.n_text_bytes
        return _ZARR_BYTES_PER_NODE * self.max_nodes

    @property
    def trees_bytes(self) -> int:
        """At most the bytes that the trees read from the file hold.

        Newick files bound them by their nodes and text, and Zarr stores by
        their nodes; the trees of an HDF5 file are counted from its JSON once
        it is read, so 0 stands for them here.
        """
        if self.format == "newick":
            # A name for every node, at most the text's bytes in all.
            if self.ascii_text:
                names = _ASCII_NAME_BYTES * self.max_nodes + self.n_text_bytes
            else:
                names = _TEXT_NAME_BYTES * self.max_nodes + 4 * self.n_text_bytes
            return _TREE_BYTES_PER_NODE * self.max_nodes + names
        if self.container == "hdf5":
            return 0
        return _ZARR_BYTES_PER_NODE * self.max_nodes

    @property
    def peak_bytes(self) -> int:
        """At most the bytes that reading the file holds at its peak, as known now."""
        if self.format == "newick":
            return self.read_bytes + self.trees_bytes
        return self.read_bytes


@dataclass(frozen=True, slots=True)
class FileTrees:
    """The trees of one file, tree by tree, each tree in preorder.

    Attributes:
        names: Each node's name as text, ``""`` for a node without one.
        parent: Each node's parent as a position among the file's nodes, -1
            for a root, as int64; a parent comes before its children.
        length: The length of the branch above each node, a root's own
            length in Newick, NaN where the file gives none, as float64.
        time: Each node's time as the file gives it, NaN where it gives none,
            as float64. Only TreeData stores times.
        leaf: Whether each node has no children.
        tree_offsets: Where each tree's nodes start, and the end, as int64.
        tree_keys: Each tree's key: its position from 1 in a Newick file,
            its ``obst`` key in a TreeData file.
    """

    names: np.ndarray
    parent: np.ndarray
    length: np.ndarray
    time: np.ndarray
    leaf: np.ndarray
    tree_offsets: np.ndarray
    tree_keys: tuple[str, ...]

    @property
    def n_nodes(self) -> int:
        return len(self.parent)

    @property
    def held_bytes(self) -> int:
        """The bytes that the trees hold: their arrays and names."""
        arrays = (self.parent, self.length, self.time, self.leaf, self.tree_offsets)
        return (
            sum(int(values.nbytes) for values in arrays)
            + 8 * len(self.names)
            + sum(map(sys.getsizeof, filter(None, self.names.tolist())))
        )


def tree_container(path: Path) -> str:
    """Return how a tree file holds its trees, from its bytes.

    Returns:
        ``"zarr"`` for a Zarr store, ``"hdf5"``, ``"gzip"`` or ``"text"``.

    Raises:
        ValueError: If the path is a directory but not a Zarr store.
        FileNotFoundError: If the path does not exist.
    """
    if path.is_dir():
        if is_zarr_store(path):
            return "zarr"
        raise ValueError(
            f"The tree path {quoted_value(os.fspath(path))} is a directory but not a Zarr "
            "store; pass a Newick file or a TreeData .h5td file or .zarr store"
        )
    signature = file_signature(path)
    if signature.startswith(HDF5_SIGNATURE):
        return "hdf5"
    if signature.startswith(GZIP_SIGNATURE):
        return "gzip"
    return "text"


def _newick_layout(path: Path, compression: str | None) -> TreeFileLayout:
    """Count a Newick file's text bytes and the marks that bound its nodes."""
    n_bytes = n_groups = n_commas = n_ends = 0
    ascii_text = True
    head = b""
    try:
        with open_binary(path) as handle:
            for block in iter(lambda: handle.read(_BLOCK_BYTES), b""):
                if len(head) < 64:
                    head += block[:64]
                n_bytes += len(block)
                n_groups += block.count(b"(")
                n_commas += block.count(b",")
                n_ends += block.count(b";")
                ascii_text = ascii_text and block.isascii()
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(
            f"The Newick file {quoted_value(os.fspath(path))} cannot be read: {error}"
        ) from None
    if head.lstrip(_BYTE_ORDER_MARK).lstrip().upper().startswith(b"#NEXUS"):
        raise ValueError(
            f"The tree file {quoted_value(os.fspath(path))} is a NEXUS file; export its trees "
            "as Newick, one tree per line, and import that file"
        )
    return TreeFileLayout(
        path=os.fspath(path),
        format="newick",
        compression=compression,
        container=None,
        n_text_bytes=n_bytes,
        # A root for each tree, a node for each '(' and one for each ','.
        max_nodes=n_groups + n_commas + n_ends + 1,
        n_groups=n_groups,
        max_trees=n_ends + 1,
        ascii_text=ascii_text,
    )


def _treedata_layout(path: Path, container: str) -> TreeFileLayout:
    if container == "hdf5":
        return TreeFileLayout(
            path=os.fspath(path),
            format="treedata",
            compression=None,
            container="hdf5",
            n_text_bytes=h5td_tree_bytes(os.fspath(path)),
            max_nodes=0,
        )
    return TreeFileLayout(
        path=os.fspath(path),
        format="treedata",
        compression=None,
        container="zarr",
        n_text_bytes=0,
        max_nodes=zarr_tree_nodes(os.fspath(path)),
    )


def tree_file_layout(
    path: str | os.PathLike[str], format: str | None = None
) -> TreeFileLayout:
    """Recognise a tree file and count what reading it will hold.

    A Zarr store or an HDF5 file holds TreeData trees; other files hold
    Newick text, plain or gzip-compressed. ``format`` names the format the
    file must have.

    Raises:
        ValueError: If ``format`` is unknown or does not fit the file, or the
            file is a NEXUS file, a directory that is not a Zarr store, or a
            TreeData file without ``obst``.
        FileNotFoundError: If the file does not exist.
    """
    if format is not None and format not in FORMATS:
        raise ValueError(
            f"format must be one of {', '.join(FORMATS)} or None, not {quoted_value(format)}"
        )
    resolved = Path(os.fspath(path))
    if not resolved.exists():
        raise FileNotFoundError(
            f"The tree file {quoted_value(os.fspath(path))} does not exist"
        )
    container = tree_container(resolved)
    detected = "treedata" if container in ("hdf5", "zarr") else "newick"
    if format is not None and format != detected:
        what = {
            "zarr": "a Zarr store",
            "hdf5": "an HDF5 file",
            "gzip": "gzip-compressed text",
            "text": "text",
        }[container]
        raise ValueError(
            f"The tree file {quoted_value(os.fspath(path))} is {what}, which holds "
            f"{detected} trees, not {format} trees; pass format={quoted_value(detected)} or "
            "None"
        )
    if detected == "treedata":
        return _treedata_layout(resolved, container)
    return _newick_layout(resolved, "gzip" if container == "gzip" else None)


def read_trees(layout: TreeFileLayout, *, ledger: MemoryLedger) -> FileTrees:
    """Read the trees of one file.

    Args:
        layout: The file, from :func:`tree_file_layout`.
        ledger: The caller's memory ledger, which each step of the read is
            checked against beside what it holds.

    Raises:
        ValueError: If the file is not well-formed, a tree is not a tree, or
            the file holds no tree.
        MemoryBudgetError: If reading the trees does not fit the ledger's
            budget.
    """
    held_bytes = ledger.held_bytes
    ledger.peak(f"reading the trees of {quoted_value(layout.path)}", layout.read_bytes)
    if layout.format == "newick":
        return _read_newick(layout, ledger=ledger)

    def check(what: str, held: int) -> None:
        ledger.refuse_beyond(what, held)

    if layout.container == "hdf5":
        graphs = h5td_tree_graphs(layout.path, check=check, held_bytes=held_bytes)
    else:
        graphs = zarr_tree_graphs(
            layout.path, check=check, held_bytes=held_bytes + layout.read_bytes
        )
    return _file_trees(graphs, layout.path)


# Newick


@njit(cache=True)
def _is_space(byte: int) -> bool:
    return byte == 32 or 9 <= byte <= 13


@njit(cache=True)
def _ends_label(byte: int) -> bool:
    # ( ) , : ; and [ end an unquoted label.
    return (
        byte == 40 or byte == 41 or byte == 44 or byte == 58 or byte == 59 or byte == 91
    )


@njit(cache=True)
def _scan_newick(  # noqa: C901 - one state machine over the text
    text: np.ndarray,
    parent: np.ndarray,
    label_start: np.ndarray,
    label_end: np.ndarray,
    length_start: np.ndarray,
    length_end: np.ndarray,
    flags: np.ndarray,
    stack: np.ndarray,
    tree_ends: np.ndarray,
) -> tuple[int, int, int, int]:
    """Scan Newick text into nodes in preorder, without recursion.

    A node is made when the scan meets it: a tree's root at its first mark,
    the first child of a group at its ``(`` and each further child at its
    ``,``, so parents come before their children. ``stack`` holds the open
    groups. Labels and lengths are recorded as positions in the text. The
    scan stops with code 11 rather than write past the end of an array.

    Returns:
        The nodes, the trees, an error code (0 for none) and the byte at which
        the error was found.
    """
    size = len(text)
    n_nodes = 0
    n_trees = 0
    depth = 0
    node = -1
    root = -1
    expect_length = False
    position = 0
    while True:
        while position < size and _is_space(text[position]):
            position += 1
        if position >= size:
            break
        byte = text[position]
        if byte == 91:
            # A comment, which may nest.
            level = 0
            end = position
            while end < size:
                if text[end] == 91:
                    level += 1
                elif text[end] == 93:
                    level -= 1
                    if level == 0:
                        break
                end += 1
            if end >= size:
                return n_nodes, n_trees, 1, position
            position = end + 1
            continue
        if byte == 59:
            if root < 0:
                position += 1
                continue
            if depth > 0 or expect_length:
                return n_nodes, n_trees, 2, position
            if n_trees == len(tree_ends):
                return n_nodes, n_trees, 11, position
            tree_ends[n_trees] = n_nodes
            n_trees += 1
            root = -1
            node = -1
            position += 1
            continue
        if root < 0:
            if n_nodes == len(parent):
                return n_nodes, n_trees, 11, position
            parent[n_nodes] = -1
            flags[n_nodes] = 0
            root = n_nodes
            node = n_nodes
            n_nodes += 1
        if byte == 39 or not _ends_label(byte):
            start = position
            quoted = byte == 39
            if quoted:
                end = position + 1
                while True:
                    if end >= size:
                        return n_nodes, n_trees, 3, position
                    if text[end] == 39:
                        if end + 1 < size and text[end + 1] == 39:
                            end += 2
                            continue
                        break
                    end += 1
                end += 1
                position = end
            else:
                end = position
                while end < size and not _ends_label(text[end]):
                    end += 1
                position = end
                while end > start and _is_space(text[end - 1]):
                    end -= 1
            if expect_length:
                length_start[node] = start
                length_end[node] = end
                flags[node] |= _HAS_LENGTH | (_QUOTED_LENGTH if quoted else 0)
                expect_length = False
            else:
                if flags[node] & (_HAS_LABEL | _HAS_LENGTH):
                    return n_nodes, n_trees, 4, start
                label_start[node] = start
                label_end[node] = end
                flags[node] |= _HAS_LABEL | (_QUOTED_LABEL if quoted else 0)
            continue
        if expect_length:
            return n_nodes, n_trees, 5, position
        if byte == 40:
            if flags[node] & (_HAS_LABEL | _HAS_LENGTH | _INTERNAL):
                return n_nodes, n_trees, 6, position
            if depth == len(stack) or n_nodes == len(parent):
                return n_nodes, n_trees, 11, position
            flags[node] |= _INTERNAL
            stack[depth] = node
            depth += 1
            parent[n_nodes] = node
            flags[n_nodes] = 0
            node = n_nodes
            n_nodes += 1
        elif byte == 44:
            if depth == 0:
                return n_nodes, n_trees, 7, position
            if n_nodes == len(parent):
                return n_nodes, n_trees, 11, position
            parent[n_nodes] = stack[depth - 1]
            flags[n_nodes] = 0
            node = n_nodes
            n_nodes += 1
        elif byte == 41:
            if depth == 0:
                return n_nodes, n_trees, 8, position
            depth -= 1
            node = stack[depth]
        else:
            if flags[node] & _HAS_LENGTH:
                return n_nodes, n_trees, 9, position
            expect_length = True
        position += 1
    if root >= 0:
        if depth > 0 or expect_length:
            return n_nodes, n_trees, 10, size
        if n_trees == len(tree_ends):
            return n_nodes, n_trees, 11, size
        tree_ends[n_trees] = n_nodes
        n_trees += 1
    return n_nodes, n_trees, 0, -1


def _context(text: bytes | bytearray, position: int) -> str:
    start = max(position - _CONTEXT_CHARACTERS, 0)
    stop = min(position + _CONTEXT_CHARACTERS, len(text))
    return bytes(text[start:stop]).decode("utf-8", errors="replace")


def _read_text(layout: TreeFileLayout) -> bytearray:
    """Read a Newick file's text into a buffer of the size its layout counted."""
    text = bytearray(layout.n_text_bytes)
    view = memoryview(text)
    filled = 0
    try:
        with open_binary(layout.path) as handle:
            while filled < len(text):
                n_read = handle.readinto(view[filled : filled + _BLOCK_BYTES])
                if not n_read:
                    break
                filled += n_read
            grown = bool(handle.read(1))
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(
            f"The Newick file {quoted_value(layout.path)} cannot be read: {error}"
        ) from None
    finally:
        view.release()
    # Names are costed by whether the text is ASCII, as its layout found,
    # and the scan refuses text of more marks than the layout counted.
    if filled != len(text) or grown or text.isascii() != layout.ascii_text:
        raise ValueError(
            f"The Newick file {quoted_value(layout.path)} changed while it was read"
        )
    return text


def _label_text(text: bytearray, start: int, end: int, flag: int, path: str) -> str:
    quoted = bool(flag & _QUOTED_LABEL)
    raw = text[start + 1 : end - 1] if quoted else text[start:end]
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"The Newick file {quoted_value(path)} has a label that is not UTF-8 text at "
            f"byte {start} of its text: {quoted_value(_context(text, start))}"
        ) from None
    return decoded.replace("''", "'") if quoted else decoded


def _branch_length(
    text: bytearray, start: int, end: int, flag: int, path: str
) -> float:
    raw = text[start + 1 : end - 1] if flag & _QUOTED_LENGTH else text[start:end]
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"The Newick file {quoted_value(path)} has a branch length that is not a number "
            f"at byte {start} of its text: {quoted_value(_context(text, start))}"
        ) from None
    if math.isinf(value):
        raise ValueError(
            f"The Newick file {quoted_value(path)} has a branch length that is not finite at "
            f"byte {start} of its text: {quoted_value(_context(text, start))}"
        )
    # A length written as NaN is unknown, as no length is.
    return value


def _batches(n_values: int) -> Iterator[slice]:
    for start in range(0, n_values, _BATCH):
        yield slice(start, min(start + _BATCH, n_values))


def _read_newick(layout: TreeFileLayout, *, ledger: MemoryLedger) -> FileTrees:
    text = _read_text(layout)
    skip = len(_BYTE_ORDER_MARK) if text.startswith(_BYTE_ORDER_MARK) else 0
    bound = layout.max_nodes
    parent = np.empty(bound, dtype=np.int64)
    label_start = np.zeros(bound, dtype=np.int64)
    label_end = np.zeros(bound, dtype=np.int64)
    length_start = np.zeros(bound, dtype=np.int64)
    length_end = np.zeros(bound, dtype=np.int64)
    flags = np.zeros(bound, dtype=np.uint8)
    stack = np.empty(layout.n_groups + 1, dtype=np.int64)
    tree_ends = np.empty(layout.max_trees, dtype=np.int64)
    n_nodes, n_trees, code, position = _scan_newick(
        np.frombuffer(text, dtype=np.uint8)[skip:],
        parent,
        label_start,
        label_end,
        length_start,
        length_end,
        flags,
        stack,
        tree_ends,
    )
    del stack
    if code == _CHANGED:
        raise ValueError(
            f"The Newick file {quoted_value(layout.path)} changed while it was read"
        )
    if code:
        at = position + skip
        place = (
            "at its end"
            if at >= len(text)
            else f"at byte {at} of its text: {quoted_value(_context(text, at))}"
        )
        raise ValueError(
            f"The Newick file {quoted_value(layout.path)} has {_SCAN_ERRORS[code]} {place}"
        )
    if not n_trees:
        raise ValueError(f"The Newick file {quoted_value(layout.path)} holds no tree")
    n_named = label_bytes = 0
    for batch in _batches(n_nodes):
        named = (flags[batch] & _HAS_LABEL) != 0
        n_named += int(np.count_nonzero(named))
        label_bytes += int(np.sum((label_end[batch] - label_start[batch])[named]))
    ascii_text = layout.ascii_text
    ledger.peak(
        f"reading the names and branch lengths of {n_nodes} nodes of "
        f"{quoted_value(layout.path)}",
        layout.read_bytes
        + _TREE_BYTES_PER_NODE * n_nodes
        + (_ASCII_NAME_BYTES if ascii_text else _TEXT_NAME_BYTES) * n_named
        + (1 if ascii_text else 4) * label_bytes,
    )
    names = np.full(n_nodes, "", dtype=object)
    lengths = np.full(n_nodes, np.nan)
    for batch in _batches(n_nodes):
        # Positions of the scan count from the end of a byte order mark.
        batch_flags = flags[batch]
        named = np.flatnonzero(batch_flags & _HAS_LABEL)
        names[batch.start + named] = [
            _label_text(text, start, end, flag, layout.path)
            for start, end, flag in zip(
                (label_start[batch][named] + skip).tolist(),
                (label_end[batch][named] + skip).tolist(),
                batch_flags[named].tolist(),
                strict=True,
            )
        ]
        measured = np.flatnonzero(batch_flags & _HAS_LENGTH)
        lengths[batch.start + measured] = [
            _branch_length(text, start, end, flag, layout.path)
            for start, end, flag in zip(
                (length_start[batch][measured] + skip).tolist(),
                (length_end[batch][measured] + skip).tolist(),
                batch_flags[measured].tolist(),
                strict=True,
            )
        ]
    # Unquoted numbers on internal nodes are support values, not names.
    labelled = np.flatnonzero(
        (flags[:n_nodes] & (_INTERNAL | _HAS_LABEL | _QUOTED_LABEL))
        == (_INTERNAL | _HAS_LABEL)
    )
    for node in labelled.tolist():
        if _SUPPORT.fullmatch(names[node]):
            names[node] = ""
    return FileTrees(
        names=names,
        parent=parent[:n_nodes].copy(),
        length=lengths,
        time=np.full(n_nodes, np.nan),
        leaf=(flags[:n_nodes] & _INTERNAL) == 0,
        tree_offsets=np.concatenate([[0], tree_ends[:n_trees]]).astype(np.int64),
        tree_keys=tuple(str(position) for position in range(1, n_trees + 1)),
    )


def _file_trees(graphs: Sequence[TreeGraph], path: str) -> FileTrees:
    names: list[str] = []
    parents: list[int] = []
    times: list[float] = []
    lengths: list[float] = []
    offsets = [0]
    keys: list[str] = []
    for graph in graphs:
        if not graph.nodes and not graph.edges:
            continue
        tree_names, tree_parents, tree_times, tree_lengths = tree_in_preorder(
            graph, path
        )
        start = len(names)
        names += tree_names
        parents += [start + parent if parent >= 0 else -1 for parent in tree_parents]
        times += tree_times
        lengths += tree_lengths
        offsets.append(len(names))
        keys.append(graph.key)
    if not keys:
        raise ValueError(
            f"The TreeData file {quoted_value(path)} holds no tree in 'obst'"
        )
    parent = np.asarray(parents, dtype=np.int64)
    has_children = np.zeros(len(parent), dtype=bool)
    has_children[parent[parent >= 0]] = True
    return FileTrees(
        names=np.asarray(names, dtype=object),
        parent=parent,
        length=np.asarray(lengths, dtype=np.float64),
        time=np.asarray(times, dtype=np.float64),
        leaf=~has_children,
        tree_offsets=np.asarray(offsets, dtype=np.int64),
        tree_keys=tuple(keys),
    )
