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

import gzip
import json
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

from ..utils.arrays import sort_categories
from ..utils.numba import njit

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
_GZIP_SIGNATURE = b"\x1f\x8b"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
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
# Parsing a TreeData JSON string holds the bytes read, the decoded text, and
# the characters of its strings. Each object then holds at most this many
# bytes once its members other than objects, times and lengths are dropped:
# the dictionary, and the lists that order the nodes and edges built from it.
_JSON_BYTES_PER_OBJECT = 600
# Each member's key is kept until the whole text is parsed, since the parser
# shares keys that repeat.
_JSON_BYTES_PER_MEMBER = 64
# Each value inside the objects being parsed is held until its object is
# read and the value dropped: a list's slot and the value, such as a float.
_JSON_BYTES_PER_VALUE = 64
# The parser reads objects and arrays nested this deep at most; Python's own
# JSON reader stops a little deeper.
_JSON_DEPTH = 8192
# Values shown in errors are cut to this many characters.
_SHOWN_CHARACTERS = 60
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


def _memory_error(what: str, held: int, memory_bytes: int) -> MemoryError:
    return MemoryError(
        f"{what} holds about {held} bytes, more than the memory budget of "
        f"{memory_bytes} bytes. Raise mem_budget."
    )


def _check(what: str, held: int, memory_bytes: int) -> None:
    if held > memory_bytes:
        raise _memory_error(what, held, memory_bytes)


def _is_zarr_store(path: Path) -> bool:
    return path.is_dir() and (
        (path / "zarr.json").is_file() or (path / ".zgroup").is_file()
    )


def _signature(path: Path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(len(_HDF5_SIGNATURE))


def tree_container(path: Path) -> str:
    """Return how a tree file holds its trees, from its bytes.

    Returns:
        ``"zarr"`` for a Zarr store, ``"hdf5"``, ``"gzip"`` or ``"text"``.

    Raises:
        ValueError: If the path is a directory but not a Zarr store.
        FileNotFoundError: If the path does not exist.
    """
    if path.is_dir():
        if _is_zarr_store(path):
            return "zarr"
        raise ValueError(
            f"The tree path {os.fspath(path)!r} is a directory but not a Zarr "
            "store; pass a Newick file or a TreeData .h5td file or .zarr store"
        )
    signature = _signature(path)
    if signature.startswith(_HDF5_SIGNATURE):
        return "hdf5"
    if signature.startswith(_GZIP_SIGNATURE):
        return "gzip"
    return "text"


def _open_binary(path: Path, compression: str | None) -> Any:
    if compression == "gzip":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _newick_layout(path: Path, compression: str | None) -> TreeFileLayout:
    """Count a Newick file's text bytes and the marks that bound its nodes."""
    n_bytes = n_groups = n_commas = n_ends = 0
    ascii_text = True
    head = b""
    try:
        with _open_binary(path, compression) as handle:
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
            f"The Newick file {os.fspath(path)!r} cannot be read: {error}"
        ) from None
    if head.lstrip(_BYTE_ORDER_MARK).lstrip().upper().startswith(b"#NEXUS"):
        raise ValueError(
            f"The tree file {os.fspath(path)!r} is a NEXUS file; export its trees "
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


def _hdf5_tree_bytes(path: Path) -> int:
    """At most the bytes of an HDF5 file's text: those outside fixed-size data.

    HDF5 keeps variable-length text, such as TreeData's JSON, outside the
    storage of its datasets, so the file's bytes less that storage bound it.
    """
    import h5py

    fixed = 0
    with h5py.File(path, "r") as handle:
        if "obst" not in handle:
            if "raw.treedata" in handle:
                raise ValueError(
                    f"The TreeData file {os.fspath(path)!r} keeps its trees in "
                    "'raw.treedata', as the earliest treedata releases did; read it "
                    "with treedata and write it again"
                )
            raise ValueError(
                f"The HDF5 file {os.fspath(path)!r} has no 'obst' element, so it "
                "holds no TreeData trees"
            )
        obst = handle["obst"]
        if (
            not isinstance(obst, h5py.Dataset)
            or h5py.check_string_dtype(obst.dtype) is None
        ):
            raise ValueError(
                f"The TreeData file {os.fspath(path)!r} does not store 'obst' as "
                "one JSON string, as .h5td files do"
            )
        datasets: list[h5py.Dataset] = []
        handle.visititems(
            lambda _name, item: (
                datasets.append(item) if isinstance(item, h5py.Dataset) else None
            )
        )
        for dataset in datasets:
            if dataset.name == obst.name:
                continue
            if h5py.check_vlen_dtype(dataset.dtype) is None and (
                h5py.check_string_dtype(dataset.dtype) is None
                or h5py.check_string_dtype(dataset.dtype).length is not None
            ):
                fixed += int(dataset.id.get_storage_size())
    return max(os.path.getsize(path) - fixed, 0)


def _zarr_obst(path: Path) -> Any:
    import zarr

    try:
        # Each node's own metadata is read, as everywhere in CyteArc, rather
        # than consolidated metadata that may be stale.
        root = zarr.open_group(os.fspath(path), mode="r", use_consolidated=False)
    except Exception as error:
        raise ValueError(
            f"The Zarr store {os.fspath(path)!r} cannot be opened: {error}"
        ) from None
    if "obst" not in root:
        raise ValueError(
            f"The Zarr store {os.fspath(path)!r} has no 'obst' element, so it "
            "holds no TreeData trees"
        )
    return root["obst"]


def _zarr_tree_groups(obst: Any) -> list[tuple[str, Any]]:
    """The trees of a columnar ``obst`` group, by key in natural order."""
    import zarr

    groups = {
        str(name): member
        for name, member in obst.members()
        if isinstance(member, zarr.Group)
    }
    return [(key, groups[key]) for key in sort_categories(groups)]


def _treedata_layout(path: Path, container: str) -> TreeFileLayout:
    import zarr

    if container == "hdf5":
        return TreeFileLayout(
            path=os.fspath(path),
            format="treedata",
            compression=None,
            container="hdf5",
            n_text_bytes=_hdf5_tree_bytes(path),
            max_nodes=0,
        )
    obst = _zarr_obst(path)
    if not isinstance(obst, zarr.Group):
        # Its size is unknown until it is read, and reading it can cost far
        # more than its text, so it is not read.
        raise ValueError(
            f"The TreeData store {os.fspath(path)!r} holds 'obst' as one JSON "
            "string, as treedata wrote Zarr stores before 0.3.0; write it again "
            "with treedata 0.3.0 or later, or as a .h5td file"
        )
    n_nodes = 0
    if isinstance(obst, zarr.Group):
        for key, tree in _zarr_tree_groups(obst):
            nodes = tree.get("nodes")
            if not isinstance(nodes, zarr.Array) or len(nodes.shape) != 1:
                raise ValueError(
                    f"The tree {key!r} of the TreeData store {os.fspath(path)!r} "
                    "has no one-dimensional 'nodes' array"
                )
            n_nodes += int(nodes.shape[0])
    return TreeFileLayout(
        path=os.fspath(path),
        format="treedata",
        compression=None,
        container="zarr",
        n_text_bytes=0,
        max_nodes=n_nodes,
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
            f"format must be one of {', '.join(FORMATS)} or None, not {format!r}"
        )
    resolved = Path(os.fspath(path))
    if not resolved.exists():
        raise FileNotFoundError(f"The tree file {os.fspath(path)!r} does not exist")
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
            f"The tree file {os.fspath(path)!r} is {what}, which holds "
            f"{detected} trees, not {format} trees; pass format={detected!r} or "
            "None"
        )
    if detected == "treedata":
        return _treedata_layout(resolved, container)
    return _newick_layout(resolved, "gzip" if container == "gzip" else None)


def read_trees(
    layout: TreeFileLayout, *, memory_bytes: int, held_bytes: int = 0
) -> FileTrees:
    """Read the trees of one file.

    Args:
        layout: The file, from :func:`tree_file_layout`.
        memory_bytes: The memory budget.
        held_bytes: Bytes that the caller holds beside the read.

    Raises:
        ValueError: If the file is not well-formed, a tree is not a tree, or
            the file holds no tree.
        MemoryError: If reading the trees does not fit the memory budget.
    """
    _check(
        f"Reading the trees of {layout.path!r}",
        held_bytes + layout.read_bytes,
        memory_bytes,
    )
    if layout.format == "newick":
        return _read_newick(layout, memory_bytes=memory_bytes, held_bytes=held_bytes)
    if layout.container == "hdf5":
        return _read_h5td(layout, memory_bytes=memory_bytes, held_bytes=held_bytes)
    return _read_zarr(layout, memory_bytes=memory_bytes, held_bytes=held_bytes)


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
        with _open_binary(Path(layout.path), layout.compression) as handle:
            while filled < len(text):
                n_read = handle.readinto(view[filled : filled + _BLOCK_BYTES])
                if not n_read:
                    break
                filled += n_read
            grown = bool(handle.read(1))
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(
            f"The Newick file {layout.path!r} cannot be read: {error}"
        ) from None
    finally:
        view.release()
    # Names are costed by whether the text is ASCII, as its layout found,
    # and the scan refuses text of more marks than the layout counted.
    if filled != len(text) or grown or text.isascii() != layout.ascii_text:
        raise ValueError(f"The Newick file {layout.path!r} changed while it was read")
    return text


def _label_text(text: bytearray, start: int, end: int, flag: int, path: str) -> str:
    quoted = bool(flag & _QUOTED_LABEL)
    raw = text[start + 1 : end - 1] if quoted else text[start:end]
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"The Newick file {path!r} has a label that is not UTF-8 text at "
            f"byte {start} of its text: {_context(text, start)!r}"
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
            f"The Newick file {path!r} has a branch length that is not a number "
            f"at byte {start} of its text: {_context(text, start)!r}"
        ) from None
    if math.isinf(value):
        raise ValueError(
            f"The Newick file {path!r} has a branch length that is not finite at "
            f"byte {start} of its text: {_context(text, start)!r}"
        )
    # A length written as NaN is unknown, as no length is.
    return value


def _batches(n_values: int) -> Iterator[slice]:
    for start in range(0, n_values, _BATCH):
        yield slice(start, min(start + _BATCH, n_values))


def _read_newick(
    layout: TreeFileLayout, *, memory_bytes: int, held_bytes: int
) -> FileTrees:
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
        raise ValueError(f"The Newick file {layout.path!r} changed while it was read")
    if code:
        at = position + skip
        place = (
            "at its end"
            if at >= len(text)
            else f"at byte {at} of its text: {_context(text, at)!r}"
        )
        raise ValueError(
            f"The Newick file {layout.path!r} has {_SCAN_ERRORS[code]} {place}"
        )
    if not n_trees:
        raise ValueError(f"The Newick file {layout.path!r} holds no tree")
    n_named = label_bytes = 0
    for batch in _batches(n_nodes):
        named = (flags[batch] & _HAS_LABEL) != 0
        n_named += int(np.count_nonzero(named))
        label_bytes += int(np.sum((label_end[batch] - label_start[batch])[named]))
    ascii_text = layout.ascii_text
    _check(
        f"Reading the names and branch lengths of {n_nodes} nodes of {layout.path!r}",
        held_bytes
        + layout.read_bytes
        + _TREE_BYTES_PER_NODE * n_nodes
        + (_ASCII_NAME_BYTES if ascii_text else _TEXT_NAME_BYTES) * n_named
        + (1 if ascii_text else 4) * label_bytes,
        memory_bytes,
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


# TreeData


@dataclass(slots=True)
class _GraphTree:
    """One TreeData tree as nodes, edges and the two attributes read."""

    key: str
    nodes: list[str]
    times: dict[str, Any]
    edges: list[tuple[str, str]]
    lengths: list[Any]


def _shown(value: Any) -> str:
    """A value as an error shows it, cut short when it is long."""
    text = repr(value)
    if len(text) <= _SHOWN_CHARACTERS:
        return text
    return text[:_SHOWN_CHARACTERS] + "..."


def _number(value: Any, what: str, path: str, key: str) -> float:
    """A time or length as a float, NaN for an absent, null or NaN value.

    treedata writes NaN where a branch has no length and below it.
    """
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"The tree {key!r} of the TreeData file {path!r} has a {what} that "
            f"is not a number: {_shown(value)}"
        )
    number = float(value)
    if math.isinf(number):
        raise ValueError(
            f"The tree {key!r} of the TreeData file {path!r} has a {what} that "
            f"is not finite: {value!r}"
        )
    return number


def _tree_in_preorder(
    graph: _GraphTree, path: str
) -> tuple[list[str], list[int], list[float], list[float]]:
    """Order a tree's nodes from its root, parents before their children.

    Returns:
        The names, each node's parent as a position, -1 for the root, its
        time, and the length of the branch above it.

    Raises:
        ValueError: If the graph has several roots or none, a node with two
            parents, or nodes that its root does not reach.
    """
    key = graph.key
    known = dict.fromkeys(graph.nodes)
    children: dict[str, list[tuple[str, int]]] = {}
    parent_of: dict[str, str] = {}
    for index, (source, target) in enumerate(graph.edges):
        known.setdefault(source)
        known.setdefault(target)
        if target in parent_of:
            raise ValueError(
                f"The tree {key!r} of the TreeData file {path!r} gives the node "
                f"{target!r} two parents, {parent_of[target]!r} and {source!r}"
            )
        parent_of[target] = source
        children.setdefault(source, []).append((target, index))
    roots = [name for name in known if name not in parent_of]
    if len(roots) != 1:
        raise ValueError(
            f"The tree {key!r} of the TreeData file {path!r} has "
            f"{len(roots)} roots, not one"
            + (f", such as {roots[:3]!r}" if roots else "; its edges form a cycle")
        )
    names: list[str] = []
    parents: list[int] = []
    times: list[float] = []
    lengths: list[float] = []
    pending: list[tuple[str, int, int]] = [(roots[0], -1, -1)]
    while pending:
        name, parent, edge = pending.pop()
        position = len(names)
        names.append(name)
        parents.append(parent)
        times.append(_number(graph.times.get(name), "time", path, key))
        lengths.append(
            math.nan if edge < 0 else _number(graph.lengths[edge], "length", path, key)
        )
        for child, index in reversed(children.get(name, [])):
            pending.append((child, position, index))
    if len(names) != len(known):
        raise ValueError(
            f"The tree {key!r} of the TreeData file {path!r} has "
            f"{len(known) - len(names)} nodes that its root does not reach"
        )
    return names, parents, times, lengths


def _file_trees(graphs: Sequence[_GraphTree], path: str) -> FileTrees:
    names: list[str] = []
    parents: list[int] = []
    times: list[float] = []
    lengths: list[float] = []
    offsets = [0]
    keys: list[str] = []
    for graph in graphs:
        if not graph.nodes and not graph.edges:
            continue
        tree_names, tree_parents, tree_times, tree_lengths = _tree_in_preorder(
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
        raise ValueError(f"The TreeData file {path!r} holds no tree in 'obst'")
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


class _ListValue:
    """Stands for a time or a length that the JSON gives as a list."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "a list"


_LIST_VALUE = _ListValue()


@njit(cache=True)
def _scan_json(  # noqa: C901 - one state machine over the text
    text: np.ndarray, open_values: np.ndarray, open_objects: np.ndarray
) -> tuple[int, int, int, int, int]:
    """Count what parsing a JSON text holds, without parsing it.

    Strings are skipped with their escapes. ``open_values`` and
    ``open_objects`` hold, for each object or array open at a depth, the
    values it holds outside its nested objects and whether it is an object;
    their size bounds the depth. A value is held until the object around it
    is read, unless the value is an object itself.

    Returns:
        The objects, the members of objects, the most values held at one
        moment, the characters, and the largest byte; the objects are -1 if
        the text nests deeper than the arrays allow.
    """
    n_objects = 0
    n_members = 0
    n_characters = 0
    largest = 0
    depth = 0
    held = 0
    most = 0
    in_string = False
    escaped = False
    for at in range(len(text)):
        byte = text[at]
        if byte > largest:
            largest = byte
        if byte & 0xC0 != 0x80:
            n_characters += 1
        if in_string:
            if escaped:
                escaped = False
            elif byte == 92:  # backslash
                escaped = True
            elif byte == 34:  # quote
                in_string = False
        elif byte == 34:
            in_string = True
        elif byte == 123 or byte == 91:  # { [
            if depth == len(open_values):
                return -1, 0, 0, 0, 0
            if byte == 123:
                n_objects += 1
            open_objects[depth] = byte == 123
            # A container holds its first value, if it has one, without a ','.
            open_values[depth] = 1
            depth += 1
            held += 1
            most = max(most, held)
        elif byte == 44:  # ,
            if depth:
                open_values[depth - 1] += 1
                held += 1
                most = max(most, held)
        elif byte == 58:  # :
            n_members += 1
        elif (byte == 125 or byte == 93) and depth:  # } ]
            depth -= 1
            if open_objects[depth]:
                held -= open_values[depth]
            elif depth:
                open_values[depth - 1] += open_values[depth]
    return n_objects, n_members, most, n_characters, largest


class _Attributes(dict[str, Any]):
    """A JSON object cut down to its objects and its ``time`` and ``length``.

    Attributes:
        dropped: How many of its members were dropped.
    """

    __slots__ = ("dropped",)

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.dropped = 0


def _attributes_only(values: dict[str, Any]) -> dict[str, Any]:
    """Keep only the members of a JSON object that a tree needs.

    The objects of the JSON nest as trees, then nodes and edges, then one
    object of attributes per node and edge. Every member whose value is an
    object is kept, so the nesting survives, and so are ``time`` and
    ``length``; every other member, such as a node's character states, is
    dropped as soon as its object is parsed, so the parsed JSON stays small
    whatever the attributes hold. The result counts what it dropped, so a
    mapping of nodes or edges whose values are not objects is still found
    malformed.
    """
    kept = _Attributes()
    for name, value in values.items():
        if isinstance(value, dict):
            kept[name] = value
        elif name in ("time", "length"):
            # A list is dropped too, so its values are held no longer than
            # any other's; what stands for it is refused as not a number.
            kept[name] = _LIST_VALUE if isinstance(value, list) else value
        else:
            kept.dropped += 1
    return kept


def _is_mapping(value: Any) -> bool:
    """Whether a parsed JSON value maps names to objects and nothing else."""
    return isinstance(value, dict) and not getattr(value, "dropped", 0)


def _attributes(value: Any, what: str, key: str, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"The tree {key!r} of the TreeData file {path!r} has {what} whose "
            f"attributes are not an object: {value!r}"
        )
    return value


def _json_graphs(text: str, path: str) -> list[_GraphTree]:
    try:
        trees = json.loads(text, object_hook=_attributes_only)
    except (ValueError, RecursionError) as error:
        # Besides malformed JSON, Python refuses integers of over 4,300
        # digits and nesting deeper than it can follow.
        raise ValueError(
            f"The TreeData file {path!r} has an 'obst' that is not JSON that "
            f"Python can read: {error}"
        ) from None
    if not _is_mapping(trees):
        raise ValueError(
            f"The TreeData file {path!r} has an 'obst' that does not map tree "
            "keys to trees"
        )
    graphs = []
    for key in sort_categories(trees):
        tree = trees[key]
        if (
            not isinstance(tree, dict)
            or not _is_mapping(tree.get("nodes"))
            or not _is_mapping(tree.get("edges"))
        ):
            raise ValueError(
                f"The tree {key!r} of the TreeData file {path!r} has no 'nodes' "
                "and 'edges' that map names to objects"
            )
        edges: list[tuple[str, str]] = []
        lengths: list[Any] = []
        for source, targets in tree["edges"].items():
            if not _is_mapping(targets):
                raise ValueError(
                    f"The tree {key!r} of the TreeData file {path!r} has edges "
                    f"from {source!r} that do not map targets to objects"
                )
            for target, attributes in targets.items():
                edges.append((str(source), str(target)))
                lengths.append(
                    _attributes(
                        attributes, f"an edge to {target!r}", str(key), path
                    ).get("length")
                )
        times = {}
        for name, attributes in tree["nodes"].items():
            node = _attributes(attributes, f"a node {name!r}", str(key), path)
            if "time" in node:
                times[str(name)] = node["time"]
        graphs.append(
            _GraphTree(
                key=str(key),
                nodes=[str(name) for name in tree["nodes"]],
                times=times,
                edges=edges,
                lengths=lengths,
            )
        )
    return graphs


def _parse_json_trees(
    raw: bytes, layout: TreeFileLayout, *, memory_bytes: int, held_bytes: int
) -> FileTrees:
    n_objects, n_members, most_held, n_characters, largest = _scan_json(
        np.frombuffer(raw, dtype=np.uint8),
        np.empty(_JSON_DEPTH, dtype=np.int64),
        np.empty(_JSON_DEPTH, dtype=np.bool_),
    )
    if n_objects < 0:
        raise ValueError(
            f"The TreeData file {layout.path!r} has an 'obst' whose objects and "
            f"arrays nest more than {_JSON_DEPTH} deep"
        )
    # Python stores text at one, two or four bytes a character, by the
    # largest character, which UTF-8's largest leading byte bounds.
    width = 1 if largest < 0xC4 else 2 if largest < 0xF0 else 4
    # Each node and edge is an object of the JSON, so the objects bound them.
    _check(
        f"Parsing the trees of {layout.path!r}",
        held_bytes
        + len(raw)
        + 2 * width * n_characters
        + _JSON_BYTES_PER_OBJECT * n_objects
        + _JSON_BYTES_PER_MEMBER * n_members
        + _JSON_BYTES_PER_VALUE * most_held,
        memory_bytes,
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"The TreeData file {layout.path!r} has an 'obst' that is not UTF-8"
        ) from None
    del raw
    graphs = _json_graphs(text, layout.path)
    del text
    return _file_trees(graphs, layout.path)


def _read_h5td(
    layout: TreeFileLayout, *, memory_bytes: int, held_bytes: int
) -> FileTrees:
    import h5py

    with h5py.File(layout.path, "r") as handle:
        value = handle["obst"][()]
    raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return _parse_json_trees(
        raw, layout, memory_bytes=memory_bytes, held_bytes=held_bytes
    )


def _text_blocks(array: Any) -> Iterator[np.ndarray]:
    """A Zarr array's values as Python strings, one chunk of rows at a time."""
    n_rows = int(array.shape[0])
    step = max(int(array.chunks[0]), 1) if n_rows else 1
    for start in range(0, n_rows, step):
        block = np.asarray(array[start : start + step])
        yield np.asarray(block.tolist(), dtype=object).reshape(block.shape)


def _held_text(values: np.ndarray) -> int:
    return 8 * values.size + sum(map(sys.getsizeof, values.reshape(-1).tolist()))


def _zarr_texts(
    array: Any, what: str, *, memory_bytes: int, held_bytes: int
) -> tuple[np.ndarray, int]:
    """Read a Zarr text array chunk by chunk, counting its strings as they come.

    Returns:
        The values, and the bytes they hold.
    """
    blocks = []
    held = 0
    for block in _text_blocks(array):
        held += _held_text(block)
        _check(what, held_bytes + 2 * held, memory_bytes)
        blocks.append(block)
    if not blocks:
        return np.empty(array.shape, dtype=object), 0
    return np.concatenate(blocks), held


def _zarr_attribute(
    group: Any,
    name: str,
    n_values: int,
    key: str,
    path: str,
    *,
    memory_bytes: int,
    held_bytes: int,
) -> tuple[list[Any], int]:
    """A node or edge attribute column, None where it is absent.

    A column of one numeric type is stored as numbers; any other is stored as
    text of JSON, ``""`` where the attribute is absent.
    """
    import zarr

    if group is None or name not in group:
        return [None] * n_values, 0
    array = group[name]
    if not isinstance(array, zarr.Array) or array.shape != (n_values,):
        raise ValueError(
            f"The tree {key!r} of the TreeData store {path!r} has a {name!r} "
            f"attribute that does not hold one value per element"
        )
    if array.dtype.kind in "iufb":
        values = np.asarray(array[...])
        return values.tolist(), int(values.nbytes)
    texts, held = _zarr_texts(
        array,
        f"Reading the {name!r} of the tree {key!r} of {path!r}",
        memory_bytes=memory_bytes,
        held_bytes=held_bytes,
    )
    decoded: list[Any] = []
    for text in texts.tolist():
        if text == "":
            decoded.append(None)
            continue
        if text.lstrip()[:1] in ("[", "{"):
            # A list or an object is no time or length, and is refused before
            # it is parsed, so its values are never held.
            raise ValueError(
                f"The tree {key!r} of the TreeData store {path!r} has a {name!r} "
                f"that is not a number: {_shown(text)}"
            )
        try:
            decoded.append(json.loads(text))
        except (ValueError, RecursionError):
            raise ValueError(
                f"The tree {key!r} of the TreeData store {path!r} has a {name!r} "
                f"that is not JSON that Python can read: {_shown(text)}"
            ) from None
    return decoded, held


def _zarr_graph(
    key: str, tree: Any, path: str, *, memory_bytes: int, held_bytes: int
) -> tuple[_GraphTree, int]:
    import zarr

    nodes_array = tree["nodes"]
    edges_array = tree.get("edges")
    if (
        not isinstance(edges_array, zarr.Array)
        or len(edges_array.shape) != 2
        or (edges_array.shape[0] and edges_array.shape[1] != 2)
    ):
        raise ValueError(
            f"The tree {key!r} of the TreeData store {path!r} has no 'edges' "
            "array of source and target pairs"
        )
    what = f"Reading the tree {key!r} of {path!r}"
    nodes, held = _zarr_texts(
        nodes_array, what, memory_bytes=memory_bytes, held_bytes=held_bytes
    )
    edges, edge_held = _zarr_texts(
        edges_array, what, memory_bytes=memory_bytes, held_bytes=held_bytes + held
    )
    held += edge_held
    node_attributes = tree.get("node_attrs")
    edge_attributes = tree.get("edge_attrs")
    times, time_held = _zarr_attribute(
        node_attributes,
        "time",
        len(nodes),
        key,
        path,
        memory_bytes=memory_bytes,
        held_bytes=held_bytes + held,
    )
    held += time_held
    lengths, length_held = _zarr_attribute(
        edge_attributes,
        "length",
        len(edges),
        key,
        path,
        memory_bytes=memory_bytes,
        held_bytes=held_bytes + held,
    )
    held += length_held
    names = [str(name) for name in nodes.tolist()]
    graph = _GraphTree(
        key=key,
        nodes=names,
        times={
            name: value
            for name, value in zip(names, times, strict=True)
            if value is not None
        },
        edges=[(str(source), str(target)) for source, target in edges.tolist()],
        lengths=lengths,
    )
    return graph, held


def _read_zarr(
    layout: TreeFileLayout, *, memory_bytes: int, held_bytes: int
) -> FileTrees:
    import zarr

    obst = _zarr_obst(Path(layout.path))
    if not isinstance(obst, zarr.Group):
        raise ValueError(
            f"The TreeData store {layout.path!r} changed while it was read"
        )
    graphs = []
    held = held_bytes + layout.read_bytes
    for key, tree in _zarr_tree_groups(obst):
        graph, tree_held = _zarr_graph(
            key, tree, layout.path, memory_bytes=memory_bytes, held_bytes=held
        )
        held += tree_held
        graphs.append(graph)
    return _file_trees(graphs, layout.path)
