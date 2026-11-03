"""Read TreeData files, their character matrices and their trees, and write
character matrices.

TreeData, the format of Cassiopeia 3.0 and of PEtracer's published data, is
AnnData's on-disk layout with trees beside it, in HDF5 (``.h5td``) or Zarr
(``.zarr``) files. A lineage recorder's character matrix is the dataframe
``obsm['characters']``: cells by characters named ``{integration}-{site}``,
whose states are integers or text. ``uns['missing_state']`` and
``uns['unmodified_state']`` name the states of a missing and an unedited
site where a file records them.

``character_table`` opens a file's character matrix without reading its
values, and ``CharacterTable.blocks`` yields its cells and states in blocks
of rows, so a matrix of any size is read within a bounded memory.
``write_character_store`` writes cells by characters of integer states as a
TreeData ``.zarr`` store in the layout that treedata 0.3.0 and later read,
with the state table that Cassiopeia's ``uns['character_states']`` holds and
no trees.

The trees of ``obst`` are one JSON string in ``.h5td`` files and, since
treedata 0.3.0, one group per tree in ``.zarr`` stores, with each node
attribute in a column; ``h5td_tree_graphs`` and ``zarr_tree_graphs`` read
each tree's nodes, edges, node times and edge lengths, and nothing else, and
``tree_in_preorder`` orders a tree's nodes from its root.
"""

import json
import math
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..utils.arrays import sort_categories
from ..utils.files import HDF5_SIGNATURE
from ..utils.numba import njit
from ._h5ad_columns import index_length, index_names
from ._text import as_text
from ..utils.arguments import quoted_value

__all__ = [
    "CharacterTable",
    "TreeGraph",
    "character_table",
    "h5td_tree_bytes",
    "h5td_tree_graphs",
    "is_zarr_store",
    "tree_in_preorder",
    "write_character_store",
    "zarr_tree_graphs",
    "zarr_tree_nodes",
]

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
# Bytes that each state of a block takes once read: its stored value, its
# text and the reference to it.
_STATE_BYTES = 120
# The encoding versions of AnnData's elements that a store holds.
_DATAFRAME = ("dataframe", "0.2.0")
_DICT = ("dict", "0.1.0")
_ARRAY = ("array", "0.2.0")
_STRING_ARRAY = ("string-array", "0.2.0")
_NUMERIC_SCALAR = ("numeric-scalar", "0.2.0")
_STRING = ("string", "0.2.0")
# Rows of each chunk of a written array of numbers, and of text, which Zarr
# encodes at about 160 bytes per value; arrays are written a chunk at a time.
_CHUNK_ROWS = 65_536
_TEXT_CHUNK_ROWS = 16_384
#: The bytes that writing a store holds beside the values it is given: a
#: chunk of text as Zarr encodes it, and a character's column copied as int64
#: with its encoded chunk.
WRITE_BYTES = 160 * _TEXT_CHUNK_ROWS + 16 * _CHUNK_ROWS
#: Bytes per cell of a character's column as it is written.
WRITE_BYTES_PER_CELL = 8


def is_zarr_store(path: str | os.PathLike[str]) -> bool:
    """Whether a path is a Zarr store's folder."""
    folder = Path(path)
    return folder.is_dir() and (
        (folder / "zarr.json").is_file() or (folder / ".zgroup").is_file()
    )


def _encoding(node: Any) -> str | None:
    value = node.attrs.get("encoding-type")
    return None if value is None else as_text(value)


def _names(raw: Any) -> list[str]:
    if raw is None:
        return []
    return [as_text(value) for value in np.asarray(raw, dtype=object).reshape(-1)]


def _is_group(node: Any) -> bool:
    return not hasattr(node, "shape")


def _scalar_token(value: Any) -> str:
    """The text of a stored state, as the character matrix's tokens read."""
    if isinstance(value, bytes | np.bytes_):
        return as_text(value)
    if isinstance(value, str):
        return value
    number = np.asarray(value).item()
    if isinstance(number, bool):
        raise ValueError(f"A state cannot be the boolean {quoted_value(number)}")
    if isinstance(number, float):
        if not np.isfinite(number) or number != int(number):
            raise ValueError(
                f"A state must be a whole number, not {quoted_value(number)}"
            )
        return str(int(number))
    return str(number)


def _block_tokens(values: np.ndarray, mask: np.ndarray | None, what: str) -> np.ndarray:
    """A block of a column's values as text, None where a value is missing."""
    tokens = np.full(len(values), None, dtype=object)
    present = np.ones(len(values), dtype=bool) if mask is None else ~mask
    kind = values.dtype.kind
    if kind in "iu":
        tokens[present] = values[present].astype(np.int64).astype(str).astype(object)
    elif kind == "f":
        present &= ~np.isnan(values)
        chosen = values[present]
        if len(chosen) and not np.array_equal(chosen, np.trunc(chosen)):
            raise ValueError(f"{what} holds states that are not whole numbers")
        tokens[present] = chosen.astype(np.int64).astype(str).astype(object)
    elif kind == "b":
        raise ValueError(f"{what} holds booleans, which are no states")
    else:
        for position in np.flatnonzero(present).tolist():
            value = values[position]
            tokens[position] = None if value is None else as_text(value)
    return tokens


@dataclass(frozen=True, slots=True)
class _Column:
    """One column of a stored dataframe and how to read it."""

    name: str
    node: Any
    encoding: str  # "array", "categorical" or "nullable"

    def read(self, start: int, stop: int, what: str) -> np.ndarray:
        if self.encoding == "categorical":
            codes = np.asarray(self.node["codes"][start:stop], dtype=np.int64)
            categories = _block_tokens(
                np.asarray(self.node["categories"][:]), None, what
            )
            tokens = np.full(len(codes), None, dtype=object)
            valid = (codes >= 0) & (codes < len(categories))
            tokens[valid] = categories[codes[valid]]
            return tokens
        if self.encoding == "nullable":
            values = np.asarray(self.node["values"][start:stop])
            mask = np.asarray(self.node["mask"][start:stop], dtype=bool)
            return _block_tokens(values, mask, what)
        return _block_tokens(np.asarray(self.node[start:stop]), None, what)


def _column(name: str, node: Any, what: str) -> _Column:
    if _is_group(node):
        keys = set(node.keys())
        if {"codes", "categories"} <= keys:
            return _Column(name, node, "categorical")
        if {"values", "mask"} <= keys:
            return _Column(name, node, "nullable")
        raise ValueError(
            f"{what} holds a column {quoted_value(name)} that CyteArc cannot read"
        )
    if len(node.shape) != 1:
        raise ValueError(
            f"{what} holds a column {quoted_value(name)} of more than one dimension"
        )
    return _Column(name, node, "array")


@dataclass(frozen=True, slots=True)
class CharacterTable:
    """A TreeData file's character matrix, opened without reading its values.

    Attributes:
        path: The file or store.
        container: ``"hdf5"`` or ``"zarr"``.
        n_cells: The matrix's rows.
        characters: Each column's name.
        missing_state: The state of a missing site that the file records in
            ``uns``, as text, or None.
        unmodified_state: The state of an unedited site that the file
            records in ``uns``, as text, or None.
        encoding: The ``encoding-type`` and ``encoding-version`` of the
            file's root, such as ``"treedata 0.1.0"``.
        names_from: Where the characters' names are read: ``"columns"`` of
            a dataframe, or ``"uns['embryo_characters']"`` for a plain array,
            as PEtracer's published files store them.
    """

    path: str
    container: str
    n_cells: int
    characters: tuple[str, ...]
    missing_state: str | None
    unmodified_state: str | None
    encoding: str | None
    names_from: str = "columns"

    def held_bytes(self) -> int:
        """The bytes that reading holds beside its blocks: the column names."""
        return sum(100 + len(name) for name in self.characters)

    def block_bytes(self, n_rows: int) -> int:
        """The bytes that a block of ``n_rows`` rows holds once read."""
        return _STATE_BYTES * int(n_rows) * (len(self.characters) + 1)

    def blocks(self, n_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield the cells and states of blocks of ``n_rows`` rows.

        Yields:
            Each row's cell name, and its states as text, rows by
            characters, None where a value is missing.
        """
        step = max(1, int(n_rows))
        with _open(self.path, self.container) as root:
            index, columns, _names_from = _dataframe(root, self.path)
            what = f"The character matrix of {self.path}"
            for start in range(0, self.n_cells, step):
                stop = min(start + step, self.n_cells)
                cells = index_names(index, start, stop, path=self.path, what="cell")
                states = np.empty((stop - start, len(columns)), dtype=object)
                for position, column in enumerate(columns):
                    states[:, position] = column.read(start, stop, what)
                yield cells, states


def _zarr_root(path: str) -> Any:
    """Open a Zarr store's root group for reading.

    Raises:
        ValueError: If the store cannot be opened.
    """
    import zarr

    try:
        # Each node's own metadata is read, as everywhere in CyteArc, rather
        # than consolidated metadata that may be stale.
        return zarr.open_group(path, mode="r", use_consolidated=False)
    except Exception as error:
        raise ValueError(
            f"The Zarr store {quoted_value(path)} cannot be opened: {error}"
        ) from None


class _open:
    """Open a TreeData file or store for reading, as a context manager."""

    def __init__(self, path: str, container: str) -> None:
        self._path, self._container = path, container
        self._handle: Any = None

    def __enter__(self) -> Any:
        if self._container == "hdf5":
            self._handle = h5py.File(self._path, "r")
            return self._handle
        return _zarr_root(self._path)

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            self._handle.close()


# Where PEtracer's published TreeData files name the columns of their plain
# array of characters.
_EMBRYO_CHARACTERS = "embryo_characters"


def _dataframe(root: Any, path: str) -> tuple[Any, list[_Column], str]:
    """The table whose index names the cells, the columns, and where the
    characters' names come from, of ``obsm['characters']``."""
    obsm = root.get("obsm")
    node = None if obsm is None or not _is_group(obsm) else obsm.get("characters")
    if node is None:
        raise ValueError(
            f"{path} holds no obsm['characters'], so it holds no character matrix"
        )
    what = f"The character matrix of {path}"
    if not _is_group(node):
        # A plain array names its cells by the file's obs names, and its
        # characters only where PEtracer's files list them in uns.
        if len(node.shape) != 2:
            raise ValueError(f"{what} must be a table of cells by characters")
        obs = root.get("obs")
        index_name = None if obs is None else obs.attrs.get("_index")
        if obs is None or index_name is None:
            raise ValueError(f"{path} names no cells in obs")
        uns = root.get("uns")
        listed = (
            None if uns is None or not _is_group(uns) else uns.get(_EMBRYO_CHARACTERS)
        )
        if listed is None or _is_group(listed) or len(listed.shape) != 1:
            raise ValueError(
                f"{what} is an array whose columns have no names, so its "
                "characters and their integrations are unknown; CyteArc reads "
                "a dataframe of named columns, or the names that PEtracer's "
                f"files list in uns[{quoted_value(_EMBRYO_CHARACTERS)}]"
            )
        names = _names(listed[:])
        if len(names) != int(node.shape[1]):
            raise ValueError(
                f"uns[{quoted_value(_EMBRYO_CHARACTERS)}] of {path} names {len(names)} "
                f"characters for the {int(node.shape[1])} columns of {what}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"{what} names a character twice")
        return (
            obs,
            [
                _Column(name, _ColumnOf(node, position), "array")
                for position, name in enumerate(names)
            ],
            f"uns[{quoted_value(_EMBRYO_CHARACTERS)}]",
        )
    if _encoding(node) != "dataframe":
        raise ValueError(f"{what} is no dataframe that CyteArc can read")
    index_name = node.attrs.get("_index")
    if index_name is None or as_text(index_name) not in node:
        raise ValueError(f"{what} names no cells")
    names = _names(node.attrs.get("column-order"))
    if len(set(names)) != len(names):
        raise ValueError(f"{what} names a character twice")
    columns = []
    for name in names:
        if name not in node:
            raise ValueError(
                f"{what} lists a character {quoted_value(name)} that it does not hold"
            )
        columns.append(_column(name, node[name], what))
    return node, columns, "columns"


class _ColumnOf:
    """One column of a stored two-dimensional array, read by rows."""

    def __init__(self, matrix: Any, position: int) -> None:
        self._matrix, self._position = matrix, position
        self.shape = (int(matrix.shape[0]),)

    def __getitem__(self, rows: slice) -> np.ndarray:
        return np.asarray(self._matrix[rows, self._position])


def _uns_state(root: Any, name: str) -> str | None:
    uns = root.get("uns")
    if uns is None or not _is_group(uns) or name not in uns:
        return None
    node = uns[name]
    if _is_group(node) or node.shape != ():
        raise ValueError(f"uns[{quoted_value(name)}] must hold one state")
    return _scalar_token(node[()])


def character_table(path: str | os.PathLike[str]) -> CharacterTable:
    """Open the character matrix of a TreeData ``.h5td`` file or ``.zarr`` store.

    Raises:
        ValueError: If the file holds no readable ``obsm['characters']``, or
            its index and columns differ in length.
    """
    text = os.fspath(path)
    container = "zarr" if is_zarr_store(text) else "hdf5"
    if container == "hdf5":
        with open(text, "rb") as handle:
            if not handle.read(len(HDF5_SIGNATURE)).startswith(HDF5_SIGNATURE):
                raise ValueError(f"{text} is neither an HDF5 file nor a Zarr store")
    with _open(text, container) as root:
        index, columns, names_from = _dataframe(root, text)
        n_cells = index_length(index, text)
        for column in columns:
            node = column.node
            length = (
                int(node["codes"].shape[0])
                if column.encoding == "categorical"
                else int(node["values"].shape[0])
                if column.encoding == "nullable"
                else int(node.shape[0])
            )
            if length != n_cells:
                raise ValueError(
                    f"The character {quoted_value(column.name)} of {text} holds {length} values "
                    f"for {n_cells} cells"
                )
        encoding = root.attrs.get("encoding-type")
        version = root.attrs.get("encoding-version")
        return CharacterTable(
            path=text,
            container=container,
            n_cells=n_cells,
            characters=tuple(column.name for column in columns),
            missing_state=_uns_state(root, "missing_state"),
            unmodified_state=_uns_state(root, "unmodified_state"),
            encoding=None
            if encoding is None
            else f"{as_text(encoding)} {as_text(version or '')}".strip(),
            names_from=names_from,
        )


def _encoded(encoding: tuple[str, str], **attributes: Any) -> dict[str, Any]:
    """The attributes of a node of this encoding, set as it is created.

    Setting attributes one by one rewrites the node's metadata each time,
    which costs as much again as creating it.
    """
    return {"encoding-type": encoding[0], "encoding-version": encoding[1], **attributes}


def _write_array(group: Any, name: str, values: np.ndarray) -> None:
    """Write a one-dimensional array of text or numbers, a chunk at a time."""
    values = np.asarray(values)
    text = values.dtype.kind in "OUS"
    rows = _TEXT_CHUNK_ROWS if text else _CHUNK_ROWS
    chunks = (max(1, min(rows, len(values))),)
    array = group.create_array(
        name,
        shape=values.shape,
        dtype=str if text else values.dtype,
        chunks=chunks,
        attributes=_encoded(_STRING_ARRAY if text else _ARRAY),
    )
    for start in range(0, len(values), chunks[0]):
        block = values[start : start + chunks[0]]
        array[start : start + len(block)] = (
            np.asarray([str(value) for value in block.tolist()], dtype=object)
            if text
            else block
        )


def _write_frame(
    group: Any,
    name: str,
    index: np.ndarray,
    names: Sequence[str],
    column: Callable[[int], np.ndarray],
) -> Any:
    """Write a dataframe whose ``column(position)`` gives each column in turn."""
    frame = group.create_group(
        name,
        attributes=_encoded(
            _DATAFRAME, _index="_index", **{"column-order": list(names)}
        ),
    )
    _write_array(frame, "_index", np.asarray(index, dtype=object))
    for position, column_name in enumerate(names):
        _write_array(frame, column_name, column(position))
    return frame


def _write_scalar(group: Any, name: str, value: Any) -> None:
    if isinstance(value, str):
        array = group.create_array(
            name, shape=(), dtype=str, attributes=_encoded(_STRING)
        )
        array[()] = value
        return
    stored = np.asarray(value)
    array = group.create_array(
        name, shape=(), dtype=stored.dtype, attributes=_encoded(_NUMERIC_SCALAR)
    )
    array[()] = stored


def write_character_store(
    path: str | os.PathLike[str],
    cell_ids: Sequence[str] | np.ndarray,
    characters: Sequence[str],
    codes: np.ndarray,
    states: pd.DataFrame,
    *,
    missing_state: int = -1,
    unmodified_state: int = 0,
) -> None:
    """Write cells by characters of integer states as a TreeData ``.zarr`` store.

    The store holds ``obs`` with the cells' names, an empty ``var``,
    ``obsm['characters']`` with one int64 column per character, and in
    ``uns`` the ``missing_state``, the ``unmodified_state`` and
    ``character_states``, a table of ``character``, ``state`` and
    ``allele`` that names each edited state's allele; ``obst`` and ``vart``
    hold no trees.

    Args:
        path: The store to create; it must not exist.
        cell_ids: Each row's cell.
        characters: Each column's character.
        codes: The states, cells by characters.
        states: ``character``, ``state`` and ``allele`` of each edited state.
        missing_state: The state of a missing site.
        unmodified_state: The state of an unedited site.

    Raises:
        FileExistsError: If ``path`` exists.
        ValueError: If the cells, characters and codes do not fit together.
    """
    import zarr

    target = os.fspath(path)
    if os.path.lexists(target):
        raise FileExistsError(
            f"{target} exists; export the characters to a new path, since a Zarr "
            "store is a folder that an export would have to delete"
        )
    matrix = np.asarray(codes)
    cells = np.asarray(cell_ids, dtype=object)
    if matrix.shape != (len(cells), len(characters)):
        raise ValueError(
            "The codes must hold one row per cell and a column per character"
        )
    if len(set(characters)) != len(characters):
        raise ValueError("Each character must be named once")
    root = zarr.open_group(
        target,
        mode="w-",
        zarr_format=3,
        use_consolidated=False,
        attributes=_encoded(("treedata", "0.1.0")),
    )
    # obs and var hold no columns beside their index.
    for name, index in (("obs", cells), ("var", np.empty(0, dtype=object))):
        _write_frame(root, name, index, (), lambda position: np.empty(0))
    for name in ("obsm", "varm", "obsp", "varp", "layers", "uns"):
        root.create_group(name, attributes=_encoded(_DICT))
    # One character's column at a time is copied as int64 and written.
    _write_frame(
        root["obsm"],
        "characters",
        cells,
        list(characters),
        lambda position: np.ascontiguousarray(matrix[:, position], dtype=np.int64),
    )
    uns = root["uns"]
    _write_scalar(uns, "missing_state", np.int64(missing_state))
    _write_scalar(uns, "unmodified_state", np.int64(unmodified_state))
    state_columns = {
        "character": states["character"].to_numpy(dtype=object),
        "state": states["state"].to_numpy(dtype=np.int64),
        "allele": states["allele"].to_numpy(dtype=object),
    }
    names = list(state_columns)
    _write_frame(
        uns,
        "character_states",
        np.arange(len(states)).astype(str).astype(object),
        names,
        lambda position: state_columns[names[position]],
    )
    _write_scalar(root, "allow_overlap", np.bool_(False))
    _write_scalar(root, "alignment", "leaves")
    root.create_group("obst")
    root.create_group("vart")


# Trees


def h5td_tree_bytes(path: str) -> int:
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
                    f"The TreeData file {quoted_value(path)} keeps its trees in "
                    "'raw.treedata', as the earliest treedata releases did; read it "
                    "with treedata and write it again"
                )
            raise ValueError(
                f"The HDF5 file {quoted_value(path)} has no 'obst' element, so it "
                "holds no TreeData trees"
            )
        obst = handle["obst"]
        if (
            not isinstance(obst, h5py.Dataset)
            or h5py.check_string_dtype(obst.dtype) is None
        ):
            raise ValueError(
                f"The TreeData file {quoted_value(path)} does not store 'obst' as "
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


def _zarr_obst(path: str) -> Any:
    root = _zarr_root(path)
    if "obst" not in root:
        raise ValueError(
            f"The Zarr store {quoted_value(path)} has no 'obst' element, so it "
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


@dataclass(slots=True)
class TreeGraph:
    """One TreeData tree as nodes, edges and the two attributes read."""

    key: str
    nodes: list[str]
    times: dict[str, Any]
    edges: list[tuple[str, str]]
    lengths: list[Any]


def _number(value: Any, what: str, path: str, key: str) -> float:
    """A time or length as a float, NaN for an absent, null or NaN value.

    treedata writes NaN where a branch has no length and below it.
    """
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} has a {what} that "
            f"is not a number: {quoted_value(value)}"
        )
    number = float(value)
    if math.isinf(number):
        raise ValueError(
            f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} has a {what} that "
            f"is not finite: {quoted_value(value)}"
        )
    return number


def tree_in_preorder(
    graph: TreeGraph, path: str
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
                f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} gives the node "
                f"{quoted_value(target)} two parents, {quoted_value(parent_of[target])} and {quoted_value(source)}"
            )
        parent_of[target] = source
        children.setdefault(source, []).append((target, index))
    roots = [name for name in known if name not in parent_of]
    if len(roots) != 1:
        raise ValueError(
            f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} has "
            f"{len(roots)} roots, not one"
            + (
                f", such as {quoted_value(roots[:3])}"
                if roots
                else "; its edges form a cycle"
            )
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
            f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} has "
            f"{len(known) - len(names)} nodes that its root does not reach"
        )
    return names, parents, times, lengths


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
            f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} has {what} whose "
            f"attributes are not an object: {quoted_value(value)}"
        )
    return value


def _json_graphs(text: str, path: str) -> list[TreeGraph]:
    try:
        trees = json.loads(text, object_hook=_attributes_only)
    except (ValueError, RecursionError) as error:
        # Besides malformed JSON, Python refuses integers of over 4,300
        # digits and nesting deeper than it can follow.
        raise ValueError(
            f"The TreeData file {quoted_value(path)} has an 'obst' that is not JSON that "
            f"Python can read: {error}"
        ) from None
    if not _is_mapping(trees):
        raise ValueError(
            f"The TreeData file {quoted_value(path)} has an 'obst' that does not map tree "
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
                f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} has no 'nodes' "
                "and 'edges' that map names to objects"
            )
        edges: list[tuple[str, str]] = []
        lengths: list[Any] = []
        for source, targets in tree["edges"].items():
            if not _is_mapping(targets):
                raise ValueError(
                    f"The tree {quoted_value(key)} of the TreeData file {quoted_value(path)} has edges "
                    f"from {quoted_value(source)} that do not map targets to objects"
                )
            for target, attributes in targets.items():
                edges.append((str(source), str(target)))
                lengths.append(
                    _attributes(
                        attributes, f"an edge to {quoted_value(target)}", str(key), path
                    ).get("length")
                )
        times = {}
        for name, attributes in tree["nodes"].items():
            node = _attributes(
                attributes, f"a node {quoted_value(name)}", str(key), path
            )
            if "time" in node:
                times[str(name)] = node["time"]
        graphs.append(
            TreeGraph(
                key=str(key),
                nodes=[str(name) for name in tree["nodes"]],
                times=times,
                edges=edges,
                lengths=lengths,
            )
        )
    return graphs


def _parse_json_trees(
    raw: bytes, path: str, *, check: Callable[[str, int], None], held_bytes: int
) -> list[TreeGraph]:
    n_objects, n_members, most_held, n_characters, largest = _scan_json(
        np.frombuffer(raw, dtype=np.uint8),
        np.empty(_JSON_DEPTH, dtype=np.int64),
        np.empty(_JSON_DEPTH, dtype=np.bool_),
    )
    if n_objects < 0:
        raise ValueError(
            f"The TreeData file {quoted_value(path)} has an 'obst' whose objects and "
            f"arrays nest more than {_JSON_DEPTH} deep"
        )
    # Python stores text at one, two or four bytes a character, by the
    # largest character, which UTF-8's largest leading byte bounds.
    width = 1 if largest < 0xC4 else 2 if largest < 0xF0 else 4
    # Each node and edge is an object of the JSON, so the objects bound them.
    check(
        f"Parsing the trees of {quoted_value(path)}",
        held_bytes
        + len(raw)
        + 2 * width * n_characters
        + _JSON_BYTES_PER_OBJECT * n_objects
        + _JSON_BYTES_PER_MEMBER * n_members
        + _JSON_BYTES_PER_VALUE * most_held,
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"The TreeData file {quoted_value(path)} has an 'obst' that is not UTF-8"
        ) from None
    del raw
    graphs = _json_graphs(text, path)
    del text
    return graphs


def h5td_tree_graphs(
    path: str, *, check: Callable[[str, int], None], held_bytes: int
) -> list[TreeGraph]:
    """Read the trees of a TreeData ``.h5td`` file, one JSON string in ``obst``.

    Args:
        path: The file.
        check: Called with what a step holds, in bytes, before it holds it;
            raises if that does not fit the memory budget.
        held_bytes: Bytes that the caller holds beside the read.

    Raises:
        ValueError: If ``obst`` is not JSON that maps tree keys to trees of
            nodes and edges.
    """
    with h5py.File(path, "r") as handle:
        value = handle["obst"][()]
    raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return _parse_json_trees(raw, path, check=check, held_bytes=held_bytes)


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
    array: Any, what: str, *, check: Callable[[str, int], None], held_bytes: int
) -> tuple[np.ndarray, int]:
    """Read a Zarr text array chunk by chunk, counting its strings as they come.

    Returns:
        The values, and the bytes they hold.
    """
    blocks = []
    held = 0
    for block in _text_blocks(array):
        held += _held_text(block)
        check(what, held_bytes + 2 * held)
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
    check: Callable[[str, int], None],
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
            f"The tree {quoted_value(key)} of the TreeData store {quoted_value(path)} has a {quoted_value(name)} "
            f"attribute that does not hold one value per element"
        )
    if array.dtype.kind in "iufb":
        values = np.asarray(array[...])
        return values.tolist(), int(values.nbytes)
    texts, held = _zarr_texts(
        array,
        f"Reading the {quoted_value(name)} of the tree {quoted_value(key)} of {quoted_value(path)}",
        check=check,
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
                f"The tree {quoted_value(key)} of the TreeData store {quoted_value(path)} has a {quoted_value(name)} "
                f"that is not a number: {quoted_value(text)}"
            )
        try:
            decoded.append(json.loads(text))
        except (ValueError, RecursionError):
            raise ValueError(
                f"The tree {quoted_value(key)} of the TreeData store {quoted_value(path)} has a {quoted_value(name)} "
                f"that is not JSON that Python can read: {quoted_value(text)}"
            ) from None
    return decoded, held


def _zarr_graph(
    key: str,
    tree: Any,
    path: str,
    *,
    check: Callable[[str, int], None],
    held_bytes: int,
) -> tuple[TreeGraph, int]:
    import zarr

    nodes_array = tree["nodes"]
    edges_array = tree.get("edges")
    if (
        not isinstance(edges_array, zarr.Array)
        or len(edges_array.shape) != 2
        or (edges_array.shape[0] and edges_array.shape[1] != 2)
    ):
        raise ValueError(
            f"The tree {quoted_value(key)} of the TreeData store {quoted_value(path)} has no 'edges' "
            "array of source and target pairs"
        )
    what = f"Reading the tree {quoted_value(key)} of {quoted_value(path)}"
    nodes, held = _zarr_texts(nodes_array, what, check=check, held_bytes=held_bytes)
    edges, edge_held = _zarr_texts(
        edges_array, what, check=check, held_bytes=held_bytes + held
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
        check=check,
        held_bytes=held_bytes + held,
    )
    held += time_held
    lengths, length_held = _zarr_attribute(
        edge_attributes,
        "length",
        len(edges),
        key,
        path,
        check=check,
        held_bytes=held_bytes + held,
    )
    held += length_held
    names = [str(name) for name in nodes.tolist()]
    graph = TreeGraph(
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


def zarr_tree_nodes(path: str) -> int:
    """Count the nodes of the trees of a TreeData ``.zarr`` store, one group each.

    Raises:
        ValueError: If the store holds no ``obst``, holds it as one JSON
            string, as treedata wrote Zarr stores before 0.3.0, or a tree has
            no one-dimensional ``nodes`` array.
    """
    import zarr

    obst = _zarr_obst(path)
    if not isinstance(obst, zarr.Group):
        # Its size is unknown until it is read, and reading it can cost far
        # more than its text, so it is not read.
        raise ValueError(
            f"The TreeData store {quoted_value(path)} holds 'obst' as one JSON string, as "
            "treedata wrote Zarr stores before 0.3.0; write it again with "
            "treedata 0.3.0 or later, or as a .h5td file"
        )
    n_nodes = 0
    for key, tree in _zarr_tree_groups(obst):
        nodes = tree.get("nodes")
        if not isinstance(nodes, zarr.Array) or len(nodes.shape) != 1:
            raise ValueError(
                f"The tree {quoted_value(key)} of the TreeData store {quoted_value(path)} has no "
                "one-dimensional 'nodes' array"
            )
        n_nodes += int(nodes.shape[0])
    return n_nodes


def zarr_tree_graphs(
    path: str, *, check: Callable[[str, int], None], held_bytes: int
) -> list[TreeGraph]:
    """Read the trees of a TreeData ``.zarr`` store, one group per tree.

    Args:
        path: The store.
        check: Called with what a step holds, in bytes, before it holds it;
            raises if that does not fit the memory budget.
        held_bytes: Bytes that the caller holds beside the read.

    Raises:
        ValueError: If a tree's arrays are malformed, or the store changed
            since its trees were counted.
    """
    import zarr

    obst = _zarr_obst(path)
    if not isinstance(obst, zarr.Group):
        raise ValueError(
            f"The TreeData store {quoted_value(path)} changed while it was read"
        )
    graphs = []
    held = held_bytes
    for key, tree in _zarr_tree_groups(obst):
        graph, tree_held = _zarr_graph(key, tree, path, check=check, held_bytes=held)
        held += tree_held
        graphs.append(graph)
    return graphs
