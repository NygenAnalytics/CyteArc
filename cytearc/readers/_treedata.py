"""Read and write the character matrices of TreeData files.

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
"""

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ._text import as_text
from ..utils.arguments import quoted_value

__all__ = [
    "HDF5_SIGNATURE",
    "CharacterTable",
    "character_table",
    "is_zarr_store",
    "write_character_store",
]

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
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
                cells = index.read(start, stop, what)
                if any(cell is None for cell in cells.tolist()):
                    raise ValueError(f"{what} has a cell without a name")
                states = np.empty((stop - start, len(columns)), dtype=object)
                for position, column in enumerate(columns):
                    states[:, position] = column.read(start, stop, what)
                yield cells, states


class _open:
    """Open a TreeData file or store for reading, as a context manager."""

    def __init__(self, path: str, container: str) -> None:
        self._path, self._container = path, container
        self._handle: Any = None

    def __enter__(self) -> Any:
        if self._container == "hdf5":
            self._handle = h5py.File(self._path, "r")
            return self._handle
        import zarr

        try:
            # Each node's own metadata is read, as everywhere in CyteArc,
            # rather than consolidated metadata that may be stale.
            return zarr.open_group(self._path, mode="r", use_consolidated=False)
        except Exception as error:
            raise ValueError(
                f"The Zarr store {quoted_value(self._path)} cannot be opened: {error}"
            ) from None

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            self._handle.close()


# Where PEtracer's published TreeData files name the columns of their plain
# array of characters.
_EMBRYO_CHARACTERS = "embryo_characters"


def _dataframe(root: Any, path: str) -> tuple[_Column, list[_Column], str]:
    """The index, the columns and where the names come from of ``obsm['characters']``."""
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
        index = _column("_index", obs[as_text(index_name)], what)
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
            index,
            [
                _Column(name, _ColumnOf(node, position), "array")
                for position, name in enumerate(names)
            ],
            f"uns[{_EMBRYO_CHARACTERS!r}]",
        )
    if _encoding(node) != "dataframe":
        raise ValueError(f"{what} is no dataframe that CyteArc can read")
    index_name = node.attrs.get("_index")
    if index_name is None or as_text(index_name) not in node:
        raise ValueError(f"{what} names no cells")
    index = _column("_index", node[as_text(index_name)], what)
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
    return index, columns, "columns"


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
        n_cells = (
            int(index.node["values"].shape[0])
            if index.encoding == "nullable"
            else (
                int(index.node["codes"].shape[0])
                if index.encoding == "categorical"
                else int(index.node.shape[0])
            )
        )
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
