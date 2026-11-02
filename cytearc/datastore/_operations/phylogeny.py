"""Import cell phylogenies, read them back, and cut them into clones.

The operations join the leaves of each tree to cells through the shared
barcode join, save the forest as a ``cell_phylogeny`` artifact, and save the
clones that a cut defines as a ``clones`` artifact, which every clone summary
reads. Parsing lives in :mod:`cytearc.readers.phylogeny` and the cutting in
:mod:`cytearc.clones.phylogeny`.
"""

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...clones.assignments import CloneAssignment, ranked_clones
from ...clones.phylogeny import (
    TREE_SEPARATOR,
    CellPhylogeny,
    PhylogenyCut,
    clade_roots,
    clone_labels,
    collapsed_depths,
    leaf_time_ranges,
    node_times,
    phylogeny_cut,
    phylogeny_frames,
)
from ...metadata.arguments import PhylogenyCloneArguments, PhylogenyImportArguments
from ...metadata.rows import (
    read_array_rows_chunkwise,
    read_metadata_rows_chunkwise,
    text_width,
)
from ...metadata.selection import cell_selection_indices
from ...storage.artifact_writer import ArrayRequirement, artifact_transaction
from ...storage.arrays import (
    MetadataBlock,
    create_streamed_metadata_column,
    create_zarr_dataset,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.arrays import sort_categories
from ...utils.logging import logger
from ...utils.warnings import warn
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    file_sha256,
    store_sha256,
    warn_unmatched_barcodes,
)
from .clone_artifacts import plan_clones, write_clones, zarr_write_buffer_bytes

if TYPE_CHECKING:
    from ...readers.phylogeny import FileTrees, TreeFileLayout
    from .clones import _CloneOperationsMixin as _PhylogenyOperationsBase
else:
    _PhylogenyOperationsBase = object

# Extensions that a tree file's name may end with, which its trees' names
# leave out, after ``.gz``.
_TREE_EXTENSIONS = (
    ".nwk",
    ".newick",
    ".nhx",
    ".tree",
    ".trees",
    ".tre",
    ".txt",
    ".h5td",
    ".h5ad",
    ".h5",
    ".zarr",
)
# The separator of a tree's name and its key when a file holds several trees.
_KEY_SEPARATOR = "_"
# Leaves are joined to cells this many at a time.
_JOIN_BATCH = 1 << 16
# Each leaf joined holds its barcode in a series, the cell ID that the
# prefix and suffix make of it, its position among the cells and the masks
# of the join, besides the text of its barcode and ID.
_JOIN_BYTES_PER_LEAF = 300
# The forest holds, for each node, its name's slot, parent, time, branch
# length, given time, cell and whether it is a leaf, its file's cell beside
# it, and while times resolve each node's tree and whether it has a time.
_FOREST_BYTES_PER_NODE = 88
# Reading a cell's ID holds it at its stored width, four bytes a character,
# and as a Python string of at most 80 bytes and four a character; and the
# read decodes a whole stored chunk of the column, and its compressed copy.
_ID_BYTES_PER_CELL = 88
_ID_CHUNK_COPIES = 2
# A node name is written at the width of the longest, four bytes a character,
# in chunks of at most this many rows and bytes; writing a chunk holds the
# block of names, Zarr's chunk and its compressed copy.
_NAME_BLOCK_ROWS = 100_000
_NAME_CHUNK_BYTES = 16 << 20
_NAME_WRITE_COPIES = 3
# Reading a chunk of names holds its compressed and its decoded bytes.
_NAME_READ_COPIES = 2
# Reading a phylogeny for a cut holds, for each node, its parent, time, cell,
# clade root, depth, the count of its children and the masks of the cut.
_CUT_BYTES_PER_NODE = 64
# Each clone holds its root, its label as a Python string beside its stored
# name, and its place in ranking the clones by size and label.
_CUT_BYTES_PER_CLONE = 600
# Each cell of a cut holds its clone code, its label's slot, and its stored
# label four bytes a character while the clones are written.
_CUT_BYTES_PER_CELL = 64
# Loading a phylogeny holds, for each node, its tree, position, parent, time,
# depth and leaf columns, and the slots of its name and cell.
_LOAD_BYTES_PER_NODE = 160
# A fixed allowance for opening files and stores and for each tree.
_FIXED_BYTES = 4 << 20
_BYTES_PER_TREE = 2_000


@dataclass(frozen=True, slots=True)
class _ImportedForest:
    """The trees of every file of an import, as the artifact stores them."""

    names: np.ndarray
    parent: np.ndarray
    times: np.ndarray
    cells: np.ndarray
    tree_offsets: np.ndarray
    tree_names: np.ndarray
    unique_names: np.ndarray
    n_leaves_without_a_cell: int
    n_unnamed_leaves: int


def _tree_file_sha256(path: str | os.PathLike[str]) -> str:
    """Identify a tree file by its bytes, or a Zarr store by its files.

    A directory that is not a Zarr store is refused before it is walked.
    """
    from ...readers.phylogeny import tree_container

    if os.path.isdir(path) and tree_container(Path(os.fspath(path))) == "zarr":
        return store_sha256(path)
    return file_sha256(path)


def _tree_stem(file: BarcodeFile) -> str:
    """A file's name without ``.gz`` and its tree extension."""
    name = Path(os.fspath(file.path)).name
    if name.lower().endswith(".gz") and len(name) > 3:
        name = name[:-3]
    for extension in _TREE_EXTENSIONS:
        if name.lower().endswith(extension) and len(name) > len(extension):
            return name[: -len(extension)]
    return name


def _tree_names(file: BarcodeFile, stem: str, trees: "FileTrees") -> list[str]:
    """Name a file's trees: the file's prefix and stem, and a key for several."""
    base = (file.barcode_prefix or "") + stem
    if len(trees.tree_keys) == 1:
        names = [base]
    else:
        names = [f"{base}{_KEY_SEPARATOR}{key}" for key in trees.tree_keys]
    for name in names:
        if TREE_SEPARATOR in name:
            raise ValueError(
                f"The tree {name!r} of {os.fspath(file.path)!r} has a "
                f"{TREE_SEPARATOR!r} in its name, which separates a tree's name "
                f"from a node's in clone labels; remove the {TREE_SEPARATOR!r} "
                "from barcode_prefix or from the tree's key in the TreeData file"
            )
    return names


def _time_unit(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("time_unit must be text, such as 'days', or None")
    unit = value.strip()
    if not unit:
        raise ValueError(
            "time_unit must name the unit of the trees' times, such as 'days', "
            "or be None when their lengths are not times"
        )
    return unit


def _unique_names(names: np.ndarray, tree_offsets: np.ndarray) -> np.ndarray:
    """Whether each tree's nodes all have names, none of them shared."""
    unique = np.zeros(len(tree_offsets) - 1, dtype=bool)
    for tree, (start, stop) in enumerate(
        zip(tree_offsets[:-1].tolist(), tree_offsets[1:].tolist(), strict=True)
    ):
        tree_names = names[start:stop]
        unique[tree] = not bool(np.any(tree_names == "")) and bool(
            pd.Index(tree_names).is_unique
        )
    return unique


def _name_rows(width: int, n_names: int) -> int:
    """Rows of node names to write and read at once: a chunk of at most 16 MiB."""
    return int(
        max(1, min(_NAME_BLOCK_ROWS, n_names, _NAME_CHUNK_BYTES // max(4 * width, 1)))
    )


def _names_width(names: np.ndarray) -> int:
    return max(1, max(map(len, names.tolist()), default=1))


def _write_bytes(width: int, n_names: int) -> int:
    """What writing names of ``width`` holds: a block, its chunk, and its copy."""
    return _NAME_WRITE_COPIES * 4 * width * _name_rows(width, n_names)


def _write_names(group: zarr.Group, names: np.ndarray, width: int) -> None:
    rows = _name_rows(width, len(names))
    dtype = np.dtype(f"U{width}")
    create_streamed_metadata_column(
        group,
        "names",
        shape=len(names),
        dtype=dtype,
        chunkSize=rows,
        blocks=(
            MetadataBlock(start, names[start : start + rows].astype(dtype))
            for start in range(0, len(names), rows)
        ),
    )


def _write_numbers(group: zarr.Group, name: str, values: np.ndarray) -> None:
    chunks = (min(max(len(values), 1), 1_000_000),)
    output = create_zarr_dataset(group, name, chunks, values.dtype, values.shape)
    output[...] = values


def _read(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[...])


def _files_text(n_files: int) -> str:
    return "the tree file" if n_files == 1 else "the tree files"


class _PhylogenyOperationsMixin(_PhylogenyOperationsBase):
    def _phylogeny_cell_ids(self, cell_index: np.ndarray) -> np.ndarray:
        return np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )

    def _phylogeny_id_bytes(self, n_cells: int) -> int:
        """What reading ``n_cells`` cell IDs holds, at their stored width.

        The IDs are read a stored chunk at a time, and a chunk may hold many
        more rows than the cells read.
        """
        ids = self.cells._get_array("ids")
        itemsize = int(ids.dtype.itemsize)
        chunk_rows = int(ids.chunks[0]) if ids.chunks else int(ids.shape[0])
        return n_cells * (_ID_BYTES_PER_CELL + 2 * itemsize) + (
            _ID_CHUNK_COPIES * chunk_rows * itemsize
        )

    def _check_phylogeny_budget(self, what: str, held: int) -> None:
        if held > self.memoryBytes:
            raise MemoryError(
                f"{what} holds about {held} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )

    def _join_leaves(
        self,
        join: CellBarcodeJoin,
        file: BarcodeFile,
        trees: "FileTrees",
        *,
        held: int,
    ) -> np.ndarray:
        """Each node's cell, as a position among the cells, -1 for none.

        Leaves without a name name no cell; other leaves are joined
        ``_JOIN_BATCH`` at a time.
        """
        cells = np.full(trees.n_nodes, -1, dtype=np.int64)
        leaves = np.flatnonzero(trees.leaf & (trees.names != ""))
        for start in range(0, len(leaves), _JOIN_BATCH):
            batch = leaves[start : start + _JOIN_BATCH]
            barcodes = trees.names[batch]
            self._check_phylogeny_budget(
                f"Joining {len(batch)} leaves to cells",
                held
                + _JOIN_BYTES_PER_LEAF * len(batch)
                + 2 * sum(map(sys.getsizeof, barcodes.tolist())),
            )
            cells[batch] = join.cells(file, barcodes)
        return cells

    def _imported_forest(
        self,
        files: Sequence[BarcodeFile],
        stems: Sequence[str],
        layouts: Sequence["TreeFileLayout"],
        join_cells: np.ndarray,
    ) -> tuple[_ImportedForest, CellBarcodeJoin, int]:
        """Read every file's trees and join their leaves to the cells.

        Returns:
            The forest, the join, and the bytes that both hold.
        """
        from ...readers.phylogeny import read_trees

        held = _FIXED_BYTES
        # Every file's trees are held until the forest is built, so the bound
        # of all of them and the largest peak of one are checked before any is
        # read; the reader checks each file again as it reads.
        self._check_phylogeny_budget(
            f"Reading the trees of {len(layouts)} files",
            held
            + sum(layout.trees_bytes for layout in layouts)
            + max(layout.peak_bytes - layout.trees_bytes for layout in layouts),
        )
        read: list[FileTrees] = []
        for layout in layouts:
            trees = read_trees(layout, memory_bytes=self.memoryBytes, held_bytes=held)
            held += trees.held_bytes
            read.append(trees)
        n_nodes = sum(trees.n_nodes for trees in read)
        n_trees = sum(len(trees.tree_keys) for trees in read)
        held += _FOREST_BYTES_PER_NODE * n_nodes + _BYTES_PER_TREE * n_trees
        # Reading the cells' IDs holds them at their stored width and decodes
        # a stored chunk of them; the join then holds its lookup and the IDs
        # as text, which it counts, beside what the trees hold.
        self._check_phylogeny_budget(
            f"Joining {n_nodes} nodes of {n_trees} trees to {len(join_cells)} cells",
            held
            + LOOKUP_BYTES_PER_CELL * len(join_cells)
            + self._phylogeny_id_bytes(len(join_cells)),
        )
        cell_ids = self._phylogeny_cell_ids(join_cells)
        join = CellBarcodeJoin(cell_ids, memory_bytes=self.memoryBytes, held_bytes=held)
        held = join.held_bytes
        # Each file's leaves are joined to cells, and its trees named.
        file_cells = []
        listed: list[tuple[str, int, int, int]] = []
        for position, (file, stem, trees) in enumerate(
            zip(files, stems, read, strict=True)
        ):
            file_cells.append(self._join_leaves(join, file, trees, held=held))
            for name, tree_start, tree_stop in zip(
                _tree_names(file, stem, trees),
                trees.tree_offsets[:-1].tolist(),
                trees.tree_offsets[1:].tolist(),
                strict=True,
            ):
                listed.append((name, position, tree_start, tree_stop))
        self._refuse_repeated_names(listed, files)
        tree_names = [name for name, *_where in listed]
        # The forest holds the trees in the natural order of their names.
        rank = {name: index for index, name in enumerate(sort_categories(tree_names))}
        listed.sort(key=lambda item: rank[item[0]])
        tree_names = [name for name, *_where in listed]
        names = np.empty(n_nodes, dtype=object)
        parent = np.empty(n_nodes, dtype=np.int64)
        length = np.empty(n_nodes, dtype=np.float64)
        given = np.empty(n_nodes, dtype=np.float64)
        cells = np.empty(n_nodes, dtype=np.int64)
        leaf = np.empty(n_nodes, dtype=bool)
        offsets = [0]
        start = 0
        for _name, position, tree_start, tree_stop in listed:
            trees = read[position]
            stop = start + tree_stop - tree_start
            part = slice(tree_start, tree_stop)
            names[start:stop] = trees.names[part]
            parent[start:stop] = np.where(
                trees.parent[part] >= 0, trees.parent[part] - tree_start + start, -1
            )
            length[start:stop] = trees.length[part]
            given[start:stop] = trees.time[part]
            leaf[start:stop] = trees.leaf[part]
            cells[start:stop] = file_cells[position][part]
            offsets.append(stop)
            start = stop
        del read, file_cells
        self._refuse_repeated_cells(cells, names, cell_ids)
        tree_offsets = np.asarray(offsets, dtype=np.int64)
        named_leaves = leaf & (names != "")
        forest = _ImportedForest(
            names=names,
            parent=parent,
            times=node_times(parent, length, given, tree_offsets),
            cells=cells,
            tree_offsets=tree_offsets,
            tree_names=np.asarray(tree_names, dtype=object),
            unique_names=_unique_names(names, tree_offsets),
            n_leaves_without_a_cell=int(np.count_nonzero(leaf & (cells < 0))),
            n_unnamed_leaves=int(np.count_nonzero(leaf & ~named_leaves)),
        )
        return forest, join, held

    @staticmethod
    def _refuse_repeated_names(
        listed: Sequence[tuple[str, int, int, int]], files: Sequence[BarcodeFile]
    ) -> None:
        """Refuse two trees of one name: their clones' labels would merge."""
        # A file names its own trees apart, so a repeat comes from two files.
        seen: dict[str, int] = {}
        for name, position, *_where in listed:
            first = seen.setdefault(name, position)
            if first != position:
                raise ValueError(
                    f"Two trees are named {name!r}, of "
                    f"{os.fspath(files[first].path)!r} and "
                    f"{os.fspath(files[position].path)!r}, so their clones would "
                    "share labels; give the files different names or barcode_prefix "
                    "values"
                )

    @staticmethod
    def _refuse_repeated_cells(
        cells: np.ndarray, names: np.ndarray, cell_ids: np.ndarray
    ) -> None:
        """Refuse two leaves that name one cell, since a cell is one leaf."""
        named = np.flatnonzero(cells >= 0)
        order = named[np.argsort(cells[named], kind="stable")]
        repeated = order[1:][cells[order[1:]] == cells[order[:-1]]]
        if not len(repeated):
            return
        repeated_cells = np.unique(cells[repeated])
        examples = []
        for cell in repeated_cells[:3].tolist():
            leaves = names[cells == cell].tolist()
            examples.append(
                f"{str(cell_ids[cell])!r} by "
                + " and ".join(repr(str(leaf)) for leaf in leaves)
            )
        raise ValueError(
            f"{len(repeated_cells)} cells are named by several leaves, "
            f"for example {'; '.join(examples)}. A cell is one leaf; a file of "
            "several trees over the same cells, such as bootstrap replicates, "
            "holds alternatives rather than one forest, so import one tree"
        )

    @validation_scoped
    def _clones_import_phylogeny(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        cell_selection: ArtifactRef,
        time_unit: str | None,
        format: str | None = None,
        barcode_prefix: str | Sequence[str | None] | None = None,
        barcode_suffix: str | Sequence[str | None] | None = None,
        barcode_pattern: str | Sequence[str | None] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import cell phylogenies whose leaves are cells.

        Lineage recorders such as PEtracer and DNA Typewriter, and tools such
        as Cassiopeia and LAML, reconstruct trees whose leaves are cells and
        whose internal nodes are their ancestors. The import reads Newick
        files, plain or gzip-compressed, with one tree or several, such as one
        tree per clone, and TreeData ``.h5td`` files and ``.zarr`` stores,
        joins each leaf to a selected cell by barcode, and saves the forest as
        a ``cell_phylogeny`` artifact that
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.phylogeny_clones`
        cuts into clones. Newick is read without recursion, so trees of a
        million leaves on one line read like small ones, with bracket comments,
        quoted labels, polytomies, single-child nodes, zero and negative
        branch lengths and a length above the root. An unquoted number on an
        internal node is a support value, as treedata reads it, not a name.

        A node's time runs from its tree's root, plus any length above the
        root, in the unit that ``time_unit`` declares, since no file states
        it. A TreeData node's ``time`` is preferred over the time its edge's
        ``length`` implies, and a node below a branch without a length has no
        time.

        A leaf names the cell whose ID is its file's ``barcode_prefix``, the
        leaf's name and ``barcode_suffix``; ``barcode_pattern`` keeps the part
        of the name that its ``barcode`` group matches first. Leaves that name
        no selected cell are counted, saved in the ``n_file_barcodes``,
        ``n_unmatched_barcodes`` and ``unmatched_barcode_examples`` attributes
        and named in a warning, which a reused import gives again; the
        selected cells in no tree are counted in ``n_cells_in_no_tree``. Two
        leaves that name one cell raise ``ValueError``.

        Each tree is named by its file's ``barcode_prefix`` and the file's
        name without ``.gz`` and its extension, such as ``E7.5-R1-C1`` for
        ``E7.5-R1-C1.nwk.gz``; a file of several trees adds ``_`` and each
        tree's position from 1, or its TreeData key. The names, which must
        differ, start the labels of clones cut from several trees.

        Args:
            paths: One tree file, or several.
            cell_selection: The cells that the leaves name.
            time_unit: The unit of the trees' times, such as ``"days"``, or
                None when their lengths are not times; then the trees are cut
                at depths only.
            format: ``"newick"`` or ``"treedata"``, or None to recognise each
                file: Zarr stores and HDF5 files hold TreeData.
            barcode_prefix: Text before each leaf's name in the cell IDs: one
                for every file, or one per file.
            barcode_suffix: Text after each leaf's name in the cell IDs.
            barcode_pattern: A regular expression whose named group
                ``barcode`` is the part of each leaf's name that names the
                cell; a name it does not match names no cell.
            invalidate_cache: Import again instead of reusing saved trees.

        Returns:
            A complete datastore-scoped ``cell_phylogeny`` artifact.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a file is not a well-formed tree file, a TreeData
                tree is not a tree, two trees share a name, two leaves name
                one cell, no leaf names a selected cell, or the selection
                holds no cell.
            FileNotFoundError: If a file does not exist.
            PermissionError: If the store is read-only and the trees are not
                saved yet.
            MemoryError: If the import does not fit the memory budget.
        """
        from ...readers.phylogeny import tree_file_layout

        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        unit = _time_unit(time_unit)
        if format is not None and not isinstance(format, str):
            raise TypeError("format must be a format name or None")
        bool_argument(invalidate_cache, "invalidate_cache")
        files = barcode_files(
            paths,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            what="tree file",
            identify=_tree_file_sha256,
        )
        layouts = [tree_file_layout(file.path, format) for file in files]
        stems = [_tree_stem(file) for file in files]
        arguments = PhylogenyImportArguments(
            cell_selection=cell_selection,
            sources=tuple(
                {**layout.record(), **file.record(), "tree_name": stem}
                for file, layout, stem in zip(files, layouts, stems, strict=True)
            ),
            time_unit=unit,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="datastore",
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("parent", dtype_kind="i"),
                ArrayRequirement("time", dtype_kind="f"),
                ArrayRequirement("cells", dtype_kind="i"),
                "names",
                ArrayRequirement("tree_offsets", dtype_kind="i"),
                "tree_names",
                ArrayRequirement("tree_unique_names", dtype_kind="b"),
            ),
            required_attributes=(
                "time_unit",
                "n_file_barcodes",
                "n_unmatched_barcodes",
                "unmatched_barcode_examples",
                "n_cells_in_no_tree",
            ),
            reuse_validator=_has_forest,
        )
        files_text = _files_text(len(files))
        if planned.reused:
            # Saved trees warn as they did when they were imported.
            warn_unmatched_barcodes(
                artifact_group(self.zw, planned.ref).attrs, files_text
            )
            return planned.ref
        self._require_writable(arguments.operation)
        # The cells are read only when the trees are imported, not reused.
        cell_index = self._clone_selection_indices(cell_selection)
        forest, join, held = self._imported_forest(files, stems, layouts, cell_index)
        n_cells = len(cell_index)
        in_tree = np.zeros(n_cells, dtype=bool)
        in_tree[forest.cells[forest.cells >= 0]] = True
        n_cells_in_trees = int(np.count_nonzero(in_tree))
        if not n_cells_in_trees:
            if not join.examples:
                raise ValueError(
                    f"No leaf of {files_text} names a selected cell, since no "
                    "leaf has a name; name the leaves by their cells' barcodes"
                )
            raise ValueError(
                f"No leaf of {files_text} names a selected cell, for example "
                + ", ".join(repr(value) for value in join.examples)
                + ". Check barcode_prefix, barcode_suffix, barcode_pattern and "
                "cell_selection."
            )
        is_leaf = np.ones(len(forest.parent), dtype=bool)
        is_leaf[forest.parent[forest.parent >= 0]] = False
        earliest, latest = leaf_time_ranges(forest.times, is_leaf, forest.tree_offsets)
        spread = latest - earliest
        attributes = {
            **join.attributes(),
            "time_unit": unit,
            "n_trees": len(forest.tree_names),
            "n_nodes": len(forest.parent),
            "n_leaves": int(np.count_nonzero(is_leaf)),
            "n_cells_in_trees": n_cells_in_trees,
            "n_cells_in_no_tree": n_cells - n_cells_in_trees,
            "n_leaves_without_a_cell": forest.n_leaves_without_a_cell,
            "n_unnamed_leaves": forest.n_unnamed_leaves,
            # How far the trees are from ultrametric, in time_unit.
            "leaf_time_spread": (
                float(np.max(spread)) if np.all(np.isfinite(spread)) else None
            ),
        }
        width = _names_width(forest.names)
        # The arrays are written beside what Zarr's writes hold in file
        # buffers.
        self._check_phylogeny_budget(
            f"Writing the names of {len(forest.names)} nodes",
            held + _write_bytes(width, len(forest.names)) + zarr_write_buffer_bytes(),
        )
        with artifact_transaction(self.zw, planned) as group:
            group.attrs.update(attributes)
            _write_numbers(group, "parent", forest.parent)
            _write_numbers(group, "time", forest.times)
            _write_numbers(group, "cells", forest.cells)
            _write_names(group, forest.names, width)
            _write_numbers(group, "tree_offsets", forest.tree_offsets)
            _write_names_column(group, "tree_names", forest.tree_names)
            _write_numbers(group, "tree_unique_names", forest.unique_names)
            _write_numbers(group, "tree_earliest_leaf_time", earliest)
            _write_numbers(group, "tree_latest_leaf_time", latest)
        warn_unmatched_barcodes(attributes, files_text)
        logger.info(
            f"Saved {len(forest.tree_names)} trees of {len(forest.parent)} nodes; "
            f"{n_cells_in_trees} of {n_cells} selected cells are leaves, and "
            f"{forest.n_leaves_without_a_cell} leaves name no selected cell"
        )
        return planned.ref

    def _phylogeny_group(self, phylogeny: ArtifactRef) -> zarr.Group:
        """Return a complete ``cell_phylogeny`` artifact's group."""
        if not isinstance(phylogeny, ArtifactRef):
            raise TypeError("phylogeny must be an ArtifactRef")
        if phylogeny.kind != PhylogenyImportArguments.artifact_kind:
            raise ValueError(
                f"phylogeny must be a cell_phylogeny artifact, not {phylogeny.kind!r}"
            )
        require_complete_artifact(self.zw, phylogeny)
        return artifact_group(self.zw, phylogeny)

    def _phylogeny_cells(self, phylogeny: ArtifactRef) -> ArtifactRef:
        return inspect_artifact(self.zw, phylogeny).input_ref("cell_selection")

    @validation_scoped
    def _clones_load_phylogeny(self, phylogeny: ArtifactRef) -> CellPhylogeny:
        """Read saved phylogenies as a table of nodes and a table of trees.

        Args:
            phylogeny: A ``cell_phylogeny`` artifact, from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_phylogeny`.

        Returns:
            The trees in the natural order of their names, with one row per
            node, each tree in preorder, and one row per tree.

        Raises:
            TypeError: If ``phylogeny`` is not an ArtifactRef.
            ValueError: If it is not a ``cell_phylogeny`` artifact.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as an import that
                stopped before it finished leaves it.
            MemoryError: If the tables do not fit the memory budget.
        """
        group = self._phylogeny_group(phylogeny)
        names_array = as_zarr_array(group["names"], name="names")
        n_nodes = int(names_array.shape[0])
        width = text_width(names_array)
        selection = self._phylogeny_cells(phylogeny)
        cell_index = cell_selection_indices(self.zw, selection)
        n_named = _count(group, "n_cells_in_trees")
        # Names are read a stored chunk at a time: its bytes and their copy.
        rows = max(1, int(names_array.chunks[0]))
        held = (
            _FIXED_BYTES
            + _LOAD_BYTES_PER_NODE * n_nodes
            + self._phylogeny_id_bytes(n_named)
            + 8 * len(cell_index)
            + rows * (8 + _NAME_READ_COPIES * 4 * width)
        )
        self._check_phylogeny_budget(f"Loading the phylogeny of {n_nodes} nodes", held)
        cells = _read(group, "cells")
        names = np.empty(n_nodes, dtype=object)
        name_bytes = 0
        for start in range(0, n_nodes, rows):
            block = np.asarray(names_array[start : start + rows]).tolist()
            name_bytes += sum(map(sys.getsizeof, block))
            self._check_phylogeny_budget(
                f"Loading the names of {n_nodes} nodes", held + name_bytes
            )
            names[start : start + len(block)] = block
        cell_ids = np.full(n_nodes, None, dtype=object)
        named = np.flatnonzero(cells >= 0)
        cell_ids[named] = self._phylogeny_cell_ids(cell_index[cells[named]])
        nodes, trees = phylogeny_frames(
            parent=_read(group, "parent"),
            times=_read(group, "time"),
            names=names,
            cell_ids=cell_ids,
            tree_offsets=_read(group, "tree_offsets"),
            tree_names=_read(group, "tree_names").astype(object),
        )
        unit = group.attrs.get("time_unit")
        return CellPhylogeny(
            nodes=nodes, trees=trees, time_unit=None if unit is None else str(unit)
        )

    @validation_scoped
    def _clones_phylogeny_clones(
        self,
        phylogeny: ArtifactRef,
        *,
        time: float | None = None,
        depth: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Cut cell phylogenies into clones at a time or at a depth.

        Cutting a tree at ``time`` makes a clone of each clade whose root is
        the first node at or past that time on the way down from the tree's
        root: the cells that descend from one lineage alive at that time,
        counted in the trees' ``time_unit`` from each root plus any length
        above it. Cutting at ``depth`` counts branchings below each root
        instead, after merging each node that has one child with that child,
        so a root with one child, as PEtracer's trees have, starts at depth 0
        like its child. Each child of a polytomy that the cut passes is a
        clone of its own. Cutting at ``depth=0``, or at a time at or before
        every root's time, such as ``time=0`` when no negative length lies
        above a root, makes each tree one clone.

        A cell whose leaf ends before the cut, earlier than ``time`` or fewer
        than ``depth`` branchings below its root, is in no clone; the
        artifact counts such cells in ``n_cells_before_cut``, and a warning
        names how many. Leaves of a tree that is not ultrametric end at
        different times, and the ``leaf_time_spread`` attribute holds the
        largest gap between a tree's earliest and latest leaf. Selected cells
        in no tree have no clone either.

        A clone is named by its root's name when every node of its tree has a
        name of its own, as PEtracer's and Cassiopeia's trees do, and
        otherwise by ``node`` and its root's position in the tree, from 0 at
        the tree's root, as the phylogeny's ``nodes`` table numbers them. With
        several trees, each label starts with its tree's name and ``/``, so
        the clones of one tree per clone stay apart. The artifact is a
        ``clones`` artifact like any other: codes number the clones by size,
        largest first, then by label.

        Args:
            phylogeny: A ``cell_phylogeny`` artifact.
            time: The time of the cut from each tree's root, in the trees'
                ``time_unit``.
            depth: The branchings below each tree's root.
            invalidate_cache: Cut again instead of reusing saved clones.

        Returns:
            A complete datastore-scoped ``clones`` artifact of the
            phylogeny's cells.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``phylogeny`` is not a ``cell_phylogeny``
                artifact, both or neither of ``time`` and ``depth`` are
                given, ``time`` is not finite, ``depth`` is negative, the
                trees were imported without a ``time_unit`` and ``time`` is
                given, or a node has no time.
            KeyError: If the store holds no such ``phylogeny`` artifact.
            RuntimeError: If the ``phylogeny`` artifact is incomplete.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If the cut does not fit the memory budget.
        """
        cut = phylogeny_cut(time, depth)
        bool_argument(invalidate_cache, "invalidate_cache")
        group = self._phylogeny_group(phylogeny)
        unit = group.attrs.get("time_unit")
        if cut.time is not None and unit is None:
            raise ValueError(
                "The phylogeny was imported with time_unit=None, so its lengths "
                "are not times; cut it at a depth, or import it again with its "
                "time unit"
            )
        selection = self._phylogeny_cells(phylogeny)
        arguments = PhylogenyCloneArguments(
            phylogeny=phylogeny,
            cell_selection=selection,
            time=cut.time,
            depth=cut.depth,
            invalidate_cache=invalidate_cache,
        )
        n_cells = len(cell_selection_indices(self.zw, selection))
        planned = plan_clones(
            self.zw,
            arguments,
            selection,
            n_cells,
            with_probability=False,
            required_attributes=("n_clones", "n_cells_before_cut"),
        )
        if planned.reused:
            return planned.ref
        names_array = as_zarr_array(group["names"], name="names")
        n_nodes = int(names_array.shape[0])
        width = text_width(names_array)
        held = _FIXED_BYTES + _CUT_BYTES_PER_NODE * n_nodes
        self._check_phylogeny_budget(f"Cutting the phylogeny of {n_nodes} nodes", held)
        self._require_writable(arguments.operation)
        parent = _read(group, "parent")
        cells = _read(group, "cells")
        tree_offsets = _read(group, "tree_offsets")
        if cut.time is not None:
            times = _read(group, "time")
            n_timeless = int(np.count_nonzero(np.isnan(times)))
            if n_timeless:
                nodes = (
                    "1 node of the phylogeny has no time, since a branch above it"
                    if n_timeless == 1
                    else f"{n_timeless} nodes of the phylogeny have no time, since "
                    "a branch above them"
                )
                raise ValueError(f"{nodes} has no length; cut it at a depth instead")
            past = times >= cut.time
            del times
        else:
            past = collapsed_depths(parent) >= cut.depth
        roots = clade_roots(parent, past)
        del past
        leaf_cells = np.flatnonzero(cells >= 0)
        cell_roots = roots[leaf_cells]
        assigned = cell_roots >= 0
        n_before = int(np.count_nonzero(~assigned))
        if not assigned.any():
            raise ValueError(
                f"No cell of the trees is in a clone at {_cut_text(cut, unit)}: "
                f"each of their {len(leaf_cells)} cells ends before the cut. "
                "Cut at an earlier time or a smaller depth"
            )
        clones, by_cell = np.unique(cell_roots[assigned], return_inverse=True)
        n_clones = len(clones)
        tree_names = _read(group, "tree_names").astype(object)
        # A label is a name or a position, after its tree's name and "/".
        label_width = max(width, len(f"node{n_nodes}"))
        if len(tree_names) > 1:
            label_width += 1 + max(map(len, tree_names.tolist()))
        # The clones are labelled and then written beside what Zarr's writes
        # hold in file buffers.
        held += (
            n_clones * (_CUT_BYTES_PER_CLONE + 8 * (1 + label_width))
            + n_cells * (_CUT_BYTES_PER_CELL + 4 * label_width)
            + max(1, int(names_array.chunks[0])) * _NAME_READ_COPIES * 4 * width
            + zarr_write_buffer_bytes()
        )
        self._check_phylogeny_budget(
            f"Labelling {n_clones} clones of {n_cells} cells", held
        )
        labels = clone_labels(
            clones,
            tree_offsets=tree_offsets,
            tree_names=tree_names,
            unique_names=_read(group, "tree_unique_names"),
            names=np.asarray(read_array_rows_chunkwise(names_array, clones)).tolist(),
        )
        codes = np.full(n_cells, -1, dtype=np.int64)
        codes[cells[leaf_cells[assigned]]] = by_cell
        ranked, clone_ids, clone_sizes = ranked_clones(codes, labels.tolist())
        attributes: dict[str, Any] = {
            "n_clones": len(clone_ids),
            "n_cells_before_cut": n_before,
            "time_unit": None if unit is None else str(unit),
            "leaf_time_spread": group.attrs.get("leaf_time_spread"),
        }
        write_clones(
            self.zw,
            planned,
            CloneAssignment(
                codes=ranked,
                clone_ids=clone_ids,
                clone_sizes=clone_sizes,
                probabilities=None,
                n_cells_in_several_clones=0,
            ),
            attributes,
        )
        if n_before:
            warn(
                f"{n_before} of the {len(leaf_cells)} cells in the trees end before "
                f"the cut at {_cut_text(cut, unit)}, so they are in no clone"
            )
        logger.info(
            f"Cut {n_clones} clones of {len(leaf_cells) - n_before} cells at "
            f"{_cut_text(cut, unit)}"
        )
        return planned.ref


def _cut_text(cut: PhylogenyCut, unit: Any) -> str:
    """Where a cut falls, such as ``time 6.5 days`` or ``depth 2``."""
    if cut.time is not None:
        return f"time {cut.time:g} {unit}"
    return f"depth {cut.depth}"


def _write_names_column(group: zarr.Group, name: str, values: np.ndarray) -> None:
    width = max(1, max(map(len, values.tolist()), default=1))
    create_streamed_metadata_column(
        group,
        name,
        shape=len(values),
        dtype=np.dtype(f"U{width}"),
        chunkSize=max(len(values), 1),
        blocks=[MetadataBlock(0, values.astype(f"U{width}"))],
    )


def _count(group: zarr.Group, name: str) -> int:
    """A count that a saved phylogeny records as an attribute."""
    value = group.attrs.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"The phylogeny's {name!r} attribute is not a count")
    return value


def _has_forest(_ref: ArtifactRef, group: zarr.Group) -> bool:
    """Whether a saved phylogeny's arrays hold one forest of one tree list.

    Planning has found each array; this checks that their lengths agree.
    """
    parent, time, cells, names, offsets, tree_names, unique = (
        as_zarr_array(group[name], name=name)
        for name in (
            "parent",
            "time",
            "cells",
            "names",
            "tree_offsets",
            "tree_names",
            "tree_unique_names",
        )
    )
    return (
        parent.shape == time.shape == cells.shape == names.shape
        and offsets.shape[0] == tree_names.shape[0] + 1
        and tree_names.shape == unique.shape
    )
