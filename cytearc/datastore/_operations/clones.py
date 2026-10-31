import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...clones.assignments import (
    MULTIPLE_POLICIES,
    CloneAssigner,
    CloneAssignment,
    Clones,
    clone_frames,
    MultiplePolicy,
    clone_label_texts,
)
from ...metadata.arguments import CloneColumnArguments, CloneImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.rows import (
    array_row_selection_parts,
    metadata_column_fingerprint,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import cell_selection_indices
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    file_sha256,
    warn_unmatched_barcodes,
)

if TYPE_CHECKING:
    from ...readers.clone_tables import CloneLayout
    from ..mapping_datastore import MappingDatastore as _CloneOperationsBase
else:
    _CloneOperationsBase = object

#: The values of ``export_clones``'s ``format``.
EXPORT_FORMATS = ("table", "matrix")
_PROBABILITY = "clone_probability"
# Bytes held for each selected cell while clones are gathered: its clone code,
# whether it was given several clones, and its probability.
_ASSIGNER_BYTES_PER_CELL = 24
# Bytes held for each distinct clone label beside its text: its entries in
# the table of labels.
_BYTES_PER_LABEL = 200
# Bytes of one parsed row of a clone file at its peak, beside its fields: the
# barcode, clone label and probability it yields, the cell ID it names and
# the arrays that join and assign it.
_BYTES_PER_ROW = 400
# Bytes of each field of a parsed row: its text and its slot in the chunk.
_BYTES_PER_FIELD = 120
# Bytes of each character of a row of a clone file: its fields as Python
# strings, and the copies of the barcode that a prefix, suffix or pattern makes.
_BYTES_PER_ROW_CHAR = 8
# Bytes of a Python string beside its text, with its slot in an object array.
_STRING_BYTES = 64
# Bytes of each listed cell of a clone matrix: its selected cell.
_BYTES_PER_LISTED_CELL = 8
# The rows of a chunk lie in [_MIN_CHUNK_ROWS, _MAX_CHUNK_ROWS].
_MIN_CHUNK_ROWS = 1_000
_MAX_CHUNK_ROWS = 1_000_000
# Bytes per cell of the tables of loaded clones beside its ID: the clone
# code, size and probability, and the pandas index.
_TABLE_BYTES_PER_CELL = 200
# Bytes per cell of an export beside its ID: its label and its code.
_EXPORT_BYTES_PER_CELL = 200


def _text_width(array: Any) -> int:
    """The characters of each value of stored fixed-width text, 0 for other arrays."""
    dtype = np.dtype(array.dtype)
    if dtype.kind == "U":
        return int(dtype.itemsize) // 4
    if dtype.kind == "S":
        return int(dtype.itemsize)
    return 0


def _text_read_bytes(array: Any, n_rows: int) -> int:
    """Bytes of reading rows of a stored text array as Python strings.

    The store keeps text at the width of its longest value and decodes whole
    chunks, so one long value widens every row that is read.
    """
    fixed, per_row = array_row_selection_parts(array)
    return int(fixed) + n_rows * (int(per_row) + _STRING_BYTES + 4 * _text_width(array))


def _file_namespace(file: BarcodeFile) -> str:
    """The text before a file's clone labels: its prefix, or its name and ``_``."""
    if file.barcode_prefix is not None:
        return file.barcode_prefix
    stem = Path(os.fspath(file.path)).name.split(".", 1)[0]
    if not stem:
        raise ValueError(
            f"The clone file {os.fspath(file.path)!r} has no name before its "
            "extensions to namespace its clones; give it a barcode_prefix"
        )
    return f"{stem}_"


def _namespaces(files: Sequence[BarcodeFile]) -> list[str | None]:
    """Namespace each file's clone labels when several files are imported."""
    if len(files) == 1:
        return [None]
    namespaces = [_file_namespace(file) for file in files]
    seen: dict[str, BarcodeFile] = {}
    for file, namespace in zip(files, namespaces, strict=True):
        other = seen.setdefault(namespace, file)
        if other is not file:
            # A file's name namespaces its labels only without a prefix, and a
            # prefix also changes the cells that its barcodes name.
            remedy = (
                "give each file its own barcode_prefix"
                if file.barcode_prefix is not None
                else "rename one of the files so their names differ before the "
                "first '.'"
            )
            raise ValueError(
                f"The clone files {os.fspath(other.path)!r} and "
                f"{os.fspath(file.path)!r} share the namespace {namespace!r} of "
                f"their clone labels, so equal clone IDs would merge; {remedy}"
            )
    return list(namespaces)


def _namespaced(labels: np.ndarray, namespace: str | None) -> np.ndarray:
    if namespace is None:
        return labels
    named = np.asarray(labels, dtype=object).copy()
    present = np.flatnonzero(pd.notna(named))
    named[present] = [namespace + label for label in named[present].tolist()]
    return named


def _files_text(n_files: int) -> str:
    return "the clone file" if n_files == 1 else "the clone files"


class _CloneOperationsMixin(_CloneOperationsBase):
    def _clone_selection_indices(self, cell_selection: ArtifactRef) -> np.ndarray:
        cell_index = cell_selection_indices(self.zw, cell_selection)
        if not len(cell_index):
            raise ValueError("cell_selection selects no cell")
        return cell_index

    def _clone_chunk_rows(self, held: int, row_bytes: int) -> int:
        """Rows of a clone file to parse at once beside ``held`` bytes."""
        room = (self.memoryBytes - held) // (2 * row_bytes)
        if room < _MIN_CHUNK_ROWS:
            raise MemoryError(
                f"Importing clones holds about {held} bytes before it reads a row, "
                f"which leaves too little of the memory budget of "
                f"{self.memoryBytes} bytes to read the files. Raise mem_budget, or "
                "pass a smaller cell_selection."
            )
        return int(min(room, _MAX_CHUNK_ROWS))

    def _check_clone_import_budget(
        self, held: int, n_labels: int, label_bytes: int, chunk_bytes: int
    ) -> None:
        peak = held + _BYTES_PER_LABEL * n_labels + label_bytes + chunk_bytes
        if peak > self.memoryBytes:
            raise MemoryError(
                f"Gathering {n_labels} clone labels holds about {peak} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget."
            )

    def _check_clone_write_budget(
        self, held: int, clone_ids: np.ndarray, n_cells: int
    ) -> None:
        """Refuse clones whose labels would not fit the memory budget as stored.

        The store saves text at the width of the longest value, so each cell's
        label takes four bytes per character of the longest as it is written.
        """
        width = max((len(str(label)) for label in clone_ids.tolist()), default=0)
        peak = held + n_cells * (8 + 4 * width)
        if peak > self.memoryBytes:
            raise MemoryError(
                f"Saving the clones of {n_cells} cells holds about {peak} bytes, "
                f"more than the memory budget of {self.memoryBytes} bytes, since the "
                "store saves each cell's clone label at the width of the longest, "
                f"{width} characters. Raise mem_budget."
            )

    @validation_scoped
    def _clones_import_clones(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        cell_selection: ArtifactRef,
        barcode_prefix: str | Sequence[str | None] | None = None,
        barcode_suffix: str | Sequence[str | None] | None = None,
        barcode_pattern: str | Sequence[str | None] | None = None,
        format: str | None = None,
        columns: Mapping[str, str | Sequence[str]] | None = None,
        multiple: str = "error",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import clones that a lineage tracing tool or a study assigned to cells.

        A clone is the set of cells that descend from one ancestor cell. Tools
        that read lineage barcodes, CNV profiles or mitochondrial variants
        write which clone each cell belongs to, and the import joins each of
        those cells to a selected cell by barcode and saves the clones as a
        ``clones`` artifact, which every clone source writes. Each file's
        format is recognised from its contents: CoSpar's clone table, TREX's
        ``clones.txt`` of every release, CellTag-multi's and CellTagR's clone
        tables, STICR's cell metadata, Cassiopeia's allele table, Numbat's
        ``clone_post_{i}.tsv``, a Matrix Market clone matrix with its cell
        list, such as LARRY's on GEO, and AnnData files with CoSpar's
        ``obsm['X_clone']`` or MiTo's ``obs['MiTo_clone']``. ``columns``
        reads any other table, or an AnnData ``obs`` column, by the columns
        it names; ``format`` names a format that the contents leave open.
        The clones API reference lists the formats and their columns.

        A cell's ID is its file's ``barcode_prefix``, the barcode and
        ``barcode_suffix``, matched exactly; ``barcode_pattern`` keeps the
        part of each barcode that its ``barcode`` group matches first.
        Barcodes that match no selected cell are counted, saved in the
        artifact's ``n_file_barcodes``, ``n_unmatched_barcodes`` and
        ``unmatched_barcode_examples`` attributes, and named in a warning,
        which a reused import gives again. A file is identified by the SHA-256
        of its bytes, and of a matrix's cell and clone lists, so a moved file
        reuses the saved clones, an edited one gives new clones, and the
        order of the files does not matter.

        Clone IDs are unique only within one tool's run, so with several
        files each file's labels start with its ``barcode_prefix``, or with
        its name without extensions and ``_``; two files with the same
        namespace raise ``ValueError``. A cell listed in two different clones
        raises ``ValueError``, and with ``multiple="drop"`` it is left
        without a clone and counted in the ``n_cells_in_several_clones``
        attribute. A selected cell that no file lists has no clone.

        The artifact's canonical ``values`` hold each cell's clone label,
        empty for a cell without one, which groupings treat as missing.
        ``codes`` numbers the clones by size, largest first, then by label,
        -1 for none, and ``clone_size`` holds the cells of each cell's clone,
        0 for none. A Numbat file, or a probability column that ``columns``
        names, adds ``clone_probability``, NaN where no probability is given.
        ``clone_ids`` and ``clone_sizes`` hold one row per clone.

        Args:
            paths: One clone file, or several, plain or gzip-compressed.
            cell_selection: The cells that the clones label.
            barcode_prefix: Text before each barcode in the cell IDs, such as
                ``"sample1__"`` in a merged store: one for every file, or one
                per file. None adds nothing.
            barcode_suffix: Text after each barcode in the cell IDs, such as
                ``"-1"``: one for every file, or one per file.
            barcode_pattern: A regular expression whose named group
                ``barcode`` is the part of each barcode that names the cell,
                found as ``re.search`` finds it, such as
                ``r"_(?P<barcode>[ACGT]{16}-1)$"``; a barcode it does not match
                names no cell. One for every file, or one per file.
            format: One of ``"cospar"``, ``"trex"``, ``"celltag_multi"``,
                ``"celltagr"``, ``"sticr"``, ``"cassiopeia"``, ``"numbat"``,
                ``"table"``, ``"matrix"``, ``"cospar_h5ad"``, ``"mito"`` and
                ``"h5ad"``, or None to recognise each file's format.
            columns: The columns to read: ``"cell"`` (a column, or a list of
                columns whose values are joined by ``"__"``, such as
                ``["Library", "Cell barcode"]`` of LARRY's metadata),
                ``"clone"`` and ``"probability"``. A table read by row names,
                such as a wide table with one clone column per labelling
                round, needs only ``"clone"``, and a clone matrix takes only
                ``"cell"``, the columns of its cell list.
            multiple: ``"error"`` or ``"drop"``, for a cell listed in two
                different clones.
            invalidate_cache: Import again instead of reusing saved clones.

        Returns:
            A complete datastore-scoped ``clones`` artifact. Read it with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_clones`
            or plot it with ``ds.plots.clone_abundance``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a file's format is unknown or malformed, a column
                is missing, an option lists a value for another number of
                files, a file is listed twice with the same prefix and suffix,
                two files share a namespace, a cell is listed in two clones
                with ``multiple="error"``, or the selection holds no cell.
            FileNotFoundError: If a file, or a clone matrix's cell list, is
                missing.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If the import does not fit the memory budget.
        """
        from ...readers.clone_tables import clone_layout

        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if multiple not in MULTIPLE_POLICIES:
            raise ValueError(
                f"multiple must be one of {', '.join(MULTIPLE_POLICIES)}, not "
                f"{multiple!r}"
            )
        policy: MultiplePolicy = "drop" if multiple == "drop" else "error"
        bool_argument(invalidate_cache, "invalidate_cache")
        if format is not None and not isinstance(format, str):
            raise TypeError("format must be a format name or None")
        files = barcode_files(
            paths,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            what="clone file",
        )
        layouts = [
            clone_layout(file.path, format=format, columns=columns) for file in files
        ]
        namespaces = _namespaces(files)
        sources = tuple(
            {
                **layout.record(),
                **file.record(),
                "related_sha256": [file_sha256(path) for path in layout.related_paths],
                "namespace": namespace,
            }
            for file, layout, namespace in zip(files, layouts, namespaces, strict=True)
        )
        cell_index = self._clone_selection_indices(cell_selection)
        n_cells = len(cell_index)
        arguments = CloneImportArguments(
            cell_selection=cell_selection,
            sources=sources,
            multiple=policy,
            invalidate_cache=invalidate_cache,
        )
        with_probability = any(layout.probability is not None for layout in layouts)
        planned = self._plan_clones(
            arguments,
            cell_selection,
            n_cells,
            with_probability=with_probability,
            required_attributes=(
                "n_file_barcodes",
                "n_unmatched_barcodes",
                "unmatched_barcode_examples",
                "n_clones",
                "n_cells_in_several_clones",
            ),
        )
        files_text = _files_text(len(files))
        if planned.reused:
            # Saved clones warn as they did when they were imported.
            warn_unmatched_barcodes(
                artifact_group(self.zw, planned.ref).attrs, files_text
            )
            return planned.ref
        ids = self.cells._get_array("ids")
        reading = (
            LOOKUP_BYTES_PER_CELL + _ASSIGNER_BYTES_PER_CELL
        ) * n_cells + _text_read_bytes(ids, n_cells)
        if reading > self.memoryBytes:
            raise MemoryError(
                f"Reading the IDs of {n_cells} selected cells holds about {reading} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget, or pass a smaller cell_selection."
            )
        self._require_writable(arguments.operation)
        # Only the selected cells' IDs are read, a sample's in a merged store.
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )
        join = CellBarcodeJoin(
            cell_ids,
            memory_bytes=self.memoryBytes,
            held_bytes=_ASSIGNER_BYTES_PER_CELL * n_cells,
        )
        held = join.held_bytes
        # Rows are parsed in chunks sized by the widest file's fields and its
        # longest sampled rows, whose barcodes are about as long as cell IDs.
        row_bytes = (
            _BYTES_PER_ROW
            + _BYTES_PER_FIELD * max(layout.n_fields for layout in layouts)
            + _BYTES_PER_ROW_CHAR
            * max(_text_width(ids), *(layout.row_chars for layout in layouts))
        )
        chunk_rows = self._clone_chunk_rows(held, row_bytes)
        assigner = CloneAssigner(cell_ids, multiple=policy)
        for position, (file, layout, namespace) in enumerate(
            zip(files, layouts, namespaces, strict=True)
        ):
            self._assign_clone_file(
                assigner,
                join,
                file,
                layout,
                namespace,
                source=position,
                held=held,
                chunk_rows=chunk_rows,
                chunk_bytes=chunk_rows * row_bytes,
            )
        assignment = assigner.finish()
        self._check_clone_write_budget(
            held + _BYTES_PER_LABEL * assigner.n_labels + assigner.label_bytes,
            assignment.clone_ids,
            n_cells,
        )
        attributes = {
            **join.attributes(),
            "n_clones": len(assignment.clone_ids),
            "n_cells_in_several_clones": assignment.n_cells_in_several_clones,
        }
        self._write_clones(planned, assignment, attributes)
        warn_unmatched_barcodes(attributes, files_text)
        self._log_clones(assignment, n_cells)
        return planned.ref

    def _assign_clone_file(
        self,
        assigner: CloneAssigner,
        join: CellBarcodeJoin,
        file: BarcodeFile,
        layout: "CloneLayout",
        namespace: str | None,
        *,
        source: int,
        held: int,
        chunk_rows: int,
        chunk_bytes: int,
    ) -> None:
        """Assign the cells of one clone file to its clones.

        ``held`` bytes are held beside the file's chunks of ``chunk_rows``
        rows, which hold up to ``chunk_bytes`` bytes.
        """
        from ...readers.clone_tables import (
            iter_clone_rows,
            matrix_cells,
            matrix_clone_ids,
            matrix_entries,
        )

        if not layout.is_matrix:
            for rows in iter_clone_rows(layout, chunk_rows):
                assigner.add(
                    join.cells(file, rows.barcodes),
                    _namespaced(rows.clones, namespace),
                    probabilities=rows.probabilities,
                    source=source,
                )
                self._check_clone_import_budget(
                    held, assigner.n_labels, assigner.label_bytes, chunk_bytes
                )
            return
        # A matrix lists its cells first; each entry names a listed cell.
        listed: list[np.ndarray] = []
        n_listed = 0
        for barcodes in matrix_cells(layout, chunk_rows):
            listed.append(join.cells(file, barcodes))
            n_listed += len(barcodes)
            self._check_clone_import_budget(
                held + _BYTES_PER_LISTED_CELL * n_listed,
                assigner.n_labels,
                assigner.label_bytes,
                chunk_bytes,
            )
        cells = np.concatenate(listed) if listed else np.empty(0, dtype=np.int64)
        labels = _namespaced(matrix_clone_ids(layout), namespace)
        # The matrix's labels are held beside those of earlier files.
        self._check_clone_import_budget(
            held + _BYTES_PER_LISTED_CELL * n_listed,
            assigner.n_labels + len(labels),
            assigner.label_bytes + sum(map(sys.getsizeof, labels)),
            chunk_bytes,
        )
        for entries in matrix_entries(layout, chunk_rows):
            assigner.add(cells[entries.cells], labels[entries.clones], source=source)

    def _plan_clones(
        self,
        arguments: CloneImportArguments | CloneColumnArguments,
        cell_selection: ArtifactRef,
        n_cells: int,
        *,
        with_probability: bool,
        required_attributes: tuple[str, ...],
    ) -> PlannedArtifact:
        record = arguments.to_record()
        arrays: dict[str, tuple[tuple[int | None, ...], str | None]] = {
            "values": ((n_cells,), None),
            "codes": ((n_cells,), "i"),
            "clone_size": ((n_cells,), "i"),
        }
        if with_probability:
            arrays[_PROBABILITY] = ((n_cells,), "f")

        def has_clones(_ref: ArtifactRef, group: zarr.Group) -> bool:
            ids = group.get("clone_ids")
            sizes = group.get("clone_sizes")
            return (
                isinstance(ids, zarr.Array)
                and isinstance(sizes, zarr.Array)
                and ids.shape == sizes.shape
            )

        return plan_cell_data_artifact(
            self.zw,
            scope="datastore",
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays=arrays,
            invalidate_cache=arguments.invalidate_cache,
            required_attributes=required_attributes,
            reuse_validator=has_clones,
        )

    def _write_clones(
        self,
        planned: PlannedArtifact,
        assignment: CloneAssignment,
        attributes: Mapping[str, Any],
    ) -> None:
        arrays: dict[str, np.ndarray] = {
            "values": assignment.values,
            "codes": assignment.codes.astype(np.int32),
            "clone_size": assignment.clone_size,
            "clone_ids": assignment.clone_ids,
            "clone_sizes": assignment.clone_sizes.astype(np.int64),
        }
        if assignment.probabilities is not None:
            arrays[_PROBABILITY] = assignment.probabilities.astype(np.float64)
        write_cell_data_artifact(self.zw, planned, arrays, attributes=dict(attributes))

    def _log_clones(self, assignment: CloneAssignment, n_cells: int) -> None:
        n_assigned = int(np.count_nonzero(assignment.codes >= 0))
        largest = int(assignment.clone_sizes[0]) if len(assignment.clone_sizes) else 0
        logger.info(
            f"Saved {len(assignment.clone_ids)} clones of {n_assigned} of {n_cells} "
            f"cells; the largest holds {largest} cells"
        )

    @validation_scoped
    def _clones_from_column(
        self,
        column: str,
        *,
        cell_selection: ArtifactRef,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Save the clones that a cell metadata column assigns.

        A study's metadata often holds each cell's clone, such as a column
        imported with an AnnData file. The selected cells' labels in
        ``column`` become a ``clones`` artifact, as
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_clones`
        writes: a missing or blank label means no clone. Labels are text;
        integers read as their text, and floating-point whole numbers, which
        pandas writes for integer labels beside missing ones, as integers.
        The artifact records the fingerprint of the column, so editing it
        gives new clones. The labels are read in blocks within the memory
        budget.

        Args:
            column: A cell metadata column.
            cell_selection: The cells that the clones label.
            invalidate_cache: Save again instead of reusing saved clones.

        Returns:
            A complete datastore-scoped ``clones`` artifact.

        Raises:
            TypeError: If an argument has the wrong type, or a label is a
                floating-point number that is not whole.
            ValueError: If ``column`` is not a cell metadata column, or the
                selection holds no cell.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If the labels do not fit the memory budget.
        """
        if not isinstance(column, str):
            raise TypeError("column must be the name of a cell metadata column")
        if column not in self.cells.columns:
            raise ValueError(f"{column!r} is not a cell metadata column")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        bool_argument(invalidate_cache, "invalidate_cache")
        cell_index = self._clone_selection_indices(cell_selection)
        n_cells = len(cell_index)
        array = self.cells._get_array(column)
        fixed, per_row = array_row_selection_parts(array)
        # Blocks of labels are read at their stored width, decoding whole
        # chunks, and become Python strings beside each cell's code.
        held = _ASSIGNER_BYTES_PER_CELL * n_cells + int(fixed)
        row_bytes = int(per_row) + _STRING_BYTES + 4 * _text_width(array)
        first_block = held + row_bytes * min(n_cells, _MIN_CHUNK_ROWS)
        if first_block > self.memoryBytes:
            raise MemoryError(
                f"Reading the clone labels of {n_cells} cells holds about "
                f"{first_block} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        block_rows = int(
            min(n_cells, _MAX_CHUNK_ROWS, (self.memoryBytes - held) // row_bytes)
        )
        arguments = CloneColumnArguments(
            cell_selection=cell_selection,
            labels_fingerprint=metadata_column_fingerprint(self.cells, column),
            column=column,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_clones(
            arguments,
            cell_selection,
            n_cells,
            with_probability=False,
            required_attributes=("n_clones",),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        # Each cell has one label, so no cell is given two clones.
        assigner = CloneAssigner(cell_index, multiple="error")
        for start in range(0, n_cells, block_rows):
            rows = cell_index[start : start + block_rows]
            assigner.add(
                np.arange(start, start + len(rows)),
                clone_label_texts(
                    np.asarray(read_metadata_rows_chunkwise(self.cells, column, rows)),
                    read_metadata_missing_rows_chunkwise(self.cells, column, rows),
                ),
            )
            self._check_clone_import_budget(
                held, assigner.n_labels, assigner.label_bytes, block_rows * row_bytes
            )
        assignment = assigner.finish()
        self._check_clone_write_budget(
            _ASSIGNER_BYTES_PER_CELL * n_cells
            + _BYTES_PER_LABEL * assigner.n_labels
            + assigner.label_bytes,
            assignment.clone_ids,
            n_cells,
        )
        self._write_clones(planned, assignment, {"n_clones": len(assignment.clone_ids)})
        self._log_clones(assignment, n_cells)
        return planned.ref

    def _clone_group(self, clones: ArtifactRef) -> zarr.Group:
        """Return a complete ``clones`` artifact's group."""
        if not isinstance(clones, ArtifactRef):
            raise TypeError("clones must be an ArtifactRef")
        if clones.kind != CloneImportArguments.artifact_kind:
            raise ValueError(f"clones must be a clones artifact, not {clones.kind!r}")
        require_complete_artifact(self.zw, clones)
        return artifact_group(self.zw, clones)

    def _clone_text_bytes(self, group: zarr.Group, n_cells: int) -> int:
        """Bytes of reading saved clones' labels and their cells' IDs as text."""
        clone_ids = as_zarr_array(group["clone_ids"], name="clone_ids")
        return _text_read_bytes(clone_ids, int(clone_ids.shape[0])) + (
            _text_read_bytes(self.cells._get_array("ids"), n_cells)
        )

    def _clone_cell_ids(self, clones: ArtifactRef) -> np.ndarray:
        """The IDs of a clones artifact's cells, in the order of its arrays."""
        selection = inspect_artifact(self.zw, clones).input_ref("cell_selection")
        cell_index = cell_selection_indices(self.zw, selection)
        return np.asarray(read_metadata_rows_chunkwise(self.cells, "ids", cell_index))

    @validation_scoped
    def _clones_load_clones(self, clones: ArtifactRef) -> Clones:
        """Read saved clones as a table of cells and a table of clones.

        Args:
            clones: A ``clones`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_clones`.

        Returns:
            The clones, with one row per cell and one row per clone, largest
            first.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a complete ``clones`` artifact.
            MemoryError: If the tables, about 200 bytes per cell beside the
                text of its ID and of the clone labels, do not fit the memory
                budget.
        """
        group = self._clone_group(clones)
        n_cells = int(as_zarr_array(group["codes"], name="codes").shape[0])
        resident = (
            self._clone_text_bytes(group, n_cells) + _TABLE_BYTES_PER_CELL * n_cells
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Loading the clones of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        extra = {}
        if _PROBABILITY in group:
            extra[_PROBABILITY] = _read(group, _PROBABILITY).astype(np.float64)
        cells, table = clone_frames(
            self._clone_cell_ids(clones),
            codes=_read(group, "codes"),
            clone_size=_read(group, "clone_size"),
            clone_ids=_read(group, "clone_ids").astype(object),
            clone_sizes=_read(group, "clone_sizes"),
            extra=extra,
        )
        return Clones(
            cells=cells,
            clones=table,
            source=str(inspect_artifact(self.zw, clones).operation),
        )

    @validation_scoped
    def _clones_export_clones(
        self,
        clones: ArtifactRef,
        path: str | os.PathLike[str],
        *,
        format: str = "table",
    ) -> int:
        """Write saved clones as a file that lineage tracing tools read.

        ``"table"`` writes CoSpar's clone table, a CSV of ``Cell_ID`` and
        ``Clone_ID`` with one row per cell with a clone, which CoSpar and
        ``import_clones`` read. ``"matrix"`` writes a cells by clones Matrix
        Market file of the cells with a clone, holding a 1 where a cell
        belongs to a clone, with ``cells.tsv`` and ``clones.tsv`` beside it
        that name its rows and columns, replacing files of those names;
        ``import_clones`` reads the matrix with its lists. Cell IDs are this
        store's, and a path ending in ``.gz`` is gzip-compressed. Clone
        probabilities are not written.

        Args:
            clones: A ``clones`` artifact.
            path: The file to write.
            format: ``"table"`` or ``"matrix"``.

        Returns:
            The number of cells written.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a complete ``clones`` artifact,
                ``format`` is unknown, a ``"table"`` would hold a clone label
                that reads as no clone, such as ``"NA"``, or a ``"matrix"``
                would list a label or cell ID that holds a line break.
            MemoryError: If the export, about 200 bytes per cell beside the
                text of its ID and of the clone labels, does not fit the
                memory budget.
        """
        from ...readers.clone_tables import MISSING_TEXT
        from ...writers.clone_tables import write_clone_matrix, write_clone_table

        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path")
        if format not in EXPORT_FORMATS:
            raise ValueError(
                f"format must be one of {', '.join(EXPORT_FORMATS)}, not {format!r}"
            )
        group = self._clone_group(clones)
        codes = _read(group, "codes").astype(np.int64)
        resident = self._clone_text_bytes(group, len(codes)) + (
            _EXPORT_BYTES_PER_CELL * len(codes)
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Exporting the clones of {len(codes)} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        clone_ids = _read(group, "clone_ids").astype(object)
        if format == "table":
            # A clone table reads these labels back as cells without a clone.
            lost = [label for label in clone_ids.tolist() if label in MISSING_TEXT]
            if lost:
                raise ValueError(
                    "The clone labels "
                    + ", ".join(repr(label) for label in lost[:5])
                    + " read as no clone in a clone table; export these clones "
                    "with format='matrix', whose clone list keeps them"
                )
        assigned = codes >= 0
        cell_ids = np.asarray(self._clone_cell_ids(clones), dtype=object)[assigned]
        if format == "table":
            return write_clone_table(path, cell_ids, clone_ids[codes[assigned]])
        return write_clone_matrix(path, cell_ids, codes[assigned], clone_ids)


def _read(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[:])
