import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
from ...clones.diversity import (
    INDEX_COLUMNS,
    LARGEST_SIZE,
    ORDERS,
    ClonalDiversity,
    CloneSizes,
    Rarefaction,
    checked_curve_arguments,
    diversity_tables,
    rarefaction_curves,
)
from ...clones.expansion import (
    DEFAULT_BREAKPOINTS,
    DEFAULT_RANK_SPLITS,
    ClonalExpansion,
    checked_bounds,
    expansion_categories,
    expansion_tables,
    frequency_labels,
    group_clone_sizes,
)
from ...clones.fate import (
    COUPLING_METHODS,
    FATE_CELL_COLUMNS,
    CloneCoupling,
    CloneFateBias,
    clone_coupling,
    coupling_plan,
    fate_bias,
    fate_bias_bytes,
    fate_bias_tables,
)
from ...clones.groups import group_frame, pair_counts
from ...clones.overlap import (
    CloneOverlap,
    CloneTracking,
    overlap_extent,
    overlap_pairs,
    top_clones,
    tracking_cells,
    tracking_reach,
)
from ...metadata.arguments import (
    ClonalExpansionArguments,
    CloneColumnArguments,
    CloneFateBiasArguments,
    CloneImportArguments,
    ClonotypeArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.rows import (
    array_row_selection_parts,
    metadata_column_fingerprint,
    previous_read_bytes,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
    text_read_bytes,
    text_row_selection_parts,
    text_width,
)
from ...metadata.selection import category_codes, cell_selection_indices
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.selections import validate_cell_selection
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.arrays import sorted_unique
from ...utils.logging import logger
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    file_sha256,
    warn_unmatched_barcodes,
)
from .cell_groups import cell_groups, group_columns, selection_chunk_bytes

if TYPE_CHECKING:
    from ...readers.clone_tables import CloneLayout
    from .repertoire import _RepertoireOperationsMixin as _CloneOperationsBase
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
# Bytes of each listed cell of a clone matrix: its selected cell.
_BYTES_PER_LISTED_CELL = 8
# The rows of a chunk lie in [_MIN_CHUNK_ROWS, _MAX_CHUNK_ROWS].
_MIN_CHUNK_ROWS = 1_000
_MAX_CHUNK_ROWS = 1_000_000
# Bytes per cell of the tables of loaded clones beside its ID: the clone
# code, size and probability, the labels kept apart, and the pandas index;
# with the ID, about 220 bytes per cell, and 230 with the receptor types of
# clonotypes, as traced on 200,000 cells.
_TABLE_BYTES_PER_CELL = 250
# Bytes per cell of an export beside its ID: its label and its code.
_EXPORT_BYTES_PER_CELL = 200
# The kinds of clone artifact that every clone summary reads.
_CLONE_SOURCE_KINDS = (
    CloneImportArguments.artifact_kind,
    ClonotypeArguments.artifact_kind,
)
# The key column of each cell's receptor type, which clonotypes keep apart.
_RECEPTOR_TYPE_COLUMN = "receptor_type"
# The array of each kind's clone labels, and the key columns whose labels its
# clones keep apart.
_CLONE_ID_ARRAYS = {
    CloneImportArguments.artifact_kind: "clone_ids",
    ClonotypeArguments.artifact_kind: "clonotype_ids",
}
_KEPT_APART_COLUMNS: dict[str, tuple[str, ...]] = {
    CloneImportArguments.artifact_kind: (),
    ClonotypeArguments.artifact_kind: (_RECEPTOR_TYPE_COLUMN,),
}
# Reading the receptor types of clonotypes holds each cell's stored text and,
# while it is coded, a Python string and a code for each cell, 92 bytes beside
# the text as traced on 200,000 cells. Each cell then keeps a label in each
# column that the clones keep apart, a pointer to one of a few shared labels.
_RECEPTOR_TYPE_CODING_BYTES = 100
_KEPT_LABEL_BYTES = 8
# Reading a clone artifact holds, for each of its cells, the stored code and
# this many bytes for its int64 copy.
_CLONE_CODE_BYTES = 8
# Reading each clone's label holds its stored text up to three times, decoded,
# compressed and copied out of its chunk, and later as a Python string, with
# this many bytes for the string's header and pointer.
_CLONE_ID_BYTES = 100
_CLONE_ID_COPIES = 3
# Matching a clone artifact's cells to other measured cells holds, for each
# clone cell, its code, its row and its place in a hash table, 45 bytes as
# traced on 200,000 cells, and for each measured cell its row, position and
# code, and its label in each column that the clones keep apart.
_MATCH_BYTES_PER_CLONE_CELL = 56
_MATCH_BYTES_PER_CELL = 32
_MATCH_BYTES_PER_LABEL = 8
# Measuring clonal expansion holds, at its peak, about 230 bytes per cell: the
# clone and group codes, the clone sizes and frequencies, and the labels as
# written, as traced on 1,000,000 cells in eight samples, and 281 bytes per
# cell without a grouping, as traced on 200,000 cells.
_EXPANSION_BYTES_PER_CELL = 300
# The tables of loaded clonal expansion hold about 310 bytes per cell at their
# peak, with the cell IDs and labels as text, as traced on the same cells,
# besides the clone labels that they read.
_EXPANSION_TABLE_BYTES_PER_CELL = 500
# Each group of a summary holds its clone sizes and its rows of the result's
# tables, 1,950 bytes for diversity as traced on 180,057 groups.
_SUMMARY_BYTES_PER_GROUP = 2_000
# Rarefying a group holds, for each pair of a whole sample size below its
# cells and a distinct clone size, the chance that the sample misses such a
# clone and three temporary arrays of the same shape.
_RAREFACTION_BYTES_PER_ENTRY = 40
# Each row of the rarefaction curves: its order, size, method, Hill number,
# coverage and group.
_CURVE_BYTES_PER_ROW = 100
# The columns of each summary's tables besides the groups' labels, which a
# key column of the groups may not share; "group" names each group, and plots
# of expansion add a "group" column.
_EXPANSION_COLUMNS = frozenset(
    {"group", "clone", "category", "group_clone_size", "group_clone_frequency"}
    | {"cells", "clones", "fraction", "window"}
)
_DIVERSITY_COLUMNS = frozenset(
    {"group", *INDEX_COLUMNS, "order", "observed", "estimated", "at_size"}
    | {"coverage_at_size", "method_at_size", "at_coverage", "size_at_coverage"}
    | {"method_at_coverage"}
)
_RAREFACTION_COLUMNS = frozenset(
    {"group", "order", "size", "method", "diversity", "coverage"}
)
_OVERLAP_COLUMNS = frozenset({"group", "cells", "clones"})
_TRACK_COLUMNS = frozenset({"group", "clone", "cells", "fraction"})
# Comparing groups holds, at its peak, about 220 bytes per pair of groups of
# one stratum, the pairs' indices and their table, and 16 bytes per product of
# two groups of the largest stratum, as measured on 1,000 and 2,000 groups;
# 2,000 groups of a million cells, 2 million pairs, traced 374 MiB beyond
# grouping the cells.
_OVERLAP_BYTES_PER_PAIR = 250
_OVERLAP_BYTES_PER_PRODUCT = 24
# Each count of a clone's cells in a group holds its group, clone code and
# cells, and its entries in the matrices that compare or follow the groups.
_BYTES_PER_CLONE_COUNT = 40
# Following clones holds, for each row of the table, about 82 bytes and 8
# more for each key column, as traced on 3 million rows with 1 to 12 key
# columns, and a byte per chosen clone and group.
_TRACK_BYTES_PER_ROW = 100
_TRACK_BYTES_PER_KEY_COLUMN = 16
# The columns of the tables of coupling and fate bias besides the groups'
# labels, which a key column of the groups may not share.
_COUPLING_COLUMNS = frozenset({"group", "cells", "clones"})
_FATE_BIAS_COLUMNS = frozenset(
    {"group", "clone", "fate", "early_cells", "late_cells", "cells"}
    | {"fraction", "p_value", "q_value"}
)
# Checking that no cell is both early and late holds, for each cell of the
# two selections, its row and its place in their sorted union.
_SELECTION_BYTES_PER_CELL = 40
# Each pair of a fate and a clone holds its fate, clone code and cells.
_BYTES_PER_FATE_PAIR = 24
# Writing the fate bias holds, for each early cell, its dominant fate as
# text of four bytes per character, and its other arrays.
_BIAS_WRITE_BYTES_PER_CELL = 40
_BIAS_WRITE_BYTES_PER_CHARACTER = 4
# The tables of loaded fate bias hold, for each early cell, its ID, clone,
# fate and late cells and the pandas index, for each of its fractions the
# value read and its column in the table, and for each row of the table of
# clones and fates its labels, counts and tests: 92 to 113 bytes at their
# peak as traced on 1 to 3 million rows with one to three key columns.
_BIAS_TABLE_BYTES_PER_CELL = 200
_BIAS_TABLE_BYTES_PER_FRACTION = 24
_BIAS_TABLE_BYTES_PER_ROW = 160
_BIAS_TABLE_BYTES_PER_KEY_COLUMN = 16


@dataclass(frozen=True, slots=True)
class _CloneInput:
    """A clone artifact as the clone summaries read it.

    Attributes:
        ref: The clone artifact.
        cell_selection: The selection of its cells, which its arrays follow.
        codes: Each cell's clone, as int64, -1 for none.
        clone_ids: Each clone's label, by code, or None when not read.
        kept_apart: The key columns whose labels the clones keep apart, each
            with every cell's label: clones keep none, and clonotypes keep the
            receptor type.
    """

    ref: ArtifactRef
    cell_selection: ArtifactRef
    codes: np.ndarray
    clone_ids: np.ndarray | None
    kept_apart: tuple[tuple[str, np.ndarray], ...]


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
        ) * n_cells + text_read_bytes(ids, n_cells)
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
            * max(text_width(ids), *(layout.row_chars for layout in layouts))
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
        # Blocks of labels are read at their stored width, decoding whole
        # chunks, and become Python strings beside each cell's code.
        fixed, row_bytes = text_row_selection_parts(self.cells._get_array(column))
        held = _ASSIGNER_BYTES_PER_CELL * n_cells + fixed
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
        """Return a complete clone artifact's group, of a kind the summaries read."""
        if not isinstance(clones, ArtifactRef):
            raise TypeError("clones must be an ArtifactRef")
        if clones.kind not in _CLONE_SOURCE_KINDS:
            raise ValueError(
                f"clones must be a {' or '.join(_CLONE_SOURCE_KINDS)} artifact, "
                f"not {clones.kind!r}"
            )
        require_complete_artifact(self.zw, clones)
        return artifact_group(self.zw, clones)

    def _clone_text_bytes(self, clones: ArtifactRef, n_cells: int) -> int:
        """Bytes of reading saved clones' labels and their cells' IDs as text."""
        name = _CLONE_ID_ARRAYS[clones.kind]
        labels = as_zarr_array(artifact_group(self.zw, clones)[name], name=name)
        return text_read_bytes(labels, int(labels.shape[0])) + (
            text_read_bytes(self.cells._get_array("ids"), n_cells)
        )

    def _clone_cell_ids(self, clones: ArtifactRef) -> np.ndarray:
        """The IDs of a clone artifact's cells, in the order of its arrays."""
        selection = inspect_artifact(self.zw, clones).input_ref("cell_selection")
        cell_index = cell_selection_indices(self.zw, selection)
        return np.asarray(read_metadata_rows_chunkwise(self.cells, "ids", cell_index))

    @validation_scoped
    def _clones_load_clones(self, clones: ArtifactRef) -> Clones:
        """Read saved clones as a table of cells and a table of clones.

        Clonotypes read as clones too, labelled ``clonotype_1`` and so on,
        and their table of cells also holds each cell's ``receptor_type``,
        whose labels every clone summary measures apart.

        Args:
            clones: A clone artifact: ``clones``, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_clones`,
                or ``clonotypes``, from
                :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.clonotypes`.

        Returns:
            The clones, with one row per cell and one row per clone, largest
            first.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a ``clones`` or ``clonotypes`` artifact.
            KeyError: If the store holds no such artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the artifact or the receptors of the clonotypes
                are incomplete, as an import that stopped before it finished
                leaves them.
            MemoryError: If the tables, about 250 bytes per cell beside the
                clones that they read and the text of the cells' IDs and of
                the clone labels, do not fit the memory budget.
        """
        group = self._clone_group(clones)
        n_cells = int(as_zarr_array(group["codes"], name="codes").shape[0])
        # The tables are built beside the codes and kept labels read, and the
        # clone labels and the cells' IDs as text.
        self._check_clone_table_budget(
            n_cells,
            _TABLE_BYTES_PER_CELL,
            "clones",
            held=self._clone_input_bytes(clones, with_ids=False)
            + self._clone_text_bytes(clones, n_cells),
        )
        source = self._read_clone_input(clones, with_ids=True)
        assert source.clone_ids is not None
        extra = {}
        if _PROBABILITY in group:
            extra[_PROBABILITY] = _read(group, _PROBABILITY).astype(np.float64)
        # Every saved clone holds a cell, so the codes give the clone sizes.
        assigned = source.codes >= 0
        clone_sizes = np.bincount(
            source.codes[assigned], minlength=len(source.clone_ids)
        )
        clone_size = np.zeros(len(source.codes), dtype=np.int64)
        clone_size[assigned] = clone_sizes[source.codes[assigned]]
        cells, table = clone_frames(
            self._clone_cell_ids(clones),
            codes=source.codes,
            clone_size=clone_size,
            clone_ids=source.clone_ids,
            clone_sizes=clone_sizes,
            extra=extra,
            kept_apart=dict(source.kept_apart),
        )
        return Clones(
            cells=cells,
            clones=table,
            source=str(inspect_artifact(self.zw, clones).operation),
            kept_apart=tuple(name for name, _labels in source.kept_apart),
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
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            path: The file to write.
            format: ``"table"`` or ``"matrix"``.

        Returns:
            The number of cells written.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a ``clones`` or ``clonotypes``
                artifact, ``format`` is unknown, a ``"table"`` would hold a
                clone label that reads as no clone, such as ``"NA"``, or a
                ``"matrix"`` would list a label or cell ID that holds a line
                break.
            KeyError: If the store holds no such artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the artifact or the receptors of the clonotypes
                are incomplete, as an import that stopped before it finished
                leaves them.
            MemoryError: If the export, about 200 bytes per cell beside the
                clones that it reads and the text of the cells' IDs and of the
                clone labels, does not fit the memory budget.
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
        n_cells = int(as_zarr_array(group["codes"], name="codes").shape[0])
        # The file is written from the codes and kept labels read, and the
        # clone labels and the cells' IDs as text.
        resident = (
            _EXPORT_BYTES_PER_CELL * n_cells
            + self._clone_input_bytes(clones, with_ids=False)
            + self._clone_text_bytes(clones, n_cells)
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Exporting the clones of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        source = self._read_clone_input(clones, with_ids=True)
        assert source.clone_ids is not None
        codes, clone_ids = source.codes, source.clone_ids
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

    def _clone_input_bytes(self, clones: ArtifactRef, *, with_ids: bool) -> int:
        """The bytes that :meth:`_read_clone_input` holds for a clone artifact.

        Only the arrays' metadata is read: each cell's code and, with
        ``with_ids``, each clone's label, which costs as much as its stored
        text is wide. Clonotypes also read each cell's receptor type as text,
        code it, and keep a label for each cell.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a kind of clone artifact that the
                summaries read.
            KeyError: If the store holds no such artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the artifact or the receptors of the clonotypes
                are incomplete.
        """
        group = self._clone_group(clones)
        codes = as_zarr_array(group["codes"], name="codes")
        n_cells = int(codes.shape[0])
        resident = n_cells * (codes.dtype.itemsize + _CLONE_CODE_BYTES)
        if with_ids:
            name = _CLONE_ID_ARRAYS[clones.kind]
            ids = as_zarr_array(group[name], name=name)
            resident += int(ids.shape[0]) * (
                _CLONE_ID_BYTES + _CLONE_ID_COPIES * ids.dtype.itemsize
            )
        if clones.kind == ClonotypeArguments.artifact_kind:
            receptor_type = self._receptor_types(clones)
            resident += int(receptor_type.shape[0]) * (
                receptor_type.dtype.itemsize + _RECEPTOR_TYPE_CODING_BYTES
            )
        return resident + n_cells * _KEPT_LABEL_BYTES * len(
            _KEPT_APART_COLUMNS[clones.kind]
        )

    def _receptor_types(self, clonotypes: ArtifactRef) -> zarr.Array:
        """The stored receptor type of each cell of saved clonotypes.

        Clonotypes label the cells of their receptors, in the same order, and
        the receptors' canonical values are the receptor types.

        Raises:
            KeyError: If the store holds no such receptors.
            RuntimeError: If the receptors are incomplete.
        """
        receptors = inspect_artifact(self.zw, clonotypes).input_ref("receptors")
        require_complete_artifact(self.zw, receptors)
        return as_zarr_array(
            artifact_group(self.zw, receptors)["values"], name="values"
        )

    def _read_clone_input(
        self, clones: ArtifactRef, *, with_ids: bool = False
    ) -> _CloneInput:
        """Read a clone artifact as every clone summary reads it.

        Each cell's clone code is read, and with ``with_ids`` the label of
        each clone. Every kind of clone artifact that the summaries accept is
        read here, with the labels of the groups that its clones keep apart:
        clones keep none, and clonotypes keep each cell's receptor type apart,
        the canonical values of their receptors, so the summaries measure T
        cells and B cells apart.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a kind of clone artifact that the
                summaries read.
            KeyError: If the store holds no such artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the artifact or the receptors of the clonotypes
                are incomplete.
            MemoryError: If the codes, and the labels when read, do not fit
                the memory budget.
        """
        resident = self._clone_input_bytes(clones, with_ids=with_ids)
        group = artifact_group(self.zw, clones)
        if resident > self.memoryBytes:
            n_cells = as_zarr_array(group["codes"], name="codes").shape[0]
            raise MemoryError(
                f"Reading the clones of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        kept_apart: tuple[tuple[str, np.ndarray], ...] = ()
        if clones.kind == ClonotypeArguments.artifact_kind:
            # Each of the few receptor types becomes one shared label, so the
            # summaries code the cells' labels without a string per cell.
            codes, names = category_codes(np.asarray(self._receptor_types(clones)[:]))
            labels = np.asarray([*names, None], dtype=object)[codes]
            kept_apart = ((_RECEPTOR_TYPE_COLUMN, labels),)
        ids = _CLONE_ID_ARRAYS[clones.kind]
        return _CloneInput(
            ref=clones,
            cell_selection=inspect_artifact(self.zw, clones).input_ref(
                "cell_selection"
            ),
            codes=_read(group, "codes").astype(np.int64, copy=False),
            clone_ids=_read(group, ids).astype(object) if with_ids else None,
            kept_apart=kept_apart,
        )

    def _matching_bytes(
        self,
        clones: ArtifactRef,
        held: ArtifactRef,
        measured: ArtifactRef,
        n_measured: int,
        n_labels: int,
    ) -> int:
        """The bytes that matching a clone artifact's cells to other measured cells holds.

        Beside what :func:`_matched_bytes` counts for the cells, the two
        selections, ``held`` of the clones' cells and ``measured``, are read
        one after the other, and the read of each can still hold the stored
        chunk of the read before, of the other selection or of the clones'
        codes.
        """
        codes = as_zarr_array(artifact_group(self.zw, clones)["codes"], name="codes")
        return _matched_bytes(
            int(codes.shape[0]), n_measured, n_labels
        ) + previous_read_bytes(
            selection_chunk_bytes(self.zw, measured),
            selection_chunk_bytes(self.zw, held),
            array_row_selection_parts(codes)[0],
        )

    def _clone_figure_labels(
        self, clones: ArtifactRef, ids: Sequence[str]
    ) -> np.ndarray:
        """The labels that figures give some clones of a clone artifact.

        A clone is labelled by its ID. A clonotype is labelled by its ID and
        the junction amino acids of its representative cell's ``VDJ_1``
        chain, or of its ``VJ_1`` chain without a VDJ chain, such as
        ``clonotype_1 (CASSLGQAYEQYF)``; clonotypes defined by
        ``receptor_arms="VJ"`` by the ``VJ_1`` chain first. Immunologists
        read clonotypes by their junctions, and the ID tells apart clonotypes
        whose junctions repeat. The clonotypes' IDs are read to find those
        given, and the repertoire operations read the junctions.

        Args:
            clones: A clone artifact: ``clones`` or ``clonotypes``.
            ids: The IDs of the clones to label, such as ``clonotype_1``.

        Returns:
            The label of each clone, in the order of ``ids``, as an object
            array.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a complete clone artifact, an ID names no
                clonotype, or a representative has no junction to show.
            MemoryError: If reading the clonotypes' IDs, three times their
                size as read, does not fit the memory budget.
        """
        group = self._clone_group(clones)
        names = np.asarray([str(name) for name in ids], dtype=object)
        if clones.kind != ClonotypeArguments.artifact_kind or not len(names):
            return names
        stored_ids = as_zarr_array(group["clonotype_ids"], name="clonotype_ids")
        n_clonotypes = int(stored_ids.shape[0])
        # The IDs as read, the buffer they are decoded from and their
        # compressed bytes, each at most their size, and a comparison of them
        # with an ID given: 56 MiB traced for 410,365 IDs of up to 16
        # characters, 26 MB as read.
        resident = 3 * int(stored_ids.nbytes) + 2 * n_clonotypes
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Labelling clonotypes reads the IDs of {n_clonotypes} clonotypes, "
                f"about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        stored = np.asarray(stored_ids[:])
        codes = np.full(len(names), -1, dtype=np.int64)
        for at, name in enumerate(names.tolist()):
            hits = np.flatnonzero(stored == name)
            if len(hits):
                codes[at] = hits[0]
        del stored
        if (codes < 0).any():
            unknown = names[codes < 0].tolist()
            raise ValueError(f"No clonotype is labelled {unknown[:5]}")
        junctions = self._representative_junctions(clones, codes)
        unnamed = np.asarray([junction is None for junction in junctions.tolist()])
        if unnamed.any():
            raise ValueError(
                f"The representative cells of the clonotypes "
                f"{names[unnamed].tolist()[:5]} have no junction amino-acid "
                "sequence in the VDJ_1 or VJ_1 chain that names them"
            )
        return np.asarray(
            [
                f"{name} ({junction})"
                for name, junction in zip(
                    names.tolist(), junctions.tolist(), strict=True
                )
            ],
            dtype=object,
        )

    def _measured_clone_cells(
        self, source: _CloneInput, cell_selection: ArtifactRef | None
    ) -> tuple[ArtifactRef, np.ndarray, tuple[tuple[str, np.ndarray], ...]]:
        """The cells that a saved summary labels, with their clones.

        The cells are those of ``cell_selection``, or the clones' cells
        without one. A selected cell that the clones do not hold has no
        clone.

        Returns:
            The cells' selection, each cell's clone code, -1 for none, and
            the labels that the clones keep apart, aligned to the cells.

        Raises:
            MemoryError: If matching the clones' cells to other cells does
                not fit the memory budget.
        """
        if cell_selection is None or cell_selection == source.cell_selection:
            return source.cell_selection, source.codes, source.kept_apart
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef or None")
        if cell_selection.kind != "cell_selection":
            raise ValueError(
                "cell_selection must be a cell_selection artifact, not "
                f"{cell_selection.kind!r}"
            )
        n_measured = int(
            validate_cell_selection(self.zw, cell_selection).selected_count
        )
        resident = self._matching_bytes(
            source.ref,
            source.cell_selection,
            cell_selection,
            n_measured,
            len(source.kept_apart),
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Matching the clones of {len(source.codes)} cells to "
                f"{n_measured} measured cells holds about {resident} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget."
            )
        measured = self._clone_selection_indices(cell_selection)
        held = self._clone_selection_indices(source.cell_selection)
        at = pd.Index(held).get_indexer(measured)
        found = at >= 0
        codes = np.full(len(measured), -1, dtype=np.int64)
        codes[found] = source.codes[at[found]]
        kept = []
        for name, labels in source.kept_apart:
            aligned = np.full(len(measured), None, dtype=object)
            aligned[found] = np.asarray(labels, dtype=object)[at[found]]
            kept.append((name, aligned))
        return cell_selection, codes, tuple(kept)

    def _clone_pair_counts(
        self,
        source: _CloneInput,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        cell_selection: ArtifactRef | None,
        reserved: frozenset[str],
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Group the cells with a clone and count each clone's cells in each group.

        The groups are those of :func:`cell_groups`, among the cells with a
        clone, with the labels that the clones keep apart.

        Returns:
            One row per group, in the natural order of the labels, with its
            name in ``group`` and its labels in the key columns, and for each
            distinct pair of group and clone, sorted by group and clone code,
            its group, its clone code and its cells.
        """
        grouped = cell_groups(
            self.zw,
            self.cells,
            source.cell_selection,
            source.codes >= 0,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            memory_bytes=self.memoryBytes,
            kept_apart=source.kept_apart,
            reserved=reserved,
        )
        counted = grouped.codes >= 0
        pair_groups, pair_codes, pair_cells = pair_counts(
            grouped.codes[counted], source.codes[counted]
        )
        keys = group_frame(grouped.key_columns, grouped.keys)
        return keys, pair_groups, pair_codes, pair_cells

    def _grouped_clone_sizes(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        cell_selection: ArtifactRef | None,
        reserved: frozenset[str],
    ) -> tuple[pd.DataFrame, list[CloneSizes]]:
        """Group the cells with a clone and summarize each group's clone sizes.

        Raises:
            ValueError: If no cell has both a clone and a group.
        """
        keys, pair_groups, _codes, pair_cells = self._clone_pair_counts(
            self._read_clone_input(clones),
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=reserved,
        )
        resident = _SUMMARY_BYTES_PER_GROUP * len(keys)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Summarizing the clones of {len(keys)} groups holds about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Group the cells by fewer labels, or "
                "raise mem_budget."
            )
        bounds = np.searchsorted(pair_groups, np.arange(len(keys) + 1))
        clone_sizes = [
            CloneSizes.from_sizes(pair_cells[bounds[index] : bounds[index + 1]])
            for index in range(len(keys))
        ]
        if not clone_sizes or all(sizes.n == 0 for sizes in clone_sizes):
            raise ValueError("No cell has both a clone and a group")
        return keys, clone_sizes

    def _check_clone_table_budget(
        self, n_cells: int, per_cell: int, name: str, *, held: int = 0
    ) -> None:
        """Refuse to load tables of ``n_cells`` cells beyond the memory budget.

        ``held`` counts the bytes of the inputs that the tables are built from.
        """
        resident = per_cell * int(n_cells) + int(held)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Loading the {name} of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )

    @validation_scoped
    def _clones_expansion(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
        breakpoints: Sequence[int] | None = None,
        frequency_breakpoints: Sequence[float] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Place each cell in an expansion category by its clone's size in its group.

        A clone that holds more than one cell has expanded: its founding cell
        divided. Its size is counted within groups of cells: the
        combinations of the labels of the ``group_by`` cell metadata
        columns, such as the sample, and of the ``groups`` label artifact,
        such as a clustering. Without a grouping, a clone's size counts its
        cells in every sample, as scirpy's ``clonal_expansion`` does without
        ``expanded_in``. Cells without a clone or a label are counted in no
        group. Clonotypes are counted within each receptor type, which joins
        the groups' key columns as ``receptor_type``.

        ``breakpoints`` sorts the cells by their clone's cells in the group,
        as scirpy's ``breakpoints`` do: the default ``(1, 2, 5, 20, 100)``
        makes the categories ``"1 cell"``, ``"2 cells"``, ``"3 to 5 cells"``,
        ``"6 to 20 cells"``, ``"21 to 100 cells"`` and ``"more than 100
        cells"``. ``frequency_breakpoints`` sorts them instead by their
        clone's frequency, its share of the group's cells with a clone, in
        bins ``(lower, upper]`` that end at 1, as scRepertoire's
        ``clonalHomeostasis`` does: ``(0.0001, 0.001, 0.01, 0.1, 1)`` makes
        its Rare, Small, Medium, Large and Hyperexpanded categories, named
        ``"0 to 0.0001"`` to ``"0.1 to 1"``. A frequency depends on the
        group's number of cells, so compare groups of similar sizes, or bounds
        in cells.

        The artifact labels the cells of ``cell_selection``, or the clones'
        cells without one; a selected cell that the clones do not hold has
        no clone. Its canonical ``values`` hold each cell's category, empty
        for a cell without a clone or a group, so
        ``ds.select_cells(expansion, include=["more than 100 cells"])`` keeps
        the cells of the largest clones. ``group_clone_size``,
        ``group_clone_frequency`` and ``group_codes`` are its other per-cell
        arrays.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the groups,
                such as the sample.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: The cells to measure, or None for every cell of
                the clones.
            breakpoints: Strictly increasing bounds in cells, or None.
            frequency_breakpoints: Strictly increasing frequency bounds, the
                last 1, or None. Pass at most one of the two bounds; without
                either, ``breakpoints`` are ``(1, 2, 5, 20, 100)``.
            invalidate_cache: Measure again instead of reusing a saved result.

        Returns:
            A complete datastore-scoped ``clonal_expansion`` artifact. Read it
            with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_expansion`
            or plot it with ``ds.plots.clonal_expansion``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, both bounds
                are given, the bounds are invalid, no measured cell has both a
                clone and a group, a column of the groups is missing or named
                twice, or two groups get one name.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            PermissionError: If the store is read-only and the result is not
                saved yet.
            MemoryError: If reading the clones, grouping the cells, or the
                measurement, about 300 bytes per cell, does not fit the memory
                budget.
        """
        bool_argument(invalidate_cache, "invalidate_cache")
        if breakpoints is not None and frequency_breakpoints is not None:
            raise ValueError("Pass breakpoints or frequency_breakpoints, not both")
        counts: tuple[int, ...] | None = None
        frequencies: tuple[float, ...] | None = None
        if frequency_breakpoints is None:
            counts = checked_bounds(
                DEFAULT_BREAKPOINTS if breakpoints is None else breakpoints,
                "breakpoints",
                whole=True,
            )
        else:
            frequencies = checked_bounds(
                frequency_breakpoints, "frequency_breakpoints", whole=False
            )
            # Naming the bins checks that the last bound is 1.
            frequency_labels(frequencies)
        # The clones' own codes are released once they follow the measured
        # cells.
        measured, codes, kept_apart = self._measured_clone_cells(
            self._read_clone_input(clones), cell_selection
        )
        grouped = cell_groups(
            self.zw,
            self.cells,
            measured,
            codes >= 0,
            group_by=group_by,
            groups=groups,
            cell_selection=None,
            memory_bytes=self.memoryBytes,
            kept_apart=kept_apart,
            reserved=_EXPANSION_COLUMNS,
        )
        if not grouped.keys:
            raise ValueError("No cell has both a clone and a group")
        columns = group_columns(group_by)
        arguments = ClonalExpansionArguments(
            clones=clones,
            groups=groups,
            cell_selection=measured,
            # Each measured cell's group by its labels, so other labels in the
            # columns give another result.
            group_assignments_fingerprint=grouped.fingerprint() if columns else None,
            group_by=columns,
            breakpoints=counts,
            frequency_breakpoints=frequencies,
            invalidate_cache=invalidate_cache,
        )
        n_cells = len(codes)
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope="datastore",
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=measured,
            arrays={
                "values": ((n_cells,), None),
                "group_clone_size": ((n_cells,), "i"),
                "group_clone_frequency": ((n_cells,), "f"),
                "group_codes": ((n_cells,), "i"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("labels", "group_keys", "key_columns"),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        # The labels that the clones keep apart stay beside the measurement.
        resident = (
            _EXPANSION_BYTES_PER_CELL + _KEPT_LABEL_BYTES * len(kept_apart)
        ) * n_cells + _SUMMARY_BYTES_PER_GROUP * len(grouped.keys)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Measuring the clonal expansion of {n_cells} cells holds about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        sizes, clone_frequencies = group_clone_sizes(codes, grouped.codes)
        categories, labels = expansion_categories(
            sizes,
            clone_frequencies,
            breakpoints=counts,
            frequency_breakpoints=frequencies,
        )
        counted = categories >= 0
        names = np.asarray(labels, dtype=object)
        values = np.full(n_cells, "", dtype=object)
        values[counted] = names[categories[counted]]
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": values,
                "group_clone_size": sizes,
                "group_clone_frequency": clone_frequencies,
                "group_codes": grouped.codes.astype(np.int32),
            },
            attributes={
                "labels": list(labels),
                "group_keys": grouped.keys,
                "key_columns": list(grouped.key_columns),
            },
        )
        n_counted = int(counted.sum())
        n_ungrouped = int(np.count_nonzero(codes >= 0)) - n_counted
        logger.info(
            f"Measured the clonal expansion of {n_counted} cells with a clone in "
            f"{len(grouped.keys)} groups"
            + (
                f"; {n_ungrouped} cells with a clone have no group"
                if n_ungrouped
                else ""
            )
        )
        return planned.ref

    @validation_scoped
    def _clones_load_expansion(
        self,
        expansion: ArtifactRef,
        *,
        splits: Sequence[int] = DEFAULT_RANK_SPLITS,
    ) -> ClonalExpansion:
        """Read saved clonal expansion as tables of cells, categories and ranks.

        Besides each cell's category and each group's cells and clones per
        category, the loader ranks each group's clones by their cells and
        counts the cells that the clones in each window of ranks hold, as
        scRepertoire's ``clonalProportion`` does: by default the 10 largest,
        11 to 100, 101 to 1000, 1001 to 10000, 10001 to 30000, 30001 to
        100000, and the rest.

        Args:
            expansion: A ``clonal_expansion`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.expansion`.
            splits: The last rank of each window but the last, strictly
                increasing.

        Returns:
            The clonal expansion, with one row per cell, one per group and
            category, and one per group and rank window.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``expansion`` is not a ``clonal_expansion``
                artifact, or ``splits`` are invalid.
            KeyError: If the store holds no such artifact, or not the clones
                that it measured or the receptors of clonotypes.
            RuntimeError: If the artifact, the clones that it measured or the
                receptors of clonotypes are incomplete.
            MemoryError: If the tables, about 500 bytes per cell and 2,000
                per group besides the cells' IDs as text and the clones and
                labels that they read, do not fit the memory budget.
        """
        if not isinstance(expansion, ArtifactRef):
            raise TypeError("expansion must be an ArtifactRef")
        if expansion.kind != ClonalExpansionArguments.artifact_kind:
            raise ValueError(
                f"expansion must be a clonal_expansion artifact, not {expansion.kind!r}"
            )
        bounds = checked_bounds(splits, "splits", whole=True)
        require_complete_artifact(self.zw, expansion)
        group = artifact_group(self.zw, expansion)
        status = inspect_artifact(self.zw, expansion)
        parameters = dict(status.parameters or {})
        clones = status.input_ref("clones")
        groups = (
            None
            if (status.inputs or {}).get("groups") is None
            else status.input_ref("groups")
        )
        measured = status.input_ref("cell_selection")
        n_cells = as_zarr_array(group["group_codes"], name="group_codes").shape[0]
        n_groups = len(cast(list[Any], group.attrs["group_keys"]))
        # Beside the tables of the cells: their IDs as text, each group's rows,
        # the clones' codes and labels, and the clones' cells matched to the
        # measured cells when the expansion measured others.
        held = _SUMMARY_BYTES_PER_GROUP * int(n_groups)
        held += text_read_bytes(self.cells._get_array("ids"), int(n_cells))
        held += self._clone_input_bytes(clones, with_ids=True)
        clone_cells = inspect_artifact(self.zw, clones).input_ref("cell_selection")
        if measured != clone_cells:
            held += self._matching_bytes(
                clones,
                clone_cells,
                measured,
                n_cells,
                len(_KEPT_APART_COLUMNS[clones.kind]),
            )
        self._check_clone_table_budget(
            n_cells, _EXPANSION_TABLE_BYTES_PER_CELL, "clonal expansion", held=held
        )
        source = self._read_clone_input(clones, with_ids=True)
        _cells, codes, _kept = self._measured_clone_cells(source, measured)
        labels = tuple(str(name) for name in cast(list[str], group.attrs["labels"]))
        group_keys = [
            {str(column): str(label) for column, label in key.items()}
            for key in cast(list[dict[str, str]], group.attrs["group_keys"])
        ]
        key_columns = tuple(
            str(column) for column in cast(list[str], group.attrs["key_columns"])
        )
        counts = parameters["breakpoints"]
        frequencies = parameters["frequency_breakpoints"]
        sizes = _read(group, "group_clone_size")
        clone_frequencies = _read(group, "group_clone_frequency")
        # Each cell's category follows from its clone size or frequency and
        # the bounds, as it was saved.
        categories, _labels = expansion_categories(
            sizes,
            clone_frequencies,
            breakpoints=counts,
            frequency_breakpoints=frequencies,
        )
        assert source.clone_ids is not None
        cells, table, ranks = expansion_tables(
            self._clone_cell_ids(expansion),
            categories=categories,
            labels=labels,
            clones=codes,
            clone_ids=source.clone_ids,
            groups=_read(group, "group_codes"),
            group_keys=group_keys,
            key_columns=key_columns,
            sizes=sizes,
            frequencies=clone_frequencies,
            splits=bounds,
        )
        return ClonalExpansion(
            cells=cells,
            categories=table,
            rank_windows=ranks,
            clones=clones,
            labels=labels,
            breakpoints=None if counts is None else tuple(int(v) for v in counts),
            frequency_breakpoints=(
                None if frequencies is None else tuple(float(v) for v in frequencies)
            ),
            splits=bounds,
            group_by=tuple(str(column) for column in parameters["group_by"]),
            groups=groups,
            cell_selection=measured,
        )

    @validation_scoped
    def _clones_diversity(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
        size: int | None = None,
        coverage: float | None = None,
    ) -> ClonalDiversity:
        """Measure the diversity of the clones of groups of cells.

        The groups are the combinations of the labels of the ``group_by``
        cell metadata columns, such as the sample, and of the ``groups``
        label artifact, such as a clustering. Cells without a clone or a
        label are left out. A clone's size in a group is its cells there.
        Clonotypes are measured within each receptor type, which joins the
        groups' key columns as ``receptor_type``.

        The indices are those of scRepertoire's ``clonalDiversity``,
        immunarch's ``repDiversity`` and vegan's ``estimateR``: Shannon
        entropy, Pielou's evenness, the Simpson indices, the Gini coefficient,
        D50, Chao1 in its classic and bias-corrected forms, and ACE. Each
        depends on how many cells a group holds, so the Hill numbers of
        orders 0, 1 and 2, the effective numbers of clones, are also rarefied
        or extrapolated to a common ``size`` and to a common sample
        ``coverage``, as iNEXT's ``estimateD`` does (Chao et al. 2014): by
        default the smallest of the groups' doubled sizes, and the lowest
        coverage of the groups at their doubled sizes. Compare groups by
        these standardized numbers. Nothing is saved.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell of the
                clones.
            size: The common size in cells, or None for the default.
            coverage: The common coverage, between 0 and 1, or None for the
                default.

        Returns:
            The groups' indices and Hill numbers, observed, asymptotic and
            standardized.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, no cell has a
                clone and a group, a column of the groups is missing or named
                twice, or ``size`` is not positive.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            MemoryError: If reading the clones or grouping the cells does not
                fit the memory budget.
        """
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, int | np.integer):
                raise TypeError("size must be a positive integer or None")
            if size < 1:
                raise ValueError("size must be a positive integer or None")
        if coverage is not None:
            if isinstance(coverage, bool) or not isinstance(
                coverage, int | float | np.integer | np.floating
            ):
                raise TypeError("coverage must be a number between 0 and 1, or None")
            if not 0 < coverage < 1:
                raise ValueError("coverage must be between 0 and 1, or None")
        keys, clone_sizes = self._grouped_clone_sizes(
            clones,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=_DIVERSITY_COLUMNS,
        )
        table, hill, common, level = diversity_tables(
            clone_sizes, keys, size=size, coverage=coverage
        )
        return ClonalDiversity(
            indices=table,
            hill=hill,
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            size=common,
            coverage=level,
        )

    @validation_scoped
    def _clones_rarefaction(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
        orders: Sequence[int] = ORDERS,
        sizes: Sequence[int] | None = None,
        endpoint: int | None = None,
        knots: int = 40,
    ) -> Rarefaction:
        """Rarefy and extrapolate the Hill numbers of groups of cells.

        Each group's Hill numbers of ``orders`` 0, 1 and 2 are computed for
        samples of more and fewer cells than it holds, with the sample
        coverage of each size, as iNEXT's ``iNEXT`` does (Chao et al. 2014,
        2023). Below a group's size the curve is the expected value of a
        random subsample; above it, an extrapolation that is reliable up to
        about twice the group's size for order 0. The groups are those of
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.diversity`.
        Nothing is saved.

        By default each group's curve has iNEXT's ``knots`` sizes: about half
        evenly from 1 to its size and half from there to ``endpoint``, by
        default twice its size. ``sizes`` gives every group the same sizes
        instead.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell of the
                clones.
            orders: Orders of the Hill numbers, from 0, 1 and 2.
            sizes: Sizes in cells for every group, or None for each group's
                default sizes.
            endpoint: The largest default size, or None for twice each
                group's size.
            knots: The number of default sizes.

        Returns:
            The curves, one row per group, order and size.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, an order,
                size, endpoint or knots is invalid, or no cell has a clone and
                a group.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            MemoryError: If reading the clones, grouping the cells, or the
                curves do not fit the memory budget.
        """
        listed_orders = _listed(orders)
        if listed_orders is None:
            raise TypeError("orders must be a sequence of 0, 1 and 2")
        if (
            not listed_orders
            or any(
                isinstance(order, bool) or order not in ORDERS
                for order in listed_orders
            )
            or len(set(listed_orders)) != len(listed_orders)
        ):
            raise ValueError("orders must hold distinct orders from 0, 1 and 2")
        chosen = tuple(int(order) for order in listed_orders)
        listed_sizes = None if sizes is None else _listed(sizes)
        if sizes is not None:
            if listed_sizes is None:
                raise TypeError("sizes must be a sequence of positive integers")
            if not listed_sizes or any(
                isinstance(value, bool)
                or not isinstance(value, int | np.integer)
                or not 1 <= value <= LARGEST_SIZE
                for value in listed_sizes
            ):
                raise ValueError(
                    f"sizes must hold positive integers up to {LARGEST_SIZE}"
                )
        # The defaults' arguments are checked once, before any group is read.
        endpoint, knots = checked_curve_arguments(endpoint, knots)
        keys, clone_sizes = self._grouped_clone_sizes(
            clones,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=_RAREFACTION_COLUMNS,
        )
        # Each group is rarefied in turn to at most this many sizes, and the
        # curves hold a row per group, order and size.
        n_sizes = knots if listed_sizes is None else len(listed_sizes)
        largest = max(
            min(n_sizes, group.n) * len(group.values) for group in clone_sizes
        )
        n_rows = len(clone_sizes) * len(chosen) * n_sizes
        resident = (
            _RAREFACTION_BYTES_PER_ENTRY * largest + _CURVE_BYTES_PER_ROW * n_rows
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Rarefying {len(clone_sizes)} groups to {n_sizes} sizes holds "
                f"about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Pass fewer sizes or knots, or raise "
                "mem_budget."
            )
        return Rarefaction(
            curves=rarefaction_curves(
                clone_sizes,
                keys,
                chosen,
                sizes=listed_sizes,
                endpoint=endpoint,
                knots=knots,
            ),
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            orders=chosen,
        )

    @validation_scoped
    def _clones_overlap(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
    ) -> CloneOverlap:
        """Compare groups of cells by the clones they share.

        The groups are the combinations of the labels of the ``group_by``
        cell metadata columns, such as the sample or the tissue, and of the
        ``groups`` label artifact, such as a clustering, among the cells of
        ``cell_selection``. Cells without a clone or a label are left out.
        Clonotypes are grouped within each receptor type, which joins the
        groups' key columns as ``receptor_type``, and only groups of one
        receptor type are compared. Nothing is saved.

        Each pair of groups gets presence indices, which count clones, and
        abundance indices, which weigh them by their cells, as scRepertoire's
        ``clonalOverlap`` and immunarch's ``repOverlap`` report them:
        ``shared`` (the clones of both), ``jaccard``, ``sorensen``,
        ``overlap`` (shared over the smaller group's clones), ``cosine``
        (Ochiai's, of presence), ``morisita_horn`` and ``abundance_cosine``,
        and the share of each group's cells whose clone the other holds.
        Presence indices grow with the cells of both groups, since a larger
        sample finds more of the rare clones; the Morisita-Horn index depends
        little on them and is near 1 when the same expanded clones dominate
        both groups.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell of the
                clones.

        Returns:
            The groups' cells and clones and the indices of every pair.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, no cell has
                both a clone and a group, fewer than two groups hold a cell
                with a clone, no two groups share the labels that the clones
                keep apart, a column of the groups is missing or named twice,
                a key column names a column of the result, or two groups get
                one name.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            MemoryError: If reading the clones, grouping the cells, or the
                pairs of groups do not fit the memory budget.
        """
        source = self._read_clone_input(clones)
        keys, pair_groups, pair_codes, pair_cells = self._clone_pair_counts(
            source,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=_OVERLAP_COLUMNS,
        )
        if len(pair_cells) == 0:
            raise ValueError("No cell has both a clone and a group")
        if len(keys) < 2:
            raise ValueError("Overlap needs at least two groups of cells with a clone")
        strata = _kept_apart_strata(source, keys)
        n_pairs, largest = overlap_extent(len(keys), strata)
        if n_pairs == 0:
            raise ValueError(
                f"No two groups share their labels in {_kept_apart(source)}, "
                "which the clones keep apart, so no pair of groups can be "
                "compared"
            )
        # The pairs, and beside them the clones read, their counts in each
        # group and each group's rows.
        resident = (
            _OVERLAP_BYTES_PER_PAIR * n_pairs
            + _OVERLAP_BYTES_PER_PRODUCT * largest**2
            + _BYTES_PER_CLONE_COUNT * len(pair_cells)
            + _SUMMARY_BYTES_PER_GROUP * len(keys)
            + self._clone_input_bytes(clones, with_ids=False)
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Comparing {len(keys)} groups gives {n_pairs} pairs and holds "
                f"about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Compare fewer groups, with fewer "
                "group_by columns or a cell_selection, or raise mem_budget."
            )
        pairs = overlap_pairs(
            pair_groups, pair_codes, pair_cells, len(keys), strata=strata
        )
        names = keys["group"].to_numpy(dtype=object)
        pairs.insert(0, "group_1", names[pairs.pop("first").to_numpy()])
        pairs.insert(1, "group_2", names[pairs.pop("second").to_numpy()])
        return CloneOverlap(
            group_sizes=_group_sizes(keys, pair_groups, pair_cells),
            pairs=pairs,
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
        )

    @validation_scoped
    def _clones_track(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
        top: int = 10,
        labels: Sequence[str] | None = None,
    ) -> CloneTracking:
        """Follow chosen clones across groups of cells.

        By default the clones are the ``top`` with the most cells in each
        group, together, as scRepertoire's ``clonalCompare`` chooses them,
        with ties going to the larger clone overall; ``labels`` names them
        instead, such as ``["3"]``. Each chosen clone gets its cells and share
        of cells in every group, zeros included, so a clone that expands
        between time points or spreads between cell types shows it. The
        groups are those of
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.overlap`, and a
        clonotype is followed through the groups of its own receptor type.
        Nothing is saved.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell of the
                clones.
            top: The clones with the most cells to take from each group.
            labels: Clone labels to follow instead, as a sequence or a
                one-dimensional array, or None.

        Returns:
            One row per chosen clone and group, and the chosen clones' labels.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, ``top`` is not
                positive, ``labels`` is empty or a label names no clone, no
                cell has both a clone and a group, a column of the groups is
                missing or named twice, a key column names a column of the
                result, or two groups get one name.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            MemoryError: If reading the clones, grouping the cells, or the
                table of the chosen clones' groups does not fit the memory
                budget.
        """
        if isinstance(top, bool) or not isinstance(top, int | np.integer):
            raise TypeError("top must be a positive integer")
        if top < 1:
            raise ValueError("top must be a positive integer")
        if labels is not None:
            listed = _listed(labels)
            if listed is None or not all(isinstance(label, str) for label in listed):
                raise TypeError("labels must be a sequence of clone labels or None")
            if not listed:
                raise ValueError("labels must name at least one clone, or be None")
            labels = listed
        source = self._read_clone_input(clones, with_ids=True)
        keys, pair_groups, pair_codes, pair_cells = self._clone_pair_counts(
            source,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=_TRACK_COLUMNS,
        )
        if len(pair_cells) == 0:
            raise ValueError("No cell has both a clone and a group")
        assert source.clone_ids is not None
        identifiers = source.clone_ids
        if labels is None:
            chosen = top_clones(
                pair_groups, pair_codes, pair_cells, len(keys), int(top)
            )
        else:
            positions = pd.Index(identifiers).get_indexer(list(labels))
            unknown = [
                label for label, at in zip(labels, positions, strict=True) if at < 0
            ]
            if unknown:
                raise ValueError(f"No clone is labelled {unknown[:5]}")
            chosen = sorted_unique(positions.astype(np.int64))
        strata = _kept_apart_strata(source, keys)
        if strata is None:
            strata = np.zeros(len(keys), dtype=np.int64)
        reach = tracking_reach(
            pair_groups,
            pair_codes,
            chosen,
            strata,
            _fallback_strata(source, keys, strata, chosen),
        )
        n_rows = int((reach * np.bincount(strata, minlength=reach.shape[1])).sum())
        # The rows, each with its clone, group, labels, cells and fraction,
        # and beside them the clones read, their counts in each group and
        # each group's rows.
        n_key_columns = len(keys.columns) - 1
        resident = (
            n_rows
            * (_TRACK_BYTES_PER_ROW + _TRACK_BYTES_PER_KEY_COLUMN * n_key_columns)
            + reach.shape[0] * len(keys)
            + _BYTES_PER_CLONE_COUNT * len(pair_cells)
            + _SUMMARY_BYTES_PER_GROUP * len(keys)
            + self._clone_input_bytes(clones, with_ids=True)
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Following {len(chosen)} clones through {len(keys)} groups "
                f"gives {n_rows} rows and holds about {resident} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Follow "
                "fewer clones, with a smaller top or labels, or fewer groups, "
                "or raise mem_budget."
            )
        followed, rows, counts = tracking_cells(
            pair_groups, pair_codes, pair_cells, chosen, strata, reach
        )
        tracked = np.asarray(identifiers[chosen].tolist(), dtype=object)
        # Without rows, the table keeps its columns and their types; clone
        # labels are text of the groups' labels' type, which pandas 3 stores
        # as its string type and pandas 2 as Python objects, never fixed-width.
        sizes = _group_sizes(keys, pair_groups, pair_cells)
        table = keys.take(rows).reset_index(drop=True)
        table.insert(
            0, "clone", pd.array(tracked[followed], dtype=table["group"].dtype)
        )
        table["cells"] = counts
        table["fraction"] = counts / sizes["cells"].to_numpy()[rows]
        return CloneTracking(
            cells=table,
            clones=clones,
            tracked=tuple(str(label) for label in tracked.tolist()),
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
        )

    @validation_scoped
    def _clones_coupling(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
        method: str = COUPLING_METHODS[0],
        permutations: int = 1000,
        seed: int = 0,
    ) -> CloneCoupling:
        """Measure how strongly groups of cells share clones, against shuffled labels.

        Cell types whose cells often share a clone descend from common
        progenitors, so the clones that two groups share measure how closely
        their fates are coupled. The groups are the combinations of the
        labels of the ``group_by`` cell metadata columns, such as the cell
        type, and of the ``groups`` label artifact, among the cells of
        ``cell_selection``, such as the cells of one time point. Cells
        without a clone or a label are left out. Clonotypes are grouped within
        each receptor type, which joins the groups' key columns as
        ``receptor_type``: only groups of one receptor type are coupled, the
        null shuffles labels within each receptor type, and a clonotype that
        spans receptor types, as ``within_receptor_type=False`` allows,
        counts as one clone in each. Nothing is saved.

        ``method`` names how a pair's coupling is measured from each clone's
        cells in each group, as CoSpar's ``fate_coupling`` measures it:
        ``"share_cosine"``, CoSpar's default, the cosine over the clones of
        each group's share of each clone, after each group's cells of a clone
        are divided by its cells with a clone and each clone's shares by
        their sum, so every clone counts once; ``"abundance_cosine"``, the
        cosine of the clone sizes, which ``overlap`` also reports;
        ``"normalized_covariance"``, the covariance over the clones of the
        two groups' cells over the product of their mean cells per clone,
        negative when they share fewer clones than chance; and ``"jaccard"``,
        the clones of both groups over the clones of either.

        Each value is compared with a null: the group labels of the cells with
        a clone are shuffled ``permutations`` times, which keeps each clone's
        cells and each group's cells, and the coupling measured again. The
        z-score is the value's distance from the null mean in null standard
        deviations, so groups of different sizes compare. Permutation ``k``
        draws from a random stream keyed by ``seed`` and ``k``, so the result
        does not depend on ``nthreads``. The groups are ordered by average
        linkage on the coupling, the most coupled together.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell of the
                clones.
            method: ``"share_cosine"``, ``"abundance_cosine"``,
                ``"normalized_covariance"`` or ``"jaccard"``.
            permutations: The permutations of the null, 0 for none, or at
                least 2 to estimate its spread.
            seed: The seed of the permutations, a non-negative integer below
                2**64.

        Returns:
            The groups' cells and clones, each pair's coupling with its null
            and z-score, and the groups' order.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, ``method`` is
                unknown, ``permutations`` is 1 or negative, ``seed`` is
                negative or 2**64 or more, no cell has both a clone and a
                group, fewer than two groups hold a cell with a clone, no two
                groups share the labels that the clones keep apart, a column
                of the groups is missing or named twice, a key column names a
                column of the result, two groups get one name, or
                ``"normalized_covariance"`` meets fewer than two clones.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            MemoryError: If reading the clones, grouping the cells, or the
                coupling and one worker of its null do not fit the memory
                budget.
        """
        if not isinstance(method, str) or method not in COUPLING_METHODS:
            raise ValueError(
                f"method must be one of {', '.join(COUPLING_METHODS)}, not {method!r}"
            )
        if isinstance(permutations, bool) or not isinstance(
            permutations, int | np.integer
        ):
            raise TypeError("permutations must be a non-negative integer")
        if permutations < 0 or permutations == 1:
            raise ValueError(
                "permutations must be 0, for no null, or at least 2, to estimate "
                "its spread"
            )
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise TypeError("seed must be a non-negative integer")
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be a non-negative integer below 2**64")
        source = self._read_clone_input(clones)
        keys, pair_groups, pair_codes, pair_cells = self._clone_pair_counts(
            source,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=_COUPLING_COLUMNS,
        )
        if len(pair_cells) == 0:
            raise ValueError("No cell has both a clone and a group")
        if len(keys) < 2:
            raise ValueError("Coupling needs at least two groups of cells with a clone")
        strata = _kept_apart_strata(source, keys)
        n_pairs, _largest = overlap_extent(len(keys), strata)
        if n_pairs == 0:
            raise ValueError(
                f"No two groups share their labels in {_kept_apart(source)}, "
                "which the clones keep apart, so no pair of groups can be "
                "compared"
            )
        if strata is not None:
            pair_codes = _clones_by_stratum(pair_codes, strata[pair_groups])
        # Beside the coupling: the clones read, their counts in each group
        # and each group's rows.
        held = (
            _BYTES_PER_CLONE_COUNT * len(pair_cells)
            + _SUMMARY_BYTES_PER_GROUP * len(keys)
            + self._clone_input_bytes(clones, with_ids=False)
        )
        import numba

        workers, batch, needed = coupling_plan(
            len(keys),
            n_pairs,
            int(pair_cells.sum()),
            len(pair_cells),
            permutations=int(permutations),
            # More workers than Numba's threads would only hold scratch.
            nthreads=min(
                self.nthreads, int(getattr(numba.config, "NUMBA_NUM_THREADS"))
            ),
            available=self.memoryBytes - held,
        )
        resident = held + needed
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Measuring the coupling of {len(keys)} groups, {n_pairs} pairs, "
                f"holds about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Compare fewer groups, with fewer "
                "group_by columns or a cell_selection, or raise mem_budget."
            )
        from scipy.cluster.hierarchy import leaves_list

        pairs, tree = clone_coupling(
            pair_groups,
            pair_codes,
            pair_cells,
            len(keys),
            method=method,
            strata=strata,
            permutations=int(permutations),
            seed=int(seed),
            workers=workers,
            batch=batch,
        )
        names = keys["group"].to_numpy(dtype=object)
        pairs.insert(0, "group_1", names[pairs.pop("first").to_numpy()])
        pairs.insert(1, "group_2", names[pairs.pop("second").to_numpy()])
        logger.info(
            f"Measured the {method} coupling of {n_pairs} pairs of {len(keys)} "
            f"groups against {int(permutations)} permutations on {workers} "
            f"threads, {batch} at once, holding about {resident} bytes"
        )
        return CloneCoupling(
            group_sizes=_group_sizes(keys, pair_groups, pair_cells),
            pairs=pairs,
            order=tuple(str(name) for name in names[leaves_list(tree)].tolist()),
            linkage=tree,
            method=method,
            permutations=int(permutations),
            seed=int(seed),
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
        )

    @validation_scoped
    def _clones_fate_bias(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        early: ArtifactRef,
        late: ArtifactRef,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Follow each early cell's clone to its late cells, and test each clone's fates.

        The late cells of an early cell's clone, such as its clone-mates
        sampled days later, show which fates its progenitor took. The fates
        are the groups of the late cells: the combinations of the labels of
        the ``group_by`` cell metadata columns, such as the cell type, and of
        the ``groups`` label artifact. A late cell without a label has no
        fate. No cell may be both early and late.

        For each early cell the artifact holds the share of its clone's late
        cells in each fate, the clonal fate map of CoSpar's ``fate_map`` on a
        transition map of clones alone, its dominant fate, the fate that
        holds the most of them, and its clone's late cells. An early cell
        whose clone has no late cell has no fractions and no dominant fate,
        and neither has one whose clone's late cells tie between fates. For
        each clone with late cells it holds their count in each fate and a
        one-sided Fisher test per fate: the chance of at least that many of
        the clone's late cells in the fate if they were drawn at random from
        all late cells with a fate, those without a clone included, as
        CoSpar's ``clonal_fate_bias`` tests a clone among the late cells,
        with the Benjamini and Hochberg adjustment over the clones within
        each fate. When the clones keep labels apart, such as receptor types,
        each fate holds one such label, which a late cell without a clone
        lacks, so it has no fate. The fates of clonotypes therefore name a
        receptor type each, such as ``"Neutrophil, TCR"``, and each clonotype
        is tested among the late cells of its receptor type that have a
        clonotype. A clonotype whose early and late cells span receptor types,
        as ``within_receptor_type=False`` allows, is refused.

        The artifact's canonical ``values`` hold each early cell's dominant
        fate, empty for none, so ``ds.select_cells(bias, include=["Neutrophil"])``
        keeps the early cells whose clones went mostly to neutrophils.
        ``codes``, each cell's clone, ``fate_codes``, its dominant fate's
        position, ``late_cells`` and ``fractions``, one column per fate, are
        its other per-cell arrays.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the fates.
            groups: A cell-label artifact whose labels define the fates.
            early: The early cells, a cell selection.
            late: The late cells, a cell selection.
            invalidate_cache: Measure again instead of reusing a saved result.

        Returns:
            A complete datastore-scoped ``clone_fate_bias`` artifact that
            labels the early cells. Read it with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_fate_bias`
            or plot it with ``ds.plots.clone_fates``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, ``early`` or
                ``late`` is not a cell selection or selects no cell, a cell is
                both early and late, no late cell has both a clone and a fate,
                a clone's early and late cells carry two labels that the clones
                keep apart, fewer than two fates hold a late cell, a column of the fates
                is missing or named twice, a key column names a column of the
                result, two fates get one name, or a fate is named
                ``"clone"``, ``"late_cells"`` or ``"fate"``.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            PermissionError: If the store is read-only and the result is not
                saved yet.
            MemoryError: If reading the clones, grouping the late cells, or
                the measurement does not fit the memory budget.
        """
        bool_argument(invalidate_cache, "invalidate_cache")
        for name, selection in (("early", early), ("late", late)):
            if not isinstance(selection, ArtifactRef):
                raise TypeError(f"{name} must be an ArtifactRef")
            if selection.kind != "cell_selection":
                raise ValueError(
                    f"{name} must be a cell_selection artifact, not {selection.kind!r}"
                )
        source = self._read_clone_input(clones)
        held = self._clone_input_bytes(clones, with_ids=False)
        n_early = int(validate_cell_selection(self.zw, early).selected_count)
        n_late = int(validate_cell_selection(self.zw, late).selected_count)
        for name, count in (("early", n_early), ("late", n_late)):
            if count == 0:
                raise ValueError(f"{name} selects no cell")
        resident = held + _SELECTION_BYTES_PER_CELL * (n_early + n_late)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Checking that none of {n_early} early cells is among {n_late} "
                f"late cells holds about {resident} bytes, more than the memory "
                f"budget of {self.memoryBytes} bytes. Raise mem_budget."
            )
        shared = np.intersect1d(
            self._clone_selection_indices(early),
            self._clone_selection_indices(late),
            assume_unique=True,
        )
        if len(shared):
            raise ValueError(
                f"early and late share {len(shared)} cells; pass cell selections "
                "that no cell is in both, such as two time points"
            )
        del shared
        _late, late_codes, late_kept = self._measured_clone_cells(source, late)
        # The labels that the clones keep apart belong to the clones, so a late
        # cell without a clone has none, and no fate, whatever label its
        # source gives it.
        without_clone = late_codes < 0
        cleared = []
        for name, labels in late_kept:
            kept_labels = np.array(labels, dtype=object)
            kept_labels[without_clone] = None
            cleared.append((name, kept_labels))
        late_kept = tuple(cleared)
        grouped = cell_groups(
            self.zw,
            self.cells,
            late,
            np.ones(len(late_codes), dtype=bool),
            group_by=group_by,
            groups=groups,
            cell_selection=None,
            memory_bytes=self.memoryBytes,
            kept_apart=late_kept,
            reserved=_FATE_BIAS_COLUMNS,
        )
        counted = (late_codes >= 0) & (grouped.codes >= 0)
        if not counted.any():
            raise ValueError("No cell has both a clone and a group")
        keys = group_frame(grouped.key_columns, grouped.keys)
        if len(keys) < 2:
            raise ValueError("Fate bias needs at least two fates among the late cells")
        names = keys["group"].to_numpy(dtype=object)
        clashes = [name for name in names.tolist() if name in FATE_CELL_COLUMNS]
        if clashes:
            raise ValueError(
                f"A fate is named {clashes[0]!r}, which is also a column of the "
                "table of early cells; rename its label"
            )
        columns = group_columns(group_by)
        n_fates = len(keys)
        arguments = CloneFateBiasArguments(
            clones=clones,
            groups=groups,
            cell_selection=early,
            late=late,
            # Each late cell's fate by its labels, so other labels in the
            # columns give another result.
            group_assignments_fingerprint=grouped.fingerprint() if columns else None,
            group_by=columns,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()

        def has_clones(_ref: ArtifactRef, group: zarr.Group) -> bool:
            shapes = {
                name: getattr(group.get(name), "shape", None)
                for name in (
                    "clone_codes",
                    "clone_early_cells",
                    "clone_fate_cells",
                    "clone_p_values",
                    "clone_q_values",
                    "fate_cells",
                )
            }
            n_tested = (shapes["clone_codes"] or (None,))[0]
            return (
                shapes["clone_early_cells"] == (n_tested,)
                and shapes["clone_fate_cells"] == (n_tested, n_fates)
                and shapes["clone_p_values"] == (n_tested, n_fates)
                and shapes["clone_q_values"] == (n_tested, n_fates)
                and shapes["fate_cells"] == (n_fates,)
            )

        planned = plan_cell_data_artifact(
            self.zw,
            scope="datastore",
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=early,
            arrays={
                "values": ((n_early,), None),
                "codes": ((n_early,), "i"),
                "fate_codes": ((n_early,), "i"),
                "late_cells": ((n_early,), "i"),
                "fractions": ((n_early, n_fates), "f"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("group_keys", "key_columns"),
            reuse_validator=has_clones,
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        # Counting each clone's late cells in each fate holds a key per late
        # cell, less than grouping them held.
        pair_fates, pair_codes, pair_cells = pair_counts(
            grouped.codes[counted], late_codes[counted]
        )
        fate_cells = np.bincount(
            grouped.codes[grouped.codes >= 0], minlength=n_fates
        ).astype(np.int64)
        strata = _kept_apart_strata(source, keys)
        # Nothing per late cell is held beyond the counts.
        group_keys, key_columns = grouped.keys, list(grouped.key_columns)
        del grouped, late_codes, late_kept, counted
        n_tested = len(sorted_unique(pair_codes))
        n_late_counted = int(pair_cells.sum())
        _early, early_codes, early_kept = self._measured_clone_cells(source, early)
        _refuse_clones_across_kept_labels(
            keys, pair_fates, pair_codes, early_codes, early_kept
        )
        del early_kept
        width = max(len(str(name)) for name in names.tolist())
        resident = (
            held
            + _BYTES_PER_FATE_PAIR * len(pair_cells)
            + fate_bias_bytes(n_early, n_tested, n_fates)
            + n_early
            * (_BIAS_WRITE_BYTES_PER_CELL + _BIAS_WRITE_BYTES_PER_CHARACTER * width)
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Measuring the fate bias of {n_early} early cells and {n_tested} "
                f"clones in {n_fates} fates holds about {resident} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Group the "
                "late cells into fewer fates, or raise mem_budget."
            )
        result = fate_bias(
            early_codes, pair_fates, pair_codes, pair_cells, fate_cells, strata
        )
        del pair_fates, pair_codes, pair_cells
        values = np.full(n_early, "", dtype=object)
        has_fate = result.dominant >= 0
        values[has_fate] = names[result.dominant[has_fate]]
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": values,
                "codes": early_codes.astype(np.int32),
                "fate_codes": result.dominant.astype(np.int32),
                "late_cells": result.late_cells,
                "fractions": result.fractions,
                "clone_codes": result.clones,
                "clone_early_cells": result.clone_early_cells,
                "clone_fate_cells": result.clone_cells,
                "clone_p_values": result.p_values,
                "clone_q_values": result.q_values,
                "fate_cells": result.fate_cells,
            },
            attributes={"group_keys": group_keys, "key_columns": key_columns},
        )
        logger.info(
            f"Measured the fate bias of {int(np.count_nonzero(result.late_cells))} "
            f"of {n_early} early cells, whose clones hold {n_late_counted} late "
            f"cells in {n_fates} fates, and tested {n_tested} clones"
        )
        return planned.ref

    @validation_scoped
    def _clones_load_fate_bias(self, bias: ArtifactRef) -> CloneFateBias:
        """Read saved fate bias as tables of early cells, clones and fates.

        Args:
            bias: A ``clone_fate_bias`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.fate_bias`.

        Returns:
            The fate bias, with one row per early cell, one per clone with
            late cells and fate of its stratum, and one per fate.

        Raises:
            TypeError: If ``bias`` is not an ArtifactRef.
            ValueError: If it is not a ``clone_fate_bias`` artifact.
            KeyError: If the store holds no such artifact, or not the clones
                that it measured or the receptors of clonotypes.
            RuntimeError: If the artifact, the clones that it measured or the
                receptors of clonotypes are incomplete.
            MemoryError: If the tables, about 200 bytes per early cell, 24 per
                fraction and 160 per row of clone and fate, beside the early
                cells' IDs as text and the clone labels that they read, do not
                fit the memory budget.
        """
        if not isinstance(bias, ArtifactRef):
            raise TypeError("bias must be an ArtifactRef")
        if bias.kind != CloneFateBiasArguments.artifact_kind:
            raise ValueError(
                f"bias must be a clone_fate_bias artifact, not {bias.kind!r}"
            )
        require_complete_artifact(self.zw, bias)
        group = artifact_group(self.zw, bias)
        status = inspect_artifact(self.zw, bias)
        parameters = dict(status.parameters or {})
        clones = status.input_ref("clones")
        groups = (
            None
            if (status.inputs or {}).get("groups") is None
            else status.input_ref("groups")
        )
        group_keys = [
            {str(column): str(label) for column, label in key.items()}
            for key in cast(list[dict[str, str]], group.attrs["group_keys"])
        ]
        key_columns = tuple(
            str(column) for column in cast(list[str], group.attrs["key_columns"])
        )
        n_early = int(as_zarr_array(group["codes"], name="codes").shape[0])
        n_tested = int(as_zarr_array(group["clone_codes"], name="clone_codes").shape[0])
        n_fates = len(group_keys)
        resident = (
            n_early
            * (_BIAS_TABLE_BYTES_PER_CELL + _BIAS_TABLE_BYTES_PER_FRACTION * n_fates)
            + n_tested
            * n_fates
            * (
                _BIAS_TABLE_BYTES_PER_ROW
                + _BIAS_TABLE_BYTES_PER_KEY_COLUMN * len(key_columns)
            )
            + _SUMMARY_BYTES_PER_GROUP * n_fates
            # The early cells' IDs, read as text for the index of their table.
            + text_read_bytes(self.cells._get_array("ids"), n_early)
            + self._clone_input_bytes(clones, with_ids=True)
        )
        self._check_clone_table_budget(n_early, 0, "fate bias", held=resident)
        source = self._read_clone_input(clones, with_ids=True)
        assert source.clone_ids is not None
        keys = group_frame(key_columns, group_keys)
        cells, clone_fates = fate_bias_tables(
            self._clone_cell_ids(bias),
            codes=_read(group, "codes"),
            dominant=_read(group, "fate_codes"),
            late_cells=_read(group, "late_cells"),
            fractions=_read(group, "fractions"),
            clone_ids=source.clone_ids,
            clones=_read(group, "clone_codes"),
            clone_cells=_read(group, "clone_fate_cells"),
            clone_early_cells=_read(group, "clone_early_cells"),
            p_values=_read(group, "clone_p_values"),
            q_values=_read(group, "clone_q_values"),
            fates=keys,
        )
        fates = keys.rename(columns={"group": "fate"}).assign(
            late_cells=_read(group, "fate_cells").astype(np.int64)
        )
        return CloneFateBias(
            cells=cells,
            clone_fates=clone_fates,
            fates=fates,
            clones=clones,
            early=status.input_ref("cell_selection"),
            late=status.input_ref("late"),
            group_by=tuple(str(column) for column in parameters["group_by"]),
            groups=groups,
        )


def _group_sizes(
    keys: pd.DataFrame, pair_groups: np.ndarray, pair_cells: np.ndarray
) -> pd.DataFrame:
    """One row per group: its name, labels, cells with a clone and clones."""
    n_groups = len(keys)
    return keys.assign(
        cells=np.bincount(pair_groups, weights=pair_cells, minlength=n_groups).astype(
            np.int64
        ),
        clones=np.bincount(pair_groups, minlength=n_groups),
    )


def _kept_apart(source: _CloneInput) -> list[str]:
    return [name for name, _labels in source.kept_apart]


def _clones_by_stratum(codes: np.ndarray, code_strata: np.ndarray) -> np.ndarray:
    """Code each clone once in each stratum that its counted cells reach.

    Every clone source promises that a clone's cells share the labels that
    it keeps apart, but clonotypes that ``within_receptor_type=False``
    defines can hold cells of several receptor types. Such a clone counts as
    one clone in each stratum, as it counts its cells of each receptor type
    in that type's groups in the other summaries. The codes come back
    unchanged when no clone spans two strata, and otherwise in the order of
    the clones and then of their strata.

    Args:
        codes: The clone code of each count of a clone in a group.
        code_strata: The stratum of each count's group.
    """
    n_strata = int(code_strata.max()) + 1 if len(code_strata) else 1
    units, recoded = np.unique(
        codes.astype(np.int64) * n_strata + code_strata, return_inverse=True
    )
    if len(units) == len(sorted_unique(codes)):
        return codes
    return recoded.reshape(-1).astype(np.int64)


def _refuse_clones_across_kept_labels(
    fates: pd.DataFrame,
    pair_fates: np.ndarray,
    pair_codes: np.ndarray,
    early_codes: np.ndarray,
    early_kept: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """Refuse clones whose early and late cells carry two labels kept apart.

    Fate bias tests a clone in the fates of the labels that its cells share
    in each column that the clones keep apart. Clonotypes that
    ``within_receptor_type=False`` defines can hold cells of several receptor
    types, and such a clonotype has no one receptor type in which to test it.

    Args:
        fates: One row per fate, with its labels in the key columns.
        pair_fates: The fate of each count of a clone's late cells in a fate.
        pair_codes: The clone code of each count.
        early_codes: Each early cell's clone code, -1 for none.
        early_kept: Each column that the clones keep apart, with each early
            cell's label.

    Raises:
        ValueError: If a clone's late cells lie in fates of two labels of
            such a column, or an early cell's label differs from that of its
            clone's late cells.
    """
    for column, early_labels in early_kept:
        late = pd.DataFrame(
            {
                "clone": pair_codes,
                "label": fates[column].to_numpy(dtype=object)[pair_fates],
            }
        ).drop_duplicates()
        across = set(late.loc[late["clone"].duplicated(keep=False), "clone"].tolist())
        late = late.drop_duplicates("clone")
        at = pd.Index(late["clone"]).get_indexer(early_codes)
        reached = at >= 0
        differs = (
            np.asarray(early_labels, dtype=object)[reached]
            != (late["label"].to_numpy(dtype=object)[at[reached]])
        )
        across |= set(early_codes[reached][differs].tolist())
        if across:
            raise ValueError(
                f"{len(across)} clones hold early or late cells of two labels in "
                f"{column!r}, which the clones keep apart, as clonotypes that "
                "within_receptor_type=False defines can span receptor types; "
                "fate bias tests each clone in the fates of one label. Define "
                "the clonotypes with within_receptor_type=True, or pass early "
                "and late cells of one receptor type."
            )


def _kept_apart_strata(source: _CloneInput, keys: pd.DataFrame) -> np.ndarray | None:
    """Each group's stratum: its labels in the key columns that the clones keep apart.

    Groups of two strata share no clone, so only the groups of one stratum
    are compared, and a clone is followed through the groups of its own
    strata.

    Returns:
        Each group's stratum, a code from 0, or None when the clones keep no
        labels apart and every group is of one stratum.
    """
    columns = _kept_apart(source)
    if not columns:
        return None
    return np.asarray(keys.groupby(columns, sort=True).ngroup(), dtype=np.int64)


def _fallback_strata(
    source: _CloneInput, keys: pd.DataFrame, strata: np.ndarray, chosen: np.ndarray
) -> np.ndarray:
    """The stratum through which to follow each chosen clone without counted cells.

    It is the stratum of the labels that the clones keep apart at the
    clone's first cell, or -1 when no group holds those labels; clones that
    keep no labels apart have the one stratum, 0.
    """
    columns = _kept_apart(source)
    if not columns:
        return np.zeros(len(chosen), dtype=np.int64)
    # Every clone holds a cell of the artifact, so the first position of each
    # chosen code among the sorted codes is one of its cells.
    order = np.argsort(source.codes, kind="stable")
    first = order[np.searchsorted(source.codes[order], chosen)]
    lookup = {
        row: int(stratum)
        for row, stratum in zip(
            keys[columns].itertuples(index=False, name=None),
            strata.tolist(),
            strict=True,
        )
    }
    return np.asarray(
        [
            lookup.get(
                tuple(str(values[cell]) for _name, values in source.kept_apart), -1
            )
            for cell in first.tolist()
        ],
        dtype=np.int64,
    )


def _matched_bytes(n_clone_cells: int, n_cells: int, n_labels: int) -> int:
    """Bytes that matching a clone artifact's cells to other measured cells holds.

    ``n_labels`` counts the columns of labels that the clones keep apart.
    """
    return _MATCH_BYTES_PER_CLONE_CELL * n_clone_cells + n_cells * (
        _MATCH_BYTES_PER_CELL + _MATCH_BYTES_PER_LABEL * n_labels
    )


def _listed(values: Any) -> list[Any] | None:
    """The items of a sequence or a one-dimensional array, or None for anything else."""
    if isinstance(values, np.ndarray):
        return values.tolist() if values.ndim == 1 else None
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        return None
    return list(values)


def _read(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[:])
