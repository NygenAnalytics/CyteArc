import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import zarr

from ...clones.assignments import (
    CLONE_ASSIGNMENT_ROW_BYTES,
    MULTIPLE_POLICIES,
    CloneAssigner,
    clone_assignment_bytes,
    Clones,
    clone_frames,
    MultiplePolicy,
    clone_label_texts,
)
from ...clones.diversity import (
    LARGEST_SIZE,
    ORDERS,
    ClonalDiversity,
    GroupedCloneSizes,
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
    CloneCoupling,
    CloneFateBias,
    clone_coupling,
    coupling_plan,
    fate_bias,
    fate_bias_bytes,
    fate_bias_tables,
)
from ...clones.groups import group_key_table, pair_count_bytes, pair_counts
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
from ...metadata.artifacts import (
    cell_data_write_bytes,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.records import MISSING_TEXT
from ...metadata.rows import (
    metadata_column_stamp,
    metadata_column_stamp_bytes,
    metadata_missing_mask,
    read_rows,
    read_text_rows,
    read_whole,
    read_whole_text,
    text_width,
)
from ...metadata.selection import (
    CELL_VALUE_NAMES,
    CellSelectionRead,
    category_codes,
    plan_cell_selection_read,
    read_cell_selection,
)
from ...repertoire.clonotypes import clonotype_nouns
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.diagnostics import ExportSummary, export_summary, warn_diagnostics
from ...storage.selections import validate_cell_selection
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument, quoted_value
from ...utils.arrays import sorted_unique
from ...utils.digest import file_sha256
from ...utils.logging import logger
from ...utils.memory_budget import (
    NUMPY_STEP_BYTES,
    MemoryLedger,
    hash_table_bytes,
    object_array_bytes,
    python_text_bytes,
)
from .cell_barcodes import (
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    derived_id_bytes,
    unmatched_barcode_rules,
)
from .cell_groups import (
    GROUPING_BYTES_PER_GROUP,
    GROUPING_BYTES_PER_GROUP_COLUMN,
    CellGroups,
    cell_groups,
    group_columns,
)
from .clone_artifacts import (
    SOURCE_ARRAYS,
    log_clones,
    plan_clones,
    write_clones,
)
from .clone_steps import (
    artifact_cell_ids,
    files_text,
    plan_artifact_cells,
    read_codes,
    text_file_write_bytes,
)

if TYPE_CHECKING:
    from ...readers.clone_tables import CloneLayout
    from .repertoire import _RepertoireOperationsMixin as _CloneOperationsBase
else:
    _CloneOperationsBase = object

#: The values of ``export_clones``'s ``format``.
EXPORT_FORMATS = ("cospar", "matrix")
#: The diagnostics of a clone export: the cells written.
CLONE_EXPORT_DIAGNOSTICS = ("cells",)


# Bytes of one parsed row of a clone file in its chunk: the headers of its
# barcode and clone label as Python strings and their places in the chunk's
# columns, and the barcodes, labels, probabilities and cells that join and
# assign it; and of each field of the file, which the parser skips. Traced on
# 200,000 rows of 2 and 5 fields, a row held 117 to 121 bytes beside one
# byte a character of ASCII text and four of text beyond U+FFFF, which each
# character of the longest sampled row is counted at.
_PARSED_ROW_BYTES = 128
_PARSED_FIELD_BYTES = 8
_PARSED_CHARACTER_BYTES = 4
# The rows of a chunk lie in [_MIN_CHUNK_ROWS, _MAX_CHUNK_ROWS].
_MIN_CHUNK_ROWS = 1_000
_MAX_CHUNK_ROWS = 1_000_000
# Bytes per cell of the tables of loaded clones beside its ID: the clone
# code, size and probability, the labels kept apart, and the pandas index;
# with the ID, about 220 bytes per cell, and 230 with the receptor types of
# clonotypes, as traced on 200,000 cells.
_TABLE_BYTES_PER_CELL = 250
# Bytes per cell of an export beside its ID: its label and its code, and the
# text of its row, beside what the text writer holds whatever its rows.
_EXPORT_BYTES_PER_CELL = 200
# Bytes of each clone that a figure labels: its ID as given and as text, its
# code, and its label, a string of up to four bytes per character, beside the
# junction that the repertoire operations read.
_FIGURE_LABEL_BYTES = 400
# What a refusal of comparing groups suggests beside a larger budget.
_FEWER_GROUPS = "compare fewer groups, with fewer group_by columns or a cell_selection"
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
# tables, 650 bytes for diversity as traced on 179,932 groups.
_SUMMARY_BYTES_PER_GROUP = 2_000
# Summarizing the groups' clone sizes holds the counts of each clone's cells
# in each group and, while it sorts them by group and size into one table, a
# key and its sorted copy of each, 48 bytes for each count as traced on
# 800,000 counts in four groups.
_SIZE_TABLE_BYTES_PER_COUNT = 56
# Each row of the table beyond the first of its group holds its group, size
# and count, and the temporaries of the measures summed over rows, 92 bytes as
# traced on one group of 4,000 distinct sizes.
_SIZE_TABLE_BYTES_PER_ROW = 128
# Rarefying a group holds, for each of its sample sizes, the size as passed,
# how its sample is reached and the whole sizes that it rarefies, 141 bytes as
# traced on one group of 10,000 one-cell clones at 5,000 sizes.
_RAREFACTION_BYTES_PER_SIZE = 192
# Each row of the rarefaction curves: its order, size, method, Hill number,
# coverage and group, and the table built from them, 121 bytes as traced on
# 2,000 groups at 40 sizes. The groups' labels stay in their own table.
_CURVE_BYTES_PER_ROW = 128
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
# Following clones holds, for each row of the table, its clone, group, cells
# and fraction, and a byte for its chosen clone and group. Pandas 2 copies
# the four columns into one block of each dtype as it builds the table:
# traced on pandas 2.2.3, 9,020 and 18,040 rows held 1,523,958 and 2,480,836
# bytes by 13 key columns and 1,103,557 and 2,060,527 by one, 106.1 bytes a
# row with that byte, against 80.1 on pandas 3. The groups' labels stay in
# their own table.
_TRACK_BYTES_PER_ROW = 106
# Each pair of a fate and a clone holds its fate, clone code and cells.
_BYTES_PER_FATE_PAIR = 24
# Counting the clones tested within strata holds, for each pair of a fate
# and a clone, its clone within its fate's stratum, the sorted copy and the
# flags of its distinct values: 23.3 bytes a pair as traced on 10,000 to
# 1,000,000 pairs.
_PART_BYTES_PER_PAIR = 30
# Finding each cell's stratum holds its code among the groups' labels, the
# code's place among the strata and its stratum, beside the flag of each
# label that no group holds, and while each column kept apart is coded its
# place among the column's labels: 25 bytes a cell of one column, as traced
# on 10,000 to 1,000,000 cells.
_STRATUM_BYTES_PER_CELL = 24
_STRATUM_BYTES_PER_COLUMN = 8
# Writing the fate bias builds, for each early cell, a reference to its
# dominant fate's name, its codes as int32, and the mask and arrays that pick
# the names, beside what the writer holds.
_BIAS_WRITE_BYTES_PER_CELL = 40
# The tables of loaded fate bias hold, for each early cell, its ID, clone,
# fate and late cells and the pandas index, for each of its fractions the
# value read and its column in the table of fractions, and for each row of
# the table of clones and fates its clone, fate, counts and tests: at most
# 113 bytes at their peak as traced on 1 to 3 million rows that also held one
# to three key columns, which the fates' own table now holds.
_BIAS_TABLE_BYTES_PER_CELL = 200
_BIAS_TABLE_BYTES_PER_FRACTION = 24
_BIAS_TABLE_BYTES_PER_ROW = 160


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
        nbytes: What reading it left charged to the ledger it was read
            within: the codes, the labels and the kept labels with their
            strings.
    """

    ref: ArtifactRef
    cell_selection: ArtifactRef
    codes: np.ndarray
    clone_ids: np.ndarray | None
    kept_apart: tuple[tuple[str, np.ndarray], ...]
    nbytes: int


def _file_namespace(file: BarcodeFile) -> str:
    """The text before a file's clone labels: its prefix, or its name and ``_``."""
    if file.barcode_prefix is not None:
        return file.barcode_prefix
    stem = Path(os.fspath(file.path)).name.split(".", 1)[0]
    if not stem:
        raise ValueError(
            f"The clone file {quoted_value(os.fspath(file.path))} has no name before its "
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
                f"The clone files {quoted_value(os.fspath(other.path))} and "
                f"{quoted_value(os.fspath(file.path))} share the namespace {quoted_value(namespace)} of "
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


def _column_label_bytes(array: Any) -> int:
    """What reading a label of a metadata column and making its text holds: its stored value, its row and its string."""
    itemsize = int(np.dtype(array.dtype).itemsize)
    if np.dtype(array.dtype).kind == "U":
        return itemsize + 8 + python_text_bytes(1, itemsize // 4, widest=0x10FFFF) + 8
    return itemsize + 8 + _converted_label_bytes(np.zeros(1, dtype=array.dtype))


def _converted_label_bytes(stored: np.ndarray) -> int:
    """What :func:`clone_label_texts` holds to make labels of ``stored``: a reference each, and for numbers their text.

    Numbers become int64 and then NumPy text of 21 characters before each
    becomes a string of at most 20; text keeps the strings read.
    """
    n_values = len(stored)
    references = object_array_bytes(n_values)
    if stored.dtype.kind in "biuf":
        return (
            references
            + n_values * (8 + 4 * 21)
            + python_text_bytes(n_values, 20 * n_values)
        )
    return references


def _chunk_rows(ledger: MemoryLedger, row_bytes: int) -> int:
    """Rows of a file to parse at once: as many as half of the ledger's room holds.

    The other half is left to what grows as the rows are read, such as the
    clone labels and the hashes of barcodes that match no cell; a chunk
    holds 1,000 rows at least and 1,000,000 at most.
    """
    room = ledger.room
    if room is None:
        return _MAX_CHUNK_ROWS
    return int(min(_MAX_CHUNK_ROWS, max(_MIN_CHUNK_ROWS, room // (2 * row_bytes))))


class _CloneOperationsMixin(_CloneOperationsBase):
    def _clone_selection_read(self, cell_selection: ArtifactRef) -> CellSelectionRead:
        """Check and count a cell selection, whose cells the caller reads within its ledger."""
        planned = plan_cell_selection_read(self.zw, cell_selection)
        if not planned.cells:
            raise ValueError("cell_selection selects no cell")
        return planned

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
        Barcodes that match no selected cell are counted by the cell ID they
        name in the artifact's diagnostics, which
        ``ds.artifacts.inspect(ref).diagnostics`` returns: ``fileBarcodes``,
        ``unmatchedBarcodes``, those that name no cell of the store,
        ``unselectedBarcodes``, those of cells of the store outside the
        selection, and up to five examples of each, which show the barcode
        beside the cell ID it derives, then ``fileRows``, the rows of the
        files, a matrix's cells. The clone writer adds ``cellsWithClone``
        and ``largestCloneShare``. Barcodes of no cell are named in a
        warning, which a reused import gives again. A file is identified by
        the SHA-256 of its bytes, and of a matrix's cell and clone lists, so a moved file
        reuses the saved clones, an edited one gives new clones, and the
        order of the files does not matter.

        Clone IDs are unique only within one tool's run, so with several
        files each file's labels start with its ``barcode_prefix``, or with
        its name without extensions and ``_``; two files with the same
        namespace raise ``ValueError``. A cell listed in two different clones
        raises ``ValueError``, and with ``multiple="drop"`` it is left
        without a clone and counted in the ``cellsInSeveralClones``
        diagnostic. A selected cell that no file lists has no clone.

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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        from ...readers.clone_tables import clone_layout

        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if multiple not in MULTIPLE_POLICIES:
            raise ValueError(
                f"multiple must be one of {', '.join(MULTIPLE_POLICIES)}, not "
                f"{quoted_value(multiple)}"
            )
        policy: MultiplePolicy = "drop" if multiple == "drop" else "error"
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
        selection = self._clone_selection_read(cell_selection)
        n_cells = selection.cells
        arguments = CloneImportArguments(
            cell_selection=cell_selection,
            sources=sources,
            multiple=policy,
            invalidate_cache=invalidate_cache,
        )
        with_probability = any(layout.probability is not None for layout in layouts)
        planned = plan_clones(
            self.zw,
            arguments,
            cell_selection,
            n_cells,
            with_probability=with_probability,
            required_attributes=("n_clones",),
        )
        warnings = unmatched_barcode_rules(files_text("clone", len(files)))
        if planned.reused:
            # Saved clones warn as they did when they were imported.
            warn_diagnostics(
                arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
            )
            return planned.ref
        ids = self.cells._get_array("ids")
        self._require_writable(arguments.operation)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Importing clones for {n_cells} selected cells",
            remedy="pass a smaller cell_selection",
            codec_threads=self.nthreads,
        )
        cell_index = read_cell_selection(
            ledger, self.zw, selection, item="the selected cells' positions"
        )
        # Only the selected cells' IDs are read, a sample's in a merged store.
        cell_ids = read_text_rows(
            ledger, ids, cell_index, item="the selected cells' IDs"
        )
        join = CellBarcodeJoin(cell_ids, ledger=ledger, store_ids=ids)
        assigner = CloneAssigner(cell_ids, multiple=policy, ledger=ledger)
        # Rows are parsed in chunks sized by the widest file's fields and its
        # longest sampled row, whose barcodes are at least about as long as
        # the cells' IDs, beside the cell IDs that a file's prefix, suffix or
        # pattern derives from each barcode.
        row_chars = max(text_width(ids), *(layout.row_chars for layout in layouts))
        row_bytes = (
            _PARSED_ROW_BYTES
            + _PARSED_FIELD_BYTES * max(layout.n_fields for layout in layouts)
            + _PARSED_CHARACTER_BYTES * row_chars
            + max(derived_id_bytes(file, 1, row_chars) for file in files)
        )
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
                ledger=ledger,
                row_bytes=row_bytes,
                row_chars=row_chars,
            )
        assignment = assigner.finish()
        write_clones(
            self.zw,
            planned,
            assignment,
            {"n_clones": len(assignment.clone_ids)},
            ledger=ledger,
            diagnostics={
                **join.diagnostics(),
                "fileRows": join.n_rows,
                "cellsInSeveralClones": assignment.n_cells_in_several_clones,
            },
        )
        warn_diagnostics(
            arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
        )
        log_clones(assignment, n_cells)
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
        ledger: MemoryLedger,
        row_bytes: int,
        row_chars: int,
    ) -> None:
        """Assign the cells of one clone file to its clones.

        The file's chunks hold as many rows of ``row_bytes`` bytes, each of
        ``row_chars`` characters, as half of the ledger's room when the file
        is planned, which leaves the other half to the labels and hashes that
        grow. A chunk of a table's text holds at most its rows' characters,
        so rows longer than those sampled make chunks of fewer rows. A chunk
        is held until the file is read, since the reader keeps each chunk
        until it parses the next.
        """
        from ...readers.clone_tables import (
            iter_clone_rows,
            matrix_cells,
            matrix_clone_ids,
            matrix_entries,
        )

        chunk_rows = _chunk_rows(ledger, row_bytes)
        # A row's characters and its line end, which the sampled rows leave out.
        chunk_chars = chunk_rows * (row_chars + 1)
        item = f"a chunk of {chunk_rows} rows of a clone file"
        chunk_bytes = chunk_rows * row_bytes
        ledger.hold(item, chunk_bytes)
        if not layout.is_matrix:
            for rows in iter_clone_rows(layout, chunk_rows, chunk_chars=chunk_chars):
                assigner.add(
                    join.cells(file, rows.barcodes),
                    _namespaced(rows.clones, namespace),
                    probabilities=rows.probabilities,
                    source=source,
                )
                del rows
            ledger.release(chunk_bytes)
            return
        # A matrix lists its cells first; each entry names a listed cell.
        listed: list[np.ndarray] = []
        for barcodes in matrix_cells(layout, chunk_rows, chunk_chars=chunk_chars):
            # Each listed cell's position among the selected cells.
            ledger.hold("the clone matrix's cells", 8 * len(barcodes))
            listed.append(join.cells(file, barcodes))
            del barcodes
        n_listed = sum(map(len, listed))
        ledger.peak("the clone matrix's cells", 8 * n_listed)
        cells = np.concatenate(listed) if listed else np.empty(0, dtype=np.int64)
        # The listed cells' positions, held as they were joined, are now
        # those of the one array.
        cell_bytes = 8 * n_listed
        del listed
        labels = _namespaced(matrix_clone_ids(layout), namespace)
        # The matrix's labels are held beside those of earlier files.
        label_bytes = ledger.hold(
            "the clone matrix's labels",
            object_array_bytes(len(labels)) + sum(map(sys.getsizeof, labels)),
        )
        for entries in matrix_entries(layout, chunk_rows):
            assigner.add(cells[entries.cells], labels[entries.clones], source=source)
            del entries
        # The assigner keeps the labels it was given, charged as it keeps
        # them, so the matrix's cells and labels go with the file.
        del cells, labels
        ledger.release(chunk_bytes + cell_bytes + label_bytes)

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
        The artifact records a stamp of the column's stored chunks and of its
        missing values, read without decoding them, so editing the column
        gives new clones. The labels are read in blocks of as many cells as
        half of what the memory budget leaves.

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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        if not isinstance(column, str):
            raise TypeError("column must be the name of a cell metadata column")
        if column not in self.cells.columns:
            raise ValueError(f"{quoted_value(column)} is not a cell metadata column")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        selection = self._clone_selection_read(cell_selection)
        n_cells = selection.cells
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Saving the clones of {quoted_value(column)} for {n_cells} cells",
            codec_threads=self.nthreads,
        )
        # The column is stamped by its stored chunks before anything is read,
        # so saved clones are reused without reading a label.
        ledger.peak(
            "a stamp of the column's stored chunks",
            metadata_column_stamp_bytes(self.cells, column),
        )
        arguments = CloneColumnArguments(
            cell_selection=cell_selection,
            labels_fingerprint=metadata_column_stamp(self.cells, column),
            column=column,
            invalidate_cache=invalidate_cache,
        )
        planned = plan_clones(
            self.zw,
            arguments,
            cell_selection,
            n_cells,
            with_probability=False,
            required_attributes=("n_clones",),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        cell_index = read_cell_selection(
            ledger, self.zw, selection, item="the selected cells' positions"
        )
        # Each cell has one label, so no cell is given two clones.
        assigner = CloneAssigner(cell_index, multiple="error", ledger=ledger)
        labels_array = self.cells._get_array(column)
        missing_array = metadata_missing_mask(self.cells, column)
        text = np.dtype(labels_array.dtype).kind == "U"
        # Each label's read and text, its missing flag, and its cell and what
        # the assigner works with while the block is assigned.
        row_bytes = (
            _column_label_bytes(labels_array)
            + (0 if missing_array is None else 1)
            + 8
            + CLONE_ASSIGNMENT_ROW_BYTES
        )
        start = 0
        while start < n_cells:
            # Each block takes as many labels as half of the ledger's room
            # holds when it is planned, beside what the labels before it left.
            block_rows = min(n_cells - start, _chunk_rows(ledger, row_bytes))
            rows = cell_index[start : start + block_rows]
            entry = ledger.charged
            item = f"a block of {block_rows} clone labels"
            stored = (read_text_rows if text else read_rows)(
                ledger, labels_array, rows, item=item
            )
            missing = (
                None
                if missing_array is None
                else read_rows(ledger, missing_array, rows, item=item)
            )
            # Each label as text, beside the strings that numbers become.
            ledger.hold(item, _converted_label_bytes(stored))
            block_bytes = ledger.charged - entry
            labels = clone_label_texts(stored, missing)
            del stored, missing
            # Each label's cell and what the assigner works with are held
            # while the block is assigned; the assigner charges what it
            # keeps of the block itself.
            assigning = ledger.hold(
                f"assigning a block of {len(rows)} clone labels",
                8 * len(rows) + clone_assignment_bytes(len(rows)),
            )
            assigner.add(np.arange(start, start + len(rows)), labels)
            del rows, labels
            ledger.release(block_bytes + assigning)
            start += block_rows
        assignment = assigner.finish()
        write_clones(
            self.zw,
            planned,
            assignment,
            {"n_clones": len(assignment.clone_ids)},
            ledger=ledger,
        )
        log_clones(assignment, n_cells)
        return planned.ref

    def _clone_group(self, clones: ArtifactRef) -> zarr.Group:
        """Return a complete clone artifact's group, of a kind the summaries read."""
        if not isinstance(clones, ArtifactRef):
            raise TypeError("clones must be an ArtifactRef")
        if clones.kind not in _CLONE_SOURCE_KINDS:
            raise ValueError(
                f"clones must be a {' or '.join(_CLONE_SOURCE_KINDS)} artifact, "
                f"not {quoted_value(clones.kind)}"
            )
        require_complete_artifact(self.zw, clones)
        return artifact_group(self.zw, clones)

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
        planned_cells = plan_artifact_cells(self.zw, clones)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Loading the clones of {n_cells} cells",
            codec_threads=self.nthreads,
        )
        # The tables are built beside the codes and kept labels read, the
        # clone labels and the cells' IDs as text, and a source's own arrays.
        source = self._read_clone_input(clones, ledger, with_ids=True)
        assert source.clone_ids is not None
        cell_ids = artifact_cell_ids(
            ledger, self.zw, self.cells._get_array("ids"), planned_cells
        )
        extra = {
            name: read_whole(ledger, group, name, item=f"each cell's {name}").astype(
                dtype, copy=False
            )
            for name, dtype in SOURCE_ARRAYS.items()
            if name in group
        }
        ledger.peak(f"the tables of {n_cells} cells", _TABLE_BYTES_PER_CELL * n_cells)
        # Every saved clone holds a cell, so the codes give the clone sizes.
        assigned = source.codes >= 0
        clone_sizes = np.bincount(
            source.codes[assigned], minlength=len(source.clone_ids)
        )
        clone_size = np.zeros(len(source.codes), dtype=np.int64)
        clone_size[assigned] = clone_sizes[source.codes[assigned]]
        cells, table = clone_frames(
            cell_ids,
            codes=source.codes,
            clone_size=clone_size,
            clone_ids=source.clone_ids,
            clone_sizes=clone_sizes,
            extra=extra,
            kept_apart=dict(source.kept_apart),
        )
        status = inspect_artifact(self.zw, clones)
        return Clones(
            cells=cells,
            clones=table,
            source=str(status.operation),
            kept_apart=tuple(name for name, _labels in source.kept_apart),
            diagnostics=status.diagnostics,
        )

    @validation_scoped
    def _clones_export_clones(
        self,
        clones: ArtifactRef,
        path: str | os.PathLike[str],
        *,
        format: str = "cospar",
    ) -> ExportSummary:
        """Write saved clones as a file that lineage tracing tools read.

        ``"cospar"`` writes CoSpar's clone table, a CSV of ``Cell_ID`` and
        ``Clone_ID`` with one row per cell with a clone, which CoSpar reads
        and ``import_clones`` reads as ``format="cospar"``. ``"matrix"``
        writes a cells by clones Matrix Market file of the cells with a
        clone, holding a 1 where a cell belongs to a clone, with
        ``cells.tsv`` and ``clones.tsv`` beside it that name its rows and
        columns, replacing files of those names; ``import_clones`` reads the
        matrix with its lists as ``format="matrix"``. Cell IDs are this
        store's, and a path ending in ``.gz`` is gzip-compressed. Clone
        probabilities are not written.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            path: The file to write.
            format: ``"cospar"`` or ``"matrix"``.

        Returns:
            The file written, its ``rows``, the cells written, and its
            ``diagnostics``: ``cells``, the cells written.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a ``clones`` or ``clonotypes``
                artifact, ``format`` is unknown, a ``"cospar"`` table would hold a
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
        from ...writers.clone_tables import write_clone_matrix, write_clone_table

        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path")
        if format not in EXPORT_FORMATS:
            raise ValueError(
                f"format must be one of {', '.join(EXPORT_FORMATS)}, not {quoted_value(format)}"
            )
        group = self._clone_group(clones)
        n_cells = int(as_zarr_array(group["codes"], name="codes").shape[0])
        planned_cells = plan_artifact_cells(self.zw, clones)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Exporting the clones of {n_cells} cells",
            codec_threads=self.nthreads,
        )
        # The file is written from the codes and kept labels read, and the
        # clone labels and the cells' IDs as text.
        source = self._read_clone_input(clones, ledger, with_ids=True)
        assert source.clone_ids is not None
        codes, clone_ids = source.codes, source.clone_ids
        if format == "cospar":
            # A clone table reads these labels back as cells without a clone,
            # since its reader strips each label before it compares it.
            lost = [
                label
                for label in clone_ids.tolist()
                if str(label).strip() in MISSING_TEXT
            ]
            if lost:
                raise ValueError(
                    "The clone labels "
                    + ", ".join(quoted_value(label) for label in lost[:5])
                    + " read as no clone in a clone table; export these clones "
                    "with format='matrix', whose clone list keeps them"
                )
        cell_ids = artifact_cell_ids(
            ledger, self.zw, self.cells._get_array("ids"), planned_cells
        )
        # Each cell's label and code as written, and its flag, beside the
        # writer's buffers.
        ledger.peak(
            f"writing the clones of {n_cells} cells",
            _EXPORT_BYTES_PER_CELL * n_cells
            + text_file_write_bytes(os.fspath(path).endswith(".gz")),
        )
        assigned = codes >= 0
        cell_ids = cell_ids[assigned]
        if format == "cospar":
            written = write_clone_table(path, cell_ids, clone_ids[codes[assigned]])
        else:
            written = write_clone_matrix(path, cell_ids, codes[assigned], clone_ids)
        return export_summary(
            "export_clones", path, written, {"cells": written}, CLONE_EXPORT_DIAGNOSTICS
        )

    def _clonotype_receptor_group(self, clonotypes: ArtifactRef) -> zarr.Group:
        """The group of saved clonotypes' receptors, whose ``values`` hold each cell's receptor type.

        Clonotypes label the cells of their receptors, in the same order, and
        the receptors' canonical values are the receptor types.

        Raises:
            KeyError: If the store holds no such receptors.
            RuntimeError: If the receptors are incomplete.
        """
        receptors = inspect_artifact(self.zw, clonotypes).input_ref("receptors")
        require_complete_artifact(self.zw, receptors)
        return artifact_group(self.zw, receptors)

    def _plan_measured_cells(
        self, clones: ArtifactRef, *selections: ArtifactRef | None
    ) -> None:
        """Check the clones' cell selection and each other selection given before the summary charges anything.

        A selection's identity check reads its flags outside the budget, so
        it runs once in the operation's scope, before the first charge; the
        reads of the cells that :meth:`_measured_clone_cells` charges reuse
        it.
        """
        clone_cells = inspect_artifact(self.zw, clones).input_ref("cell_selection")
        plan_cell_selection_read(self.zw, clone_cells)
        for selection in selections:
            if (
                isinstance(selection, ArtifactRef)
                and selection.kind == "cell_selection"
            ):
                plan_cell_selection_read(self.zw, selection)

    def _clone_ledger(self, clones: ArtifactRef, doing: str) -> MemoryLedger:
        """The ledger of a summary of a clone artifact's cells, which ``doing`` names with their number."""
        group = self._clone_group(clones)
        n_cells = int(as_zarr_array(group["codes"], name="codes").shape[0])
        return MemoryLedger(
            self.memoryBytes, f"{doing} of {n_cells} cells", codec_threads=self.nthreads
        )

    def _read_clone_input(
        self, clones: ArtifactRef, ledger: MemoryLedger, *, with_ids: bool = False
    ) -> _CloneInput:
        """Read a clone artifact as every clone summary reads it, within ``ledger``.

        Each cell's clone code is read as int64, and with ``with_ids`` the
        label of each clone as a Python string. Every kind of clone artifact
        that the summaries accept is read here, with the labels of the groups
        that its clones keep apart: clones keep none, and clonotypes keep each
        cell's receptor type apart, the canonical values of their receptors,
        so the summaries measure T cells and B cells apart. Each read is
        charged to ``ledger``, and what the input holds stays charged,
        ``nbytes`` of it.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a kind of clone artifact that the
                summaries read.
            KeyError: If the store holds no such artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the artifact or the receptors of the clonotypes
                are incomplete.
            MemoryBudgetError: If a read does not fit the ledger's budget.
        """
        group = self._clone_group(clones)
        # The artifact's record is read before anything is charged.
        cell_selection = inspect_artifact(self.zw, clones).input_ref("cell_selection")
        receptor_group = (
            self._clonotype_receptor_group(clones)
            if clones.kind == ClonotypeArguments.artifact_kind
            else None
        )
        entry = ledger.charged
        kept_apart: tuple[tuple[str, np.ndarray], ...] = ()
        if receptor_group is not None:
            # Each of the few receptor types becomes one shared label, so the
            # summaries code the cells' labels without a string per cell.
            receptor_types = read_whole_text(
                ledger, receptor_group, "values", item="each cell's receptor type"
            )
            codes, names = category_codes(receptor_types, ledger=ledger)
            # Each cell's code, while its label is picked, and its label.
            coded = ledger.hold(
                "each cell's receptor type",
                codes.nbytes + object_array_bytes(len(receptor_types)),
            )
            labels = np.asarray([*names, None], dtype=object)[codes]
            read_bytes = ledger.charged - entry - coded
            code_bytes = codes.nbytes
            del receptor_types, codes, names
            ledger.release(read_bytes + code_bytes)
            kept_apart = ((_RECEPTOR_TYPE_COLUMN, labels),)
            del labels
        # The clones' few labels are read before each cell's code, whose
        # larger chunk a codec thread may hold after the read: a read of
        # small chunks need not replace it.
        clone_ids = (
            read_whole_text(
                ledger, group, _CLONE_ID_ARRAYS[clones.kind], item="the clones' labels"
            )
            if with_ids
            else None
        )
        clone_codes = read_codes(ledger, group, "codes", item="each cell's clone")
        return _CloneInput(
            ref=clones,
            cell_selection=cell_selection,
            codes=clone_codes,
            clone_ids=clone_ids,
            kept_apart=kept_apart,
            nbytes=ledger.charged - entry,
        )

    def _clone_figure_nouns(self, clones: ArtifactRef) -> tuple[str, str]:
        """What figures call the clones of a clone artifact, plural and singular.

        Clones are ``"clones"``, one ``"clone"``, and clonotypes
        ``"clonotypes"``, or ``"clonotype clusters"`` when a distance metric
        linked their chains, as their definition logs them.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a clone artifact.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
        """
        self._clone_group(clones)
        if clones.kind != ClonotypeArguments.artifact_kind:
            return "clones", "clone"
        parameters = inspect_artifact(self.zw, clones).parameters or {}
        return clonotype_nouns(str(parameters["metric"]))

    def _clone_figure_labels(
        self, clones: ArtifactRef, ids: Sequence[str]
    ) -> np.ndarray:
        """The labels that one figure gives some clones of a clone artifact.

        A clone is labelled by its ID. A clonotype is labelled by the junction
        amino acids of its representative cell's ``VDJ_1`` chain, or of its
        ``VJ_1`` chain without a VDJ chain, such as ``CASSLGQAYEQYF``, since
        immunologists read clonotypes by their junctions; clonotypes defined
        by ``receptor_arms="VJ"`` by the ``VJ_1`` chain first. Where two of
        the clonotypes given share a junction, the label of each also shows
        its number, the rank in its ID, such as ``CAVA 1`` and ``CAVA 3`` for
        ``clonotype_1`` and ``clonotype_3``. The clonotypes' IDs are read to
        find those given, and the repertoire operations read the junctions.

        Args:
            clones: A clone artifact: ``clones`` or ``clonotypes``.
            ids: The IDs of the clones that one figure shows, such as
                ``clonotype_1``; an ID given twice is one clonotype.

        Returns:
            The label of each clone, in the order of ``ids``, as an object
            array.

        Raises:
            TypeError: If ``clones`` is not an ArtifactRef.
            ValueError: If it is not a clone artifact, an ID names no
                clonotype, or a representative has no junction to show.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
            MemoryError: If reading the clonotypes' IDs as text and a hash
                index of them, and 400 bytes for each clone given, or reading
                their representatives' junctions, a chunk of each array at a
                time, does not fit the memory budget.
        """
        group = self._clone_group(clones)
        names = np.asarray([str(name) for name in ids], dtype=object)
        if clones.kind != ClonotypeArguments.artifact_kind or not len(names):
            return names
        n_clonotypes = int(
            as_zarr_array(group["clonotype_ids"], name="clonotype_ids").shape[0]
        )
        ledger = MemoryLedger(
            self.memoryBytes,
            "Labelling clonotypes by their junctions",
            remedy="label fewer clonotypes",
            codec_threads=self.nthreads,
        )
        # Each clone given, as its ID, code and label, is held throughout.
        ledger.hold(
            f"the {len(names)} clonotypes labelled", _FIGURE_LABEL_BYTES * len(names)
        )
        # The IDs read as Python strings, and pandas' hash table of them,
        # which finds those given.
        entry = ledger.charged
        stored_ids = read_whole_text(
            ledger, group, "clonotype_ids", item=f"the IDs of {n_clonotypes} clonotypes"
        )
        ledger.peak(
            f"the index of the IDs of {n_clonotypes} clonotypes",
            hash_table_bytes(n_clonotypes) + 8 * len(names),
        )
        codes = np.asarray(
            pd.Index(stored_ids, dtype=object).get_indexer(names), dtype=np.int64
        )
        id_bytes = ledger.charged - entry
        del stored_ids
        ledger.release(id_bytes)
        if (codes < 0).any():
            unknown = names[codes < 0].tolist()
            raise ValueError(f"No clonotype is labelled {unknown[:5]}")
        junctions = self._representative_junctions(clones, codes, ledger=ledger)
        unnamed = np.asarray([junction is None for junction in junctions.tolist()])
        if unnamed.any():
            raise ValueError(
                f"The representative cells of the clonotypes "
                f"{names[unnamed].tolist()[:5]} have no junction amino-acid "
                "sequence in the VDJ_1 or VJ_1 chain that names them"
            )
        # A junction that two of the clonotypes share names each with its
        # number too.
        distinct = np.unique(codes, return_index=True)[1]
        counted = Counter(junctions[distinct].tolist())
        return np.asarray(
            [
                f"{junction} {code + 1}" if counted[junction] > 1 else junction
                for junction, code in zip(
                    junctions.tolist(), codes.tolist(), strict=True
                )
            ],
            dtype=object,
        )

    def _figure_groups(
        self,
        artifact: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        kept_apart: Sequence[tuple[str, np.ndarray]] = (),
    ) -> CellGroups:
        """Group the cells of a cell-aligned artifact for one figure, as the summaries do.

        The groups are those of :func:`cell_groups` over every cell of the
        artifact: the combinations of the labels of the ``group_by`` columns,
        of the ``groups`` artifact and of ``kept_apart``, such as the receptor
        types that clonotypes keep apart, in the natural order of the labels.
        A cell without a label has no group, and so does a cell that a label
        artifact of fewer cells lacks.

        Args:
            artifact: A cell-aligned artifact, such as a ``clones`` artifact.
            group_by: Cell metadata columns, or None.
            groups: A cell-label artifact, or None.
            kept_apart: Key columns whose labels the artifact keeps apart, each
                with every artifact cell's label.

        Returns:
            The groups' key columns and labels, and each cell's group.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a ``group_by`` column is missing, named twice or
                kept apart already, two key columns share a name, or two
                groups get one name.
            KeyError: If the store holds no such artifact.
            MemoryError: If grouping the cells does not fit the memory budget.
        """
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")
        selection = inspect_artifact(self.zw, artifact).input_ref("cell_selection")
        n_cells = int(validate_cell_selection(self.zw, selection).selected_count)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Grouping the {n_cells} cells of a figure",
            codec_threads=self.nthreads,
        )
        ledger.hold(
            "the labels that the clones keep apart",
            sum(np.asarray(labels).nbytes for _name, labels in kept_apart),
        )
        # Every cell may join a group, a flag each while they are grouped.
        rows = ledger.hold("the cells to group", n_cells)
        grouped = cell_groups(
            self.zw,
            self.cells,
            selection,
            np.ones(n_cells, dtype=bool),
            group_by=group_by,
            groups=groups,
            cell_selection=None,
            ledger=ledger,
            kept_apart=tuple(kept_apart),
        )
        ledger.release(rows)
        return grouped

    def _measured_clone_cells(
        self,
        source: _CloneInput,
        cell_selection: ArtifactRef | None,
        ledger: MemoryLedger,
    ) -> tuple[ArtifactRef, np.ndarray, tuple[tuple[str, np.ndarray], ...]]:
        """The cells that a saved summary labels, with their clones.

        The cells are those of ``cell_selection``, or the clones' cells
        without one. A selected cell that the clones do not hold has no
        clone. Matching other cells reads both selections and an index of the
        clones' cells within ``ledger``, which keeps each measured cell's
        clone and kept labels charged.

        Returns:
            The cells' selection, each cell's clone code, -1 for none, and
            the labels that the clones keep apart, aligned to the cells.

        Raises:
            MemoryBudgetError: If matching the clones' cells to other cells
                does not fit the ledger's budget.
        """
        if cell_selection is None or cell_selection == source.cell_selection:
            return source.cell_selection, source.codes, source.kept_apart
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef or None")
        if cell_selection.kind != "cell_selection":
            raise ValueError(
                "cell_selection must be a cell_selection artifact, not "
                f"{quoted_value(cell_selection.kind)}"
            )
        measured = read_cell_selection(
            ledger,
            self.zw,
            self._clone_selection_read(cell_selection),
            item="the measured cells' positions",
        )
        held = read_cell_selection(
            ledger,
            self.zw,
            plan_cell_selection_read(self.zw, source.cell_selection),
            item="the positions of the clones' cells",
        )
        n_measured = len(measured)
        # Each measured cell's place among the clones' cells, found through
        # pandas' hash table of their positions and its copy of them, as
        # traced on 48 to 1,000,000 positions, beside the objects of its
        # index; and each measured cell's clone and labels.
        ledger.hold(
            "each measured cell's clone and the labels its clone keeps apart",
            (9 + 8 * len(source.kept_apart)) * n_measured,
        )
        ledger.peak(
            "matching the clones' cells to the measured cells",
            hash_table_bytes(len(held))
            + 8 * len(held)
            + 8 * n_measured
            + NUMPY_STEP_BYTES,
        )
        at = pd.Index(held).get_indexer(measured)
        found = at >= 0
        codes = np.full(n_measured, -1, dtype=np.int64)
        codes[found] = source.codes[at[found]]
        kept = []
        for name, labels in source.kept_apart:
            aligned = np.full(n_measured, None, dtype=object)
            aligned[found] = np.asarray(labels, dtype=object)[at[found]]
            kept.append((name, aligned))
        positions = measured.nbytes + held.nbytes + n_measured
        del at, found, measured, held
        ledger.release(positions)
        return cell_selection, codes, tuple(kept)

    def _clone_pair_counts(
        self,
        source: _CloneInput,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        cell_selection: ArtifactRef | None,
        ledger: MemoryLedger,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        """Group the cells with a clone and count each clone's cells in each group.

        The groups are those of :func:`cell_groups`, among the cells with a
        clone, with the labels that the clones keep apart, grouped within
        ``ledger``, which holds ``source`` as it was read.

        Returns:
            The groups' keys, one row per group in the natural order of the
            labels, indexed by its name and with its labels in the key
            columns; for each distinct pair of group and clone, sorted by
            group and clone code, its group, its clone code and its cells;
            and the grouping's diagnostics.
        """
        # The cells with a clone may join a group, a flag each while they
        # are grouped.
        rows = ledger.hold("the cells with a clone", len(source.codes))
        grouped = cell_groups(
            self.zw,
            self.cells,
            source.cell_selection,
            source.codes >= 0,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            ledger=ledger,
            kept_apart=source.kept_apart,
        )
        ledger.release(rows)
        # Each counted cell's group and clone, then their pairs: at most one
        # a cell, and at most one for each group and clone.
        flags = ledger.hold("the cells with a group", len(grouped.codes))
        counted = grouped.codes >= 0
        n_counted = int(np.count_nonzero(counted))
        codes = ledger.hold("each counted cell's group and clone", 16 * n_counted)
        group_codes = grouped.codes[counted]
        clone_codes = source.codes[counted]
        del counted
        ledger.release(flags)
        n_pairs = (
            min(n_counted, (int(group_codes.max()) + 1) * (int(clone_codes.max()) + 1))
            if n_counted
            else 0
        )
        ledger.peak(
            "counting each pair of group and clone",
            pair_count_bytes(n_counted, n_pairs),
        )
        pair_groups, pair_codes, pair_cells = pair_counts(group_codes, clone_codes)
        # The pairs are held before the codes they were counted from go.
        ledger.hold("each pair's group, clone and cells", 24 * len(pair_cells))
        del group_codes, clone_codes
        ledger.release(codes)
        keys = group_key_table(grouped.key_columns, grouped.keys)
        return keys, pair_groups, pair_codes, pair_cells, grouped.diagnostics

    def _grouped_clone_sizes(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        cell_selection: ArtifactRef | None,
        ledger: MemoryLedger | None = None,
    ) -> tuple[pd.DataFrame, GroupedCloneSizes, dict[str, int], MemoryLedger]:
        """Group the cells with a clone and summarize each group's clone sizes.

        The summary is charged to ``ledger``, which may hold the caller's
        arguments already, or to a ledger of its own.

        Returns:
            The groups' keys, as :meth:`_clone_pair_counts` gives them, each
            group's clone sizes, the grouping's diagnostics and the ledger
            that holds them.

        Raises:
            ValueError: If no cell has both a clone and a group.
        """
        if ledger is None:
            ledger = self._clone_ledger(clones, "Summarizing the clone sizes")
        source = self._read_clone_input(clones, ledger)
        keys, pair_groups, _codes, pair_cells, grouping = self._clone_pair_counts(
            source,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            ledger=ledger,
        )

        def check(rows: int) -> None:
            # Each group's rows, and while the table is built each count and
            # each row beyond a group's first.
            ledger.peak(
                f"summarizing the clones of {len(keys)} groups",
                _SUMMARY_BYTES_PER_GROUP * len(keys)
                + _SIZE_TABLE_BYTES_PER_COUNT * len(pair_cells)
                + _SIZE_TABLE_BYTES_PER_ROW * max(rows - len(keys), 0),
                remedy="group the cells by fewer labels",
            )

        # Each group's table holds a row at least, and its further rows are
        # counted once the table is built.
        check(len(keys))
        clone_sizes = GroupedCloneSizes.from_clones(pair_groups, pair_cells, len(keys))
        check(len(clone_sizes.values))
        if not clone_sizes.n.any():
            raise ValueError("No cell has both a clone and a group")
        return keys, clone_sizes, grouping, ledger

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
        arrays. Its diagnostics, which ``ds.artifacts.inspect(expansion)``
        returns, count the cells that it leaves out: ``cellsWithoutClone``,
        ``cellsWithoutGroup``, the cells with a clone that no group holds, and
        the ``groups``.

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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
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
        self._clone_group(clones)
        self._plan_measured_cells(clones, cell_selection)
        ledger = self._clone_ledger(clones, "Measuring the clonal expansion")
        source = self._read_clone_input(clones, ledger)
        measured, codes, kept_apart = self._measured_clone_cells(
            source, cell_selection, ledger
        )
        # The clones' own codes are released once they follow the measured
        # cells.
        if codes is not source.codes:
            input_bytes = source.nbytes
            del source
            ledger.release(input_bytes)
        else:
            del source
        # The cells with a clone may join a group, a flag each while they
        # are grouped.
        rows = ledger.hold("the cells with a clone", len(codes))
        grouped = cell_groups(
            self.zw,
            self.cells,
            measured,
            codes >= 0,
            group_by=group_by,
            groups=groups,
            cell_selection=None,
            ledger=ledger,
            kept_apart=kept_apart,
        )
        ledger.release(rows)
        if not grouped.keys:
            raise ValueError("No cell has both a clone and a group")
        columns = group_columns(group_by)
        if columns:
            ledger.peak(
                "fingerprinting the groups' labels", grouped.fingerprint_bytes()
            )
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
            required_attributes=("labels", "n_groups", "key_columns"),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        ledger.peak(
            f"measuring the clonal expansion of {n_cells} cells",
            _EXPANSION_BYTES_PER_CELL * n_cells
            + _SUMMARY_BYTES_PER_GROUP * len(grouped.keys),
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
        arrays = {
            "values": values,
            "group_clone_size": sizes,
            "group_clone_frequency": clone_frequencies,
            "group_codes": grouped.codes.astype(np.int32),
        }
        if grouped.key_columns:
            # Each group's label in each key column, group by group, as an
            # array, so opening the artifact reads none of them.
            arrays["group_labels"] = np.asarray(
                [key[column] for key in grouped.keys for column in grouped.key_columns],
                dtype=object,
            )
        # The arrays written, and each cell's category and flag, are held
        # while they are written.
        ledger.hold(
            f"the clonal expansion of {n_cells} cells as it is written",
            sum(array.nbytes for array in arrays.values())
            + categories.nbytes
            + counted.nbytes,
        )
        ledger.peak(
            "writing the clonal expansion",
            cell_data_write_bytes(
                {
                    name: (
                        array.shape,
                        f"U{max(1, max(map(len, array.tolist()), default=1))}"
                        if array.dtype == object
                        else array.dtype,
                    )
                    for name, array in arrays.items()
                }
            ),
        )
        write_cell_data_artifact(
            self.zw,
            planned,
            arrays,
            attributes={
                "labels": list(labels),
                "n_groups": len(grouped.keys),
                "key_columns": list(grouped.key_columns),
            },
            diagnostics=grouped.diagnostics,
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
            category, and one per group and rank window, each naming its
            group, and the groups' labels in ``group_keys``.

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
                f"expansion must be a clonal_expansion artifact, not {quoted_value(expansion.kind)}"
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
        n_cells = int(as_zarr_array(group["group_codes"], name="group_codes").shape[0])
        n_groups = int(cast(int, group.attrs["n_groups"]))
        self._clone_group(clones)
        self._plan_measured_cells(clones, measured)
        planned_cells = plan_artifact_cells(self.zw, expansion)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Loading the clonal expansion of {n_cells} cells",
            codec_threads=self.nthreads,
        )
        source = self._read_clone_input(clones, ledger, with_ids=True)
        _cells, codes, _kept = self._measured_clone_cells(source, measured, ledger)
        labels = tuple(str(name) for name in cast(list[str], group.attrs["labels"]))
        key_columns = tuple(
            str(column) for column in cast(list[str], group.attrs["key_columns"])
        )
        stored_labels = (
            read_whole_text(ledger, group, "group_labels", item="the groups' labels")
            if key_columns
            else np.empty(0, dtype=object)
        )
        group_keys = (
            [
                dict(zip(key_columns, (str(label) for label in row), strict=True))
                for row in stored_labels.reshape(n_groups, len(key_columns)).tolist()
            ]
            if key_columns
            else [{} for _group in range(n_groups)]
        )
        del stored_labels
        keys = group_key_table(key_columns, group_keys)
        cell_ids = artifact_cell_ids(
            ledger, self.zw, self.cells._get_array("ids"), planned_cells
        )
        # The tables of the cells and each group's rows.
        ledger.peak(
            f"the tables of the clonal expansion of {n_cells} cells",
            _EXPANSION_TABLE_BYTES_PER_CELL * n_cells
            + _SUMMARY_BYTES_PER_GROUP * n_groups,
        )
        counts = parameters["breakpoints"]
        frequencies = parameters["frequency_breakpoints"]
        sizes = read_array(group, "group_clone_size")
        clone_frequencies = read_array(group, "group_clone_frequency")
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
            cell_ids,
            categories=categories,
            labels=labels,
            clones=codes,
            clone_ids=source.clone_ids,
            groups=read_array(group, "group_codes"),
            group_names=keys.index.tolist(),
            sizes=sizes,
            frequencies=clone_frequencies,
            splits=bounds,
        )
        return ClonalExpansion(
            cells=cells,
            categories=table,
            rank_windows=ranks,
            group_keys=keys,
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
            diagnostics=status.diagnostics,
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
            standardized, and the groups' labels in ``group_keys``.

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
        keys, clone_sizes, grouping, ledger = self._grouped_clone_sizes(
            clones,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
        )
        table, hill, common, level = diversity_tables(
            clone_sizes, keys.index, size=size, coverage=coverage
        )
        return ClonalDiversity(
            indices=table,
            hill=hill,
            group_keys=keys,
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            size=common,
            coverage=level,
            diagnostics=grouping,
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
            The curves, one row per group, order and size, and the groups'
            labels in ``group_keys``.

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
        ledger = self._clone_ledger(clones, "Rarefying the clone sizes")
        if listed_sizes is not None:
            # The sizes asked for are held as the Python integers they are.
            ledger.hold(
                f"the {len(listed_sizes)} sizes asked for",
                object_array_bytes(len(listed_sizes))
                + sum(map(sys.getsizeof, listed_sizes)),
            )
        keys, clone_sizes, grouping, ledger = self._grouped_clone_sizes(
            clones,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            ledger=ledger,
        )
        # Each group is rarefied in turn to at most this many sizes, and the
        # curves hold a row per group, order and size, with the group's name.
        n_sizes = knots if listed_sizes is None else len(listed_sizes)
        n_rows = len(clone_sizes) * len(chosen) * n_sizes
        ledger.peak(
            f"rarefying {len(clone_sizes)} groups to {n_sizes} sizes",
            _RAREFACTION_BYTES_PER_SIZE * n_sizes + _CURVE_BYTES_PER_ROW * n_rows,
            remedy="pass fewer sizes or knots",
        )
        return Rarefaction(
            curves=rarefaction_curves(
                clone_sizes,
                keys.index,
                chosen,
                sizes=listed_sizes,
                endpoint=endpoint,
                knots=knots,
            ),
            group_keys=keys,
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            orders=chosen,
            diagnostics=grouping,
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
            The groups' cells and clones, the indices of every pair, and the
            groups' labels in ``group_keys``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, no cell has
                both a clone and a group, fewer than two groups hold a cell
                with a clone, no two groups share the labels that the clones
                keep apart, a column of the groups is missing or named twice,
                two key columns share a name, or two groups get one name.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            MemoryError: If reading the clones, grouping the cells, or the
                pairs of groups do not fit the memory budget.
        """
        ledger = self._clone_ledger(clones, "Comparing the groups of the clones")
        source = self._read_clone_input(clones, ledger)
        keys, pair_groups, pair_codes, pair_cells, grouping = self._clone_pair_counts(
            source,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            ledger=ledger,
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
        # The pairs, and beside the counts held, their entries in the
        # matrices that compare the groups and each group's rows.
        ledger.peak(
            f"comparing {len(keys)} groups, {n_pairs} pairs",
            _OVERLAP_BYTES_PER_PAIR * n_pairs
            + _OVERLAP_BYTES_PER_PRODUCT * largest**2
            + _BYTES_PER_CLONE_COUNT * len(pair_cells)
            + _SUMMARY_BYTES_PER_GROUP * len(keys),
            remedy=_FEWER_GROUPS,
        )
        pairs = overlap_pairs(
            pair_groups, pair_codes, pair_cells, len(keys), strata=strata
        )
        names = keys.index.to_numpy(dtype=object)
        pairs.insert(0, "group_1", names[pairs.pop("first").to_numpy()])
        pairs.insert(1, "group_2", names[pairs.pop("second").to_numpy()])
        return CloneOverlap(
            group_sizes=_group_sizes(keys, pair_groups, pair_cells),
            pairs=pairs,
            group_keys=keys,
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            diagnostics=grouping,
        )

    @validation_scoped
    def _clones_track(
        self,
        clones: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
        top_n: int = 10,
        labels: Sequence[str] | None = None,
    ) -> CloneTracking:
        """Follow chosen clones across groups of cells.

        By default the clones are the ``top_n`` with the most cells in each
        group, together, as scRepertoire's ``clonalCompare`` chooses them,
        with ties going to the larger clone overall; ``labels`` names them
        instead, such as ``["clone_3"]``. Each chosen clone gets its cells and share
        of cells in every group, zeros included, so a clone that expands
        between time points or spreads between cell types shows it. The
        groups are those of
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.overlap`, and a
        clonotype is followed through the groups of its own receptor type.
        A clonotype that spans receptor types, as
        ``within_receptor_type=False`` allows, is followed once in each, its
        cells of each type as a clonotype of that type, as every summary
        counts it: it is chosen in a type where it holds the most cells, and
        named in ``labels``, it is followed in each type of its cells.
        Nothing is saved.

        Args:
            clones: A clone artifact, such as ``clones`` or ``clonotypes``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell of the
                clones.
            top_n: The clones with the most cells to take from each group.
            labels: Clone labels to follow instead, as a sequence or a
                one-dimensional array, or None.

        Returns:
            One row per chosen clone and group in its ``table``, the chosen
            clones' labels, a clonotype's once for each receptor type it is
            followed in, and the groups' labels in ``group_keys``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, ``top_n`` is not
                positive, ``labels`` is empty or a label names no clone, no
                cell has both a clone and a group, a column of the groups is
                missing or named twice, two key columns share a name, or two
                groups get one name.
            KeyError: If the store holds no such ``clones`` artifact, or not the
                receptors of the clonotypes.
            RuntimeError: If the ``clones`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before it
                finished leaves them.
            MemoryError: If reading the clones, grouping the cells, or the
                table of the chosen clones' groups does not fit the memory
                budget.
        """
        if isinstance(top_n, bool) or not isinstance(top_n, int | np.integer):
            raise TypeError("top_n must be a positive integer")
        if top_n < 1:
            raise ValueError("top_n must be a positive integer")
        if labels is not None:
            listed = _listed(labels)
            if listed is None or not all(isinstance(label, str) for label in listed):
                raise TypeError("labels must be a sequence of clone labels or None")
            if not listed:
                raise ValueError("labels must name at least one clone, or be None")
            labels = listed
        ledger = self._clone_ledger(clones, "Following the clones")
        source = self._read_clone_input(clones, ledger, with_ids=True)
        keys, pair_groups, pair_codes, pair_cells, grouping = self._clone_pair_counts(
            source,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            ledger=ledger,
        )
        if len(pair_cells) == 0:
            raise ValueError("No cell has both a clone and a group")
        assert source.clone_ids is not None
        identifiers = source.clone_ids
        kept_strata = _kept_apart_strata(source, keys)
        strata = (
            np.zeros(len(keys), dtype=np.int64) if kept_strata is None else kept_strata
        )
        n_strata = int(strata.max()) + 1
        # A clone is followed once in each stratum, as its part there, so each
        # pair's part is coded by its clone and then its stratum; the code past
        # the last stratum stands for a named clone whose labels no group
        # holds, which is listed and followed through no group.
        radix = n_strata + 1
        ledger.peak(
            "coding each pair's clone within its stratum",
            _PART_BYTES_PER_PAIR * len(pair_codes) + NUMPY_STEP_BYTES,
        )
        pair_parts = pair_codes * radix + strata[pair_groups]
        ledger.hold("each pair's clone within its stratum", pair_parts.nbytes)
        if labels is None:
            chosen = top_clones(
                pair_groups, pair_parts, pair_cells, len(keys), int(top_n)
            )
        else:
            positions = pd.Index(identifiers).get_indexer(list(labels))
            unknown = [
                label for label, at in zip(labels, positions, strict=True) if at < 0
            ]
            if unknown:
                raise ValueError(f"No clone is labelled {unknown[:5]}")
            chosen = _named_parts(
                source,
                keys,
                strata,
                sorted_unique(positions.astype(np.int64)),
                radix,
                ledger,
            )
        fallback = chosen % radix
        fallback[fallback == n_strata] = -1
        reach = tracking_reach(pair_groups, pair_parts, chosen, strata, fallback)
        n_rows = int((reach * np.bincount(strata, minlength=reach.shape[1])).sum())
        # The rows, each with its clone, group, cells and fraction, and beside
        # the counts held, their entries in the matrices that follow the
        # groups and each group's rows.
        ledger.peak(
            f"following {len(chosen)} clones through {len(keys)} groups, {n_rows} rows",
            n_rows * _TRACK_BYTES_PER_ROW
            + reach.shape[0] * len(keys)
            + _BYTES_PER_CLONE_COUNT * len(pair_cells)
            + _SUMMARY_BYTES_PER_GROUP * len(keys),
            remedy="follow fewer clones, with a smaller top_n or labels, or fewer groups",
        )
        followed, rows, counts = tracking_cells(
            pair_groups, pair_parts, pair_cells, chosen, strata, reach
        )
        tracked = np.asarray(identifiers[chosen // radix].tolist(), dtype=object)
        # Without rows, the table keeps its columns and their types; clone
        # labels are text of the groups' names' type, which pandas 3 stores
        # as its string type and pandas 2 as Python objects, never fixed-width.
        sizes = _group_sizes(keys, pair_groups, pair_cells)
        named = keys.index.take(rows)
        table = pd.DataFrame(
            {
                "clone": pd.array(tracked[followed], dtype=named.dtype),
                "group": named,
                "n_cells": counts,
                "fraction": counts / sizes["n_cells"].to_numpy()[rows],
            }
        )
        return CloneTracking(
            table=table,
            group_keys=keys,
            clones=clones,
            tracked=tuple(str(label) for label in tracked.tolist()),
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            diagnostics={**grouping, "trackedClones": len(tracked)},
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
        n_permutations: int = 1000,
        random_seed: int = 0,
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
        a clone are shuffled ``n_permutations`` times, which keeps each clone's
        cells and each group's cells, and the coupling measured again. The
        z-score is the value's distance from the null mean in null standard
        deviations, so groups of different sizes compare. Permutation ``k``
        draws from a random stream keyed by ``random_seed`` and ``k``, so the result
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
            n_permutations: The permutations of the null, 0 for none, or at
                least 2 to estimate its spread.
            random_seed: The seed of the permutations, a non-negative integer
                below 2**64.

        Returns:
            The groups' cells and clones, each pair's coupling with its null
            and z-score, the groups' order, and the groups' labels in
            ``group_keys``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, ``method`` is
                unknown, ``n_permutations`` is 1 or negative, ``random_seed`` is
                negative or 2**64 or more, no cell has both a clone and a
                group, fewer than two groups hold a cell with a clone, no two
                groups share the labels that the clones keep apart, a column
                of the groups is missing or named twice, two key columns share
                a name, two groups get one name, or
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
                f"method must be one of {', '.join(COUPLING_METHODS)}, not {quoted_value(method)}"
            )
        if isinstance(n_permutations, bool) or not isinstance(
            n_permutations, int | np.integer
        ):
            raise TypeError("n_permutations must be a non-negative integer")
        if n_permutations < 0 or n_permutations == 1:
            raise ValueError(
                "n_permutations must be 0, for no null, or at least 2, to estimate "
                "its spread"
            )
        if isinstance(random_seed, bool) or not isinstance(
            random_seed, int | np.integer
        ):
            raise TypeError("random_seed must be a non-negative integer")
        if not 0 <= random_seed < 2**64:
            raise ValueError("random_seed must be a non-negative integer below 2**64")
        ledger = self._clone_ledger(clones, "Measuring the fate coupling of the clones")
        source = self._read_clone_input(clones, ledger)
        keys, pair_groups, pair_codes, pair_cells, grouping = self._clone_pair_counts(
            source,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            ledger=ledger,
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
        # Beside the coupling and the counts held: their entries in the
        # matrices that compare the groups, and each group's rows.
        held = _BYTES_PER_CLONE_COUNT * len(
            pair_cells
        ) + _SUMMARY_BYTES_PER_GROUP * len(keys)
        room = ledger.room
        assert room is not None
        import numba

        workers, batch, needed = coupling_plan(
            len(keys),
            n_pairs,
            int(pair_cells.sum()),
            len(pair_cells),
            n_permutations=int(n_permutations),
            # More workers than Numba's threads would only hold scratch.
            nthreads=min(
                self.nthreads, int(getattr(numba.config, "NUMBA_NUM_THREADS"))
            ),
            available=max(0, room - held),
        )
        resident = ledger.held_bytes + held + needed
        ledger.peak(
            f"measuring the coupling of {len(keys)} groups, {n_pairs} pairs",
            held + needed,
            remedy=_FEWER_GROUPS,
        )
        from scipy.cluster.hierarchy import leaves_list

        pairs, tree = clone_coupling(
            pair_groups,
            pair_codes,
            pair_cells,
            len(keys),
            method=method,
            strata=strata,
            n_permutations=int(n_permutations),
            random_seed=int(random_seed),
            workers=workers,
            batch=batch,
        )
        names = keys.index.to_numpy(dtype=object)
        pairs.insert(0, "group_1", names[pairs.pop("first").to_numpy()])
        pairs.insert(1, "group_2", names[pairs.pop("second").to_numpy()])
        logger.info(
            f"Measured the {method} coupling of {n_pairs} pairs of {len(keys)} "
            f"groups against {int(n_permutations)} permutations on {workers} "
            f"threads, {batch} at once, holding about {resident} bytes"
        )
        return CloneCoupling(
            group_sizes=_group_sizes(keys, pair_groups, pair_cells),
            pairs=pairs,
            order=tuple(str(name) for name in names[leaves_list(tree)].tolist()),
            linkage=tree,
            group_keys=keys,
            method=method,
            n_permutations=int(n_permutations),
            random_seed=int(random_seed),
            clones=clones,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            diagnostics={
                **grouping,
                # Without permutations there is no null to spread.
                "pairsWithoutNullSpread": (
                    int(np.count_nonzero(pairs["null_sd"].to_numpy() == 0))
                    if int(n_permutations)
                    else None
                ),
            },
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
        clonotype. A clonotype that spans receptor types, as
        ``within_receptor_type=False`` allows, is tested once in each, as every
        summary counts it: its cells of each type are a clone of that type,
        whose early cells are followed to its late cells of that type.

        The artifact's canonical ``values`` hold each early cell's dominant
        fate, empty for none, so ``ds.select_cells(bias, include=["Neutrophil"])``
        keeps the early cells whose clones went mostly to neutrophils.
        ``codes``, each cell's clone, ``fate_codes``, its dominant fate's
        position, ``n_late_cells``, its clone's late cells, and ``fractions``,
        one column per fate, are its other per-cell arrays. Its diagnostics,
        which ``ds.artifacts.inspect(bias).diagnostics`` returns, count the
        cells that it leaves out: ``cellsWithoutClone``, the late cells
        without a clone, ``cellsWithoutGroup``, those with a clone but no
        fate, the fates in ``groups``, ``earlyCellsWithoutLateCells``, the
        early cells with a clone whose clone has no late cell with a fate,
        ``earlyCellsWithTiedFates``, those whose clone's late cells tie
        between fates, and ``testedClones``, the clones tested, a clonotype
        once in each receptor type that it spans.

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
            or plot it with ``ds.plots.clone_fate_bias``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``clones`` is not a clone artifact, ``early`` or
                ``late`` is not a cell selection or selects no cell, a cell is
                both early and late, the clones give no late cell a clone, no
                late cell with a clone has a fate, fewer than two fates hold a
                late cell, a column of
                the fates is missing or named twice, two key columns share a
                name, or two fates get one name.
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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        for name, selection in (("early", early), ("late", late)):
            if not isinstance(selection, ArtifactRef):
                raise TypeError(f"{name} must be an ArtifactRef")
            if selection.kind != "cell_selection":
                raise ValueError(
                    f"{name} must be a cell_selection artifact, not {quoted_value(selection.kind)}"
                )
        planned_early = plan_cell_selection_read(self.zw, early)
        planned_late = plan_cell_selection_read(self.zw, late)
        n_early, n_late = planned_early.cells, planned_late.cells
        for name, count in (("early", n_early), ("late", n_late)):
            if count == 0:
                raise ValueError(f"{name} selects no cell")
        self._clone_group(clones)
        self._plan_measured_cells(clones)
        ledger = self._clone_ledger(clones, "Measuring the fate bias of the clones")
        source = self._read_clone_input(clones, ledger)
        early_cells = read_cell_selection(
            ledger, self.zw, planned_early, item="the early cells' positions"
        )
        late_cells = read_cell_selection(
            ledger, self.zw, planned_late, item="the late cells' positions"
        )
        # Both selections' positions joined and sorted, and the flags of
        # neighbours that are equal.
        ledger.peak(
            f"checking that none of {n_early} early cells is among {n_late} late cells",
            9 * (n_early + n_late),
        )
        shared = np.intersect1d(early_cells, late_cells, assume_unique=True)
        positions = early_cells.nbytes + late_cells.nbytes
        del early_cells, late_cells
        ledger.release(positions)
        if len(shared):
            raise ValueError(
                f"early and late share {len(shared)} cells; pass cell selections "
                "that no cell is in both, such as two time points"
            )
        del shared
        _late, late_codes, late_kept = self._measured_clone_cells(source, late, ledger)
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
        del cleared
        # The clones' codes and their late cells' are held while the late
        # cells are grouped into fates; a late selection of the clones' own
        # cells shares their codes.
        if late_codes is source.codes:
            ledger.hold(
                "the labels that the late cells' clones keep apart",
                8 * len(late_codes) * len(late_kept),
            )
        # Every late cell may join a fate, a flag each while they are grouped.
        rows = ledger.hold("the late cells to group", len(late_codes))
        grouped = cell_groups(
            self.zw,
            self.cells,
            late,
            np.ones(len(late_codes), dtype=bool),
            group_by=group_by,
            groups=groups,
            cell_selection=None,
            ledger=ledger,
            kept_apart=late_kept,
        )
        ledger.release(rows)
        counted = (late_codes >= 0) & (grouped.codes >= 0)
        if not (late_codes >= 0).any():
            raise ValueError(
                f"None of the {len(late_codes)} cells of late has a clone; pass "
                "late cells that the clones give a clone"
            )
        if not counted.any():
            raise ValueError(
                "No cell has both a clone and a group among the late cells: "
                "group_by and groups label none of the late cells with a clone"
            )
        keys = group_key_table(grouped.key_columns, grouped.keys, index="fate")
        if len(keys) < 2:
            raise ValueError("Fate bias needs at least two fates among the late cells")
        names = keys.index.to_numpy(dtype=object)
        columns = group_columns(group_by)
        if columns:
            ledger.peak(
                "fingerprinting the groups' labels", grouped.fingerprint_bytes()
            )
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
                "n_late_cells": ((n_early,), "i"),
                "fractions": ((n_early, n_fates), "f"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("group_keys", "key_columns"),
            reuse_validator=has_clones,
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        # Counting each clone's late cells in each fate holds each counted
        # cell's fate and clone and their count, and each fate's cells.
        n_counted = int(np.count_nonzero(counted))
        ledger.peak(
            "counting each pair of fate and clone",
            16 * n_counted
            + pair_count_bytes(n_counted, n_counted)
            + 9 * len(grouped.codes)
            + 16 * n_fates,
        )
        pair_fates, pair_codes, pair_cells = pair_counts(
            grouped.codes[counted], late_codes[counted]
        )
        ledger.hold("each pair's fate, clone and cells", 24 * len(pair_cells))
        fate_cells = np.bincount(
            grouped.codes[grouped.codes >= 0], minlength=n_fates
        ).astype(np.int64)
        strata = _kept_apart_strata(source, keys)
        # The late cells are grouped whether they have a clone or not, since
        # the tests draw from every late cell with a fate, so the cells that
        # the fates leave out are counted among those with a clone.
        ledger.peak(
            "counting the late cells that the fates leave out", 3 * len(late_codes)
        )
        late_with_clone = late_codes >= 0
        n_late_without_clone = int(np.count_nonzero(~late_with_clone))
        n_late_without_fate = int(
            np.count_nonzero(late_with_clone & (grouped.codes < 0))
        )
        del late_with_clone
        # Nothing per late cell is held beyond the counts.
        group_keys, key_columns = grouped.keys, list(grouped.key_columns)
        del grouped, late_codes, late_kept, counted
        # A clone is tested once in each stratum of its late cells, as its
        # part there, so the tests are the distinct clones within strata.
        if strata is None:
            n_tested = len(sorted_unique(pair_codes))
        else:
            ledger.peak(
                "counting the clones tested in each stratum",
                _PART_BYTES_PER_PAIR * len(pair_codes) + NUMPY_STEP_BYTES,
            )
            n_tested = len(
                sorted_unique(pair_codes * (int(strata.max()) + 1) + strata[pair_fates])
            )
        n_late_counted = int(pair_cells.sum())
        _early, early_codes, early_kept = self._measured_clone_cells(
            source, early, ledger
        )
        early_strata = None
        if strata is not None:
            # An early cell belongs to its clone's part in the stratum of its
            # own labels, whose late cells it is followed to.
            ledger.peak(
                "finding the stratum of each early cell",
                _cell_strata_bytes(len(early_codes), len(early_kept)),
            )
            early_strata = _cell_strata(keys, strata, early_kept)
            ledger.hold("the stratum of each early cell", early_strata.nbytes)
        del early_kept
        width = max(len(str(name)) for name in names.tolist())
        ledger.peak(
            f"measuring the fate bias of {n_early} early cells and {n_tested} "
            f"clones in {n_fates} fates",
            _BYTES_PER_FATE_PAIR * len(pair_cells)
            + fate_bias_bytes(n_early, n_tested, n_fates)
            + _fate_bias_write_bytes(n_early, n_tested, n_fates, width),
            remedy="group the late cells into fewer fates",
        )
        result = fate_bias(
            early_codes,
            pair_fates,
            pair_codes,
            pair_cells,
            fate_cells,
            strata,
            early_strata,
        )
        del early_strata
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
                "n_late_cells": result.late_cells,
                "fractions": result.fractions,
                "clone_codes": result.clones,
                "clone_early_cells": result.clone_early_cells,
                "clone_fate_cells": result.clone_cells,
                "clone_p_values": result.p_values,
                "clone_q_values": result.q_values,
                "fate_cells": result.fate_cells,
            },
            attributes={"group_keys": group_keys, "key_columns": key_columns},
            diagnostics={
                "cellsWithoutClone": n_late_without_clone,
                "cellsWithoutGroup": n_late_without_fate,
                "groups": n_fates,
                "earlyCellsWithoutLateCells": int(
                    np.count_nonzero((early_codes >= 0) & (result.late_cells == 0))
                ),
                "earlyCellsWithTiedFates": int(
                    np.count_nonzero((result.late_cells > 0) & (result.dominant < 0))
                ),
                "testedClones": n_tested,
            },
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
            The fate bias, with one row per early cell, its shares of the
            fates in a table of their own, one row per clone with late cells
            and fate of its stratum, one per fate, and the fates' labels in
            ``group_keys``.

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
                f"bias must be a clone_fate_bias artifact, not {quoted_value(bias.kind)}"
            )
        require_complete_artifact(self.zw, bias)
        group = artifact_group(self.zw, bias)
        declared = CELL_VALUE_NAMES[CloneFateBiasArguments.artifact_kind]
        missing = [
            name
            for name in (declared.name, *declared.alternatives)
            if name not in group
        ]
        if missing:
            raise ValueError(
                "bias was saved by an earlier version of CyteArc, without the "
                f"arrays {', '.join(missing)}; measure the fate bias again with "
                "ds.clones.fate_bias, which saves it anew"
            )
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
        self._clone_group(clones)
        planned_cells = plan_artifact_cells(self.zw, bias)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Loading the fate bias of {n_early} early cells",
            codec_threads=self.nthreads,
        )
        # The fates' labels, read with the artifact's attributes and copied as
        # a dictionary for each fate, are held while the tables are built.
        fate_labels = [label for key in group_keys for label in key.values()]
        ledger.hold(
            f"the labels of {n_fates} fates",
            2 * n_fates * GROUPING_BYTES_PER_GROUP
            + 2 * n_fates * len(key_columns) * GROUPING_BYTES_PER_GROUP_COLUMN
            + python_text_bytes(
                len(fate_labels),
                sum(map(len, fate_labels)),
                widest=max(
                    (ord(max(label)) for label in fate_labels if label), default=0
                ),
            ),
        )
        del fate_labels
        source = self._read_clone_input(clones, ledger, with_ids=True)
        assert source.clone_ids is not None
        # The early cells' IDs, read as text for the index of their table.
        cell_ids = artifact_cell_ids(
            ledger, self.zw, self.cells._get_array("ids"), planned_cells
        )
        ledger.peak(
            f"the tables of the fate bias of {n_early} early cells",
            n_early
            * (_BIAS_TABLE_BYTES_PER_CELL + _BIAS_TABLE_BYTES_PER_FRACTION * n_fates)
            + n_tested * n_fates * _BIAS_TABLE_BYTES_PER_ROW
            + _SUMMARY_BYTES_PER_GROUP * n_fates,
        )
        keys = group_key_table(key_columns, group_keys, index="fate")
        cells, fractions, clone_fates = fate_bias_tables(
            cell_ids,
            codes=read_array(group, "codes"),
            dominant=read_array(group, "fate_codes"),
            late_cells=read_array(group, "n_late_cells"),
            fractions=read_array(group, "fractions"),
            clone_ids=source.clone_ids,
            clones=read_array(group, "clone_codes"),
            clone_cells=read_array(group, "clone_fate_cells"),
            clone_early_cells=read_array(group, "clone_early_cells"),
            p_values=read_array(group, "clone_p_values"),
            q_values=read_array(group, "clone_q_values"),
            fates=keys.index,
        )
        fates = pd.DataFrame(
            {
                "fate": keys.index,
                "n_late_cells": read_array(group, "fate_cells").astype(np.int64),
            }
        )
        return CloneFateBias(
            cells=cells,
            fractions=fractions,
            clone_fates=clone_fates,
            fates=fates,
            group_keys=keys,
            clones=clones,
            early=status.input_ref("cell_selection"),
            late=status.input_ref("late"),
            group_by=tuple(str(column) for column in parameters["group_by"]),
            groups=groups,
            diagnostics=status.diagnostics,
        )


def _group_sizes(
    keys: pd.DataFrame, pair_groups: np.ndarray, pair_cells: np.ndarray
) -> pd.DataFrame:
    """One row per group of ``keys``: its name, cells with a clone and clones."""
    n_groups = len(keys)
    return pd.DataFrame(
        {
            "group": keys.index,
            "n_cells": np.bincount(
                pair_groups, weights=pair_cells, minlength=n_groups
            ).astype(np.int64),
            "n_clones": np.bincount(pair_groups, minlength=n_groups),
        }
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


def _fate_bias_write_bytes(
    n_early: int, n_tested: int, n_fates: int, width: int
) -> int:
    """What writing fate bias holds: the arrays it builds and what the writer holds.

    ``width`` is the characters of the longest fate name, at which each early
    cell's dominant fate is stored.
    """
    return _BIAS_WRITE_BYTES_PER_CELL * n_early + cell_data_write_bytes(
        {
            "values": ((n_early,), f"U{max(1, width)}"),
            "codes": ((n_early,), np.int32),
            "fate_codes": ((n_early,), np.int32),
            "late_cells": ((n_early,), np.int64),
            "fractions": ((n_early, n_fates), np.float64),
            "clone_codes": ((n_tested,), np.int64),
            "clone_early_cells": ((n_tested,), np.int64),
            "clone_fate_cells": ((n_tested, n_fates), np.int64),
            "clone_p_values": ((n_tested, n_fates), np.float64),
            "clone_q_values": ((n_tested, n_fates), np.float64),
            "fate_cells": ((n_fates,), np.int64),
        }
    )


def _cell_strata_bytes(n_cells: int, n_columns: int) -> int:
    """What :func:`_cell_strata` holds for ``n_cells`` cells of ``n_columns`` columns kept apart."""
    return (
        _STRATUM_BYTES_PER_CELL + _STRATUM_BYTES_PER_COLUMN * n_columns
    ) * n_cells + NUMPY_STEP_BYTES


def _cell_strata(
    keys: pd.DataFrame,
    strata: np.ndarray,
    kept: Sequence[tuple[str, np.ndarray]],
) -> np.ndarray:
    """Each cell's stratum: that of the groups whose labels kept apart are the cell's.

    A clone's cells of one stratum are its part in that stratum, so a clone
    that spans the labels kept apart, as clonotypes that
    ``within_receptor_type=False`` defines can, has a part in each.

    Args:
        keys: One row per group, with its labels in the key columns.
        strata: Each group's stratum, as :func:`_kept_apart_strata` gives it.
        kept: Each column that the clones keep apart, with each cell's label.

    Returns:
        Each cell's stratum, -1 for a cell without labels or with labels that
        no group holds.
    """
    n_cells = len(kept[0][1])
    cell_codes = np.zeros(n_cells, dtype=np.int64)
    unheld = np.zeros(n_cells, dtype=bool)
    group_codes = np.zeros(len(keys), dtype=np.int64)
    for name, labels in kept:
        # The cells' labels are coded among the groups' few labels; an explicit
        # object dtype keeps them the Python strings charged, whether or not
        # pandas stores text with pyarrow.
        groups_labels = keys[name].to_numpy(dtype=object)
        names = pd.Index(pd.unique(groups_labels), dtype=object)
        at = names.get_indexer(pd.Index(labels, dtype=object))
        unheld |= at < 0
        cell_codes *= len(names)
        cell_codes += at
        del at
        group_codes = group_codes * len(names) + names.get_indexer(groups_labels)
    # Each group's labels name its stratum, so each code has one.
    lookup = pd.Series(strata, index=group_codes).groupby(level=0).first()
    at = lookup.index.get_indexer(cell_codes)
    del cell_codes
    at[unheld] = -1
    del unheld
    found: np.ndarray = lookup.to_numpy(dtype=np.int64)[at]
    found[at < 0] = -1
    return found


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


def _named_parts(
    source: _CloneInput,
    keys: pd.DataFrame,
    strata: np.ndarray,
    named: np.ndarray,
    radix: int,
    ledger: MemoryLedger,
) -> np.ndarray:
    """The parts that the tracking of named clones follows, as codes of clone and stratum.

    A named clone is followed once in each stratum of its cells' labels kept
    apart, counted among its cells or not, so a clonotype of two receptor
    types is followed in each. A named clone whose labels no group holds is
    coded once past the last stratum, which no group's stratum reaches, so
    it is listed and followed through no group; clones that keep no labels
    apart have the one stratum, 0.

    Args:
        source: The clones, as the summaries read them.
        keys: One row per group, with its labels in the key columns.
        strata: Each group's stratum.
        named: The codes of the named clones, ascending.
        radix: The strata and one more, by which a part's code counts its
            clone.
        ledger: The tracking's ledger, which each step is charged to.

    Returns:
        The codes of the parts, ascending.
    """
    if not source.kept_apart:
        return named * radix
    unheld = radix - 1
    # Each cell's place among the named clones, the named clone found there
    # and whether it is the cell's, and then the positions of those cells.
    ledger.peak(
        f"finding the cells of {len(named)} named clones",
        17 * len(source.codes) + NUMPY_STEP_BYTES,
    )
    at = np.searchsorted(named, source.codes)
    np.minimum(at, len(named) - 1, out=at)
    cells = np.flatnonzero(named[at] == source.codes)
    del at
    n_columns = len(source.kept_apart)
    # The cells' labels and strata, and the parts they give with their sorted
    # copy and flags.
    ledger.peak(
        f"finding the strata of {len(cells)} cells of the named clones",
        (8 * n_columns + 48) * len(cells) + _cell_strata_bytes(len(cells), n_columns),
    )
    kept = [
        (name, np.asarray(labels, dtype=object)[cells])
        for name, labels in source.kept_apart
    ]
    cell_strata = _cell_strata(keys, strata, kept)
    del kept
    cell_strata[cell_strata < 0] = unheld
    parts = sorted_unique(source.codes[cells] * radix + cell_strata)
    del cells, cell_strata
    # A clone followed in a stratum is not listed again for its labels that
    # no group holds.
    clones = parts // radix
    unlisted = (parts % radix == unheld) & np.isin(
        clones, clones[parts % radix != unheld]
    )
    listed: np.ndarray = parts[~unlisted]
    return listed


def _listed(values: Any) -> list[Any] | None:
    """The items of a sequence or a one-dimensional array, or None for anything else."""
    if isinstance(values, np.ndarray):
        return values.tolist() if values.ndim == 1 else None
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        return None
    return list(values)
