import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...clones.allele_clones import (
    AlleleBank,
    AlleleObservations,
    allele_banks,
    allele_call_settings,
    allele_clone_calls,
)
from ...clones.assignments import CloneAssignment
from ...clones.barcode_clones import clone_labels
from ...clones.recorders import (
    FIELDS,
    MARK,
    STATES,
    UNKNOWN_MARK,
    UNKNOWN_POLICIES,
    LineageAlleles,
    allele_chunk,
    cell_summaries,
    character_matrix,
    character_name,
    empty_allele_chunk,
    lineage_allele_frames,
    records_in_line_order,
)
from ...metadata.arguments import AlleleCloneArguments, RecorderAlleleImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.records import RecordEncoder, RecordTable, pack_text, unpack_text
from ...metadata.rows import read_metadata_rows_chunkwise, text_read_bytes
from ...metadata.selection import cell_selection_indices
from ...storage.arrays import zarr_write_buffer_bytes
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.arrays import sorted_unique, unique_inverse
from ...utils.digest import file_sha256, path_sha256
from ...utils.logging import logger
from ...utils.warnings import warn
from .cell_groups import group_columns, within_groups
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    warn_unmatched_barcodes,
)
from .clone_artifacts import clone_write_bytes, plan_clones, write_clones
from .record_tables import (
    read_array,
    record_arrays,
    record_chunk_lengths,
    stored_records,
)

if TYPE_CHECKING:
    from ...readers.recorder_alleles import AlleleLayout
    from ..mapping_datastore import MappingDatastore as _AlleleOperationsBase
else:
    _AlleleOperationsBase = object

# The prefix of the stored record arrays, such as record_cells.
_RECORDS = "record"
# The per-cell arrays of a lineage_alleles artifact; n_edits is canonical.
_CELL_ARRAYS = ("n_edits", "n_integrations", "n_records")
# The attributes that an import saves and that reusing it requires.
_IMPORT_ATTRIBUTES = (
    "fields",
    "states",
    "files",
    "n_file_rows",
    "n_unmatched_rows",
    "n_beyond_length",
    "n_rows_without_integration",
    "n_file_barcodes",
    "n_unmatched_barcodes",
    "unmatched_barcode_examples",
)
#: The values of ``format`` of ``export_characters``.
EXPORT_FORMATS = ("cassiopeia", "laml")
# A chunk of a file may take a twentieth of the budget.
_CHUNK_SHARE = 20
# Bytes that the per-cell arrays and the arrays written take per selected cell.
_CELL_BYTES = 64
# The encoder's peak while it merges its chunks into one table, beside the
# text of their distinct values, as a multiple of the bytes of the chunks it
# holds: the chunks, the merged codes, their order and the table.
_MERGE_FACTOR = 2
# Bytes that merging takes for each distinct text value that a chunk holds:
# its Python string, its entry in the table that codes the text, and its
# rank.
_MERGE_BYTES_PER_TEXT = 150
# Beside the chunks' packed bytes, which the merge factor counts, merging
# holds for each byte of a chunk's distinct text its UTF-8 copy as each
# string caches it, and at the widest character's bytes per byte, each
# value's string, the values joined, and the encoder's buffer of the joined
# text, which takes up to four bytes per character of the widest strings.
_MERGED_TEXT_BYTES = 1
_MERGED_TEXT_COPIES = 3
# Bytes that writing takes per record beside the table: the int32 copies of
# its cells and files, the int8 copy of its states, and the keys that count
# each cell's integrations, 43 as traced over 6 million records. The chunks
# that the merge freed stay resident beside them, since the allocator keeps
# them for reuse: at 30 million records the resident peak of writing was
# twice the table's bytes and about as many bytes per record.
_WRITE_BYTES_PER_RECORD = 44
# Bytes that an import, a load and an export hold whatever their size: a
# block of a file as it is hashed, pandas' parser buffers, and a Zarr chunk of
# 2 MiB as it is filled and encoded.
_FIXED_BYTES = 8 * 2**20
# Bytes that a loaded record takes at the peak of building the tables: its
# stored arrays and its row of the table of records.
_TABLE_BYTES_PER_RECORD = 120
# Decoding a stored text field holds, for each byte stored, the bytes read
# and their copy as one buffer, and, at the widest character's bytes per
# stored byte, the buffer decoded as one text and each value's string.
_STORED_TEXT_COPIES = 2
_DECODED_TEXT_COPIES = 2
# The text fields of the records, which loads and exports decode.
_TEXT_FIELDS = ("integration", "site", "allele")
# Python holds a character in one byte up to U+00FF, two up to U+FFFF and
# four beyond, and UTF-8 writes it in at least as many bytes, from these lead
# bytes, so each stored byte decodes to at most that width.
_WIDE_LEAD_BYTES = ((0xF0, 4), (0xC4, 2))
# Bytes that a loaded distinct integration, site or allele, or a cell ID,
# takes as a Python string with its entry in the index of its categories.
_TABLE_BYTES_PER_TEXT = 160
# Bytes that the character matrix takes per record at its peak: the stored
# arrays, the order of the rows, their groups, the keys of the edits and the
# codes of each record.
_EXPORT_BYTES_PER_RECORD = 260
# Bytes that a distinct allele takes while it is ranked, beside its text as
# decoding holds it, which the counts of the saved import give, and a
# character while it is named and ordered.
_EXPORT_BYTES_PER_ALLELE = 150
_EXPORT_BYTES_PER_CHARACTER = 1_000
# Bytes that writing a store takes per exported cell: a character's column
# copied as int64 and Zarr's buffer of its chunk.
_STORE_BYTES_PER_CELL = 24
# Bytes that writing a store takes per byte of the alleles of its table of
# states: each value encoded as UTF-8 and the chunk's buffer of them.
_STORE_BYTES_PER_ALLELE_BYTE = 2
# The columns of the table of loaded clones, which a within column may not
# share.
_CLONE_COLUMNS = frozenset({"clone", "clone_size", "n_alleles"})
# Bytes of each row of a bank while it is checked: its allele as a Python
# string with its references, sorted copies and packed bytes for the
# fingerprint, and its frequency and samples with their copies, 73 as traced
# on 300,000 rows; of each row once checked: its string, reference,
# frequency and samples; and of each character of its text.
_BANK_ROW_BYTES = 128
_BANK_HELD_ROW_BYTES = 80
_BANK_CHARACTER_BYTES = 4
# Calling clones holds, for each cell, its group and whether it has an edit;
# while the states are read, each record's state, whether it is an edit and
# the comparison that adds unknown marks, 3 as traced; while the integrations
# and sites are read, both full int32 columns and each record's edit flag;
# and for each edit, its allele as int64, its integration and site as int32,
# their key as int64 and the sort that numbers the keys, 57 at their peak, or
# its cell as it is read beside the allele and character. It then holds each
# edit's cell, character and allele.
_CALL_CELL_BYTES = 16
_STATE_READ_BYTES = 4
_COLUMN_READ_BYTES = 8
_CALL_EDIT_BYTES = 80
_HELD_EDIT_BYTES = 24
# Bytes of each distinct allele, integration or site while its packed text is
# read and decoded, beside the copies of its bytes that decoding holds: its
# offset as read and in two lists of Python integers, its Python string and
# two references to it, 145 as traced for ASCII text; and once decoded, its
# string and its reference beside its characters, 49 as traced for ASCII
# text. A string beyond ASCII holds 16 bytes more and a terminator as wide as
# its characters, up to 4 bytes.
_DECODE_TEXT_BYTES = 168
_HELD_TEXT_BYTES = 68
# The records whose characters are keyed at a time, while both full columns
# of integrations and sites are held.
_KEY_BLOCK = 1 << 16
# Saving the clones holds, for each cell beside what saving them holds, its
# rare alleles from the calls as int64; each clone's label as a Python
# string is counted at its size.
_CALLED_CELL_BYTES = 8


def _files_text(n_files: int) -> str:
    return "the allele file" if n_files == 1 else "the allele files"


def _per_file(integrations: Any, n_files: int) -> list[Any]:
    """One item of ``integrations`` per listed file."""
    if integrations is None or isinstance(integrations, str | Mapping):
        return [integrations] * n_files
    if isinstance(integrations, Sequence) and not isinstance(integrations, bytes):
        items = list(integrations)
        if len(items) != n_files:
            raise ValueError(
                f"integrations lists {len(items)} values for {n_files} allele files; "
                "pass one per file, or one for every file"
            )
        for item in items:
            if item is not None and not isinstance(item, str | Mapping):
                raise TypeError(
                    "Each item of integrations must be an integration's name, a "
                    "mapping of integrations to their slots, or None"
                )
        return items
    raise TypeError(
        "integrations must be an integration's name, a mapping of integrations to "
        "their slots, a sequence of these with one per file, or None"
    )


def _saved_count(group: zarr.Group, name: str) -> int:
    """A count that a saved import records in its attributes."""
    value = group.attrs.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"The lineage_alleles artifact records no count {name!r}")
    return value


def _without_integration_warning(n_rows: int) -> str | None:
    if not n_rows:
        return None
    return (
        f"{n_rows} rows of the allele files name no integration, as PEtracer "
        "leaves intID empty where an intBC is not on its whitelist; they are left "
        "out."
    )


def _warn_import(attributes: Any, files_text: str) -> None:
    """Warn about what an import counted, as it does again when it is reused."""
    warn_unmatched_barcodes(attributes, files_text)
    for message in (
        _beyond_length_warning(int(attributes["n_beyond_length"])),
        _without_integration_warning(int(attributes["n_rows_without_integration"])),
    ):
        if message is not None:
            warn(message)


def _beyond_length_warning(n_beyond: int) -> str | None:
    if not n_beyond:
        return None
    return (
        f"{n_beyond} slots beyond the length that integrations gives their "
        "integration hold an insertion; they are not applicable, as the contracted "
        "array has no such slot. Check the slots that integrations names."
    )


class _RecorderAlleleOperationsMixin(_AlleleOperationsBase):
    @validation_scoped
    def _clones_import_alleles(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        cell_selection: ArtifactRef,
        format: str | None = None,
        integrations: str
        | Mapping[str, int]
        | Sequence[str | Mapping[str, int] | None]
        | None = None,
        barcode_prefix: str | Sequence[str | None] | None = None,
        barcode_suffix: str | Sequence[str | None] | None = None,
        barcode_pattern: str | Sequence[str | None] | None = None,
        columns: Mapping[str, Any] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import the alleles of a lineage recorder for a set of cells.

        Lineage recorders such as DARLIN, PEtracer and DNA Typewriter write
        heritable edits into target arrays, so cells that share an edit
        descend from the cell in which it arose. Their pipelines write each
        cell's alleles, the edits that each copy of an array carries, as
        tables of their own. The import reads them, joins each row to a
        selected cell by its cell barcode and saves a ``lineage_alleles``
        artifact, which
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_alleles`
        reads back and
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.export_characters`
        writes for tree builders.

        An integration is one copy of a target array, named as its tool
        names it, such as a DARLIN array or a PEtracer ``intID``. Each site
        of a cell is unedited, a mark, an unknown mark, missing or not
        applicable, and each format writes these states with tokens of its
        own: Cassiopeia's ``None`` is an uncut site, PEtracer's ``None`` an
        unedited one, a blank DNA Typewriter slot unedited after the last
        edit, and an insertion that holds ``N`` missing. A placeholder of an
        unknown mark, such as PEtracer's ``!``, names no allele. Text that
        pandas reads as missing is missing only in tables read through
        ``columns``. The clones API reference lists the formats and their
        tokens.

        Each file's format is recognised from its contents unless ``format``
        names it. A file that holds one array, as a CARLIN pair does, has
        its array named by ``integrations``, never inferred; the arrays of
        one library, such as DARLIN's ``CA``, ``TA`` and ``RA``, must share
        cells when they share a barcode prefix and suffix. Several alleles
        of one cell and integration are kept, as the ``darlin`` CLI,
        Cassiopeia and PEtracer write them.

        Cell barcodes join cells as the other importers join them:
        ``barcode_prefix``, the barcode and ``barcode_suffix`` name a cell's
        ID, ``barcode_pattern`` keeps the part of each barcode that its
        ``barcode`` group matches, and barcodes that match no selected cell
        are counted in the ``n_file_barcodes``, ``n_unmatched_barcodes`` and
        ``unmatched_barcode_examples`` attributes and named in a warning,
        which a reused import gives again. A file is identified by the
        SHA-256 of its bytes and of the files read beside it, so the order
        of the files does not matter.

        Args:
            paths: One allele file, or several, plain or gzip-compressed: a
                table, a CARLIN pair's file or folder, a TreeData ``.h5td``
                file or ``.zarr`` store, or a parquet table.
            cell_selection: The cells that the alleles belong to.
            format: One of ``"carlin"``, ``"darlin"``, ``"cassiopeia"``,
                ``"cassiopeia_parquet"``, ``"petracer"``,
                ``"petracer_characters"``, ``"typewriter"``,
                ``"typewriter_chains"``, ``"treedata"``, ``"characters"`` and
                ``"table"``, or None to recognise each file's format.
            integrations: For a file that holds one array, its name, such as
                ``"CA"``; for DNA Typewriter's ordered slots, a mapping of
                each integration shorter than the recorder's arrays to its
                number of slots, such as ``{"TTCACGTA": 2}``, each named by
                a row or column of the import. One for every file, or a
                sequence with one per file.
            barcode_prefix: Text before each cell barcode in the cell IDs:
                one for every file, or one per file.
            barcode_suffix: Text after each cell barcode in the cell IDs, such
                as ``"-1"``: one for every file, or one per file.
            barcode_pattern: A regular expression whose named group
                ``barcode`` is the part of each cell barcode that names the
                cell: one for every file, or one per file.
            columns: For ``format="table"``, the columns to read by role:
                ``"cell"``, ``"integration"`` (or ``integrations`` names the
                one integration), ``"sites"`` (a column per site), or
                ``"allele"`` with ``"site"`` (one site per row) or
                ``"separator"`` (sites joined in one column, as many in each
                row as in the file's first allele, or at most ``"slots"``
                ordered slots), ``"umis"``, ``"reads"``, ``"kind"``
                (``"indel"``, ``"independent"`` or ``"ordered"``) and
                ``"slots"``, and the tokens ``"unedited"``, ``"missing"``,
                ``"not_applicable"`` and ``"unknown"``, the last placeholders
                that name no allele.
            invalidate_cache: Import again instead of reusing saved alleles.

        Returns:
            A complete datastore-scoped ``lineage_alleles`` artifact.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a file's format is unknown or malformed, a site
                holds a token that its format does not read, a file that
                holds one array has no integration named, an option lists a
                value for another number of files, arrays of one library
                share no cell, or the selection holds no cell.
            FileNotFoundError: If a file does not exist.
            PermissionError: If the store is read-only and the alleles are not
                saved yet.
            MemoryError: If the import does not fit the memory budget.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        bool_argument(invalidate_cache, "invalidate_cache")
        if format is not None and not isinstance(format, str):
            raise TypeError("format must be a format name or None")
        files, layouts = self._allele_layouts(
            paths,
            format=format,
            integrations=integrations,
            columns=columns,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
        )
        sources = tuple(
            {
                **layout.record(),
                "companions": {
                    role: file_sha256(path) for role, path in layout.companions
                },
                **file.record(),
            }
            for file, layout in zip(files, layouts, strict=True)
        )
        cell_index = self._allele_selection_indices(cell_selection)
        n_cells = len(cell_index)
        arguments = RecorderAlleleImportArguments(
            cell_selection=cell_selection,
            sources=sources,
            invalidate_cache=invalidate_cache,
        )
        files_text = _files_text(len(files))
        planned = self._plan_alleles(arguments, cell_selection, n_cells)
        if planned.reused:
            # Saved alleles warn as they did when they were imported.
            saved = artifact_group(self.zw, planned.ref)
            _warn_import(
                {
                    **saved.attrs,
                    "n_beyond_length": _saved_count(saved, "n_beyond_length"),
                    "n_rows_without_integration": _saved_count(
                        saved, "n_rows_without_integration"
                    ),
                },
                files_text,
            )
            return planned.ref
        self._require_writable(arguments.operation)
        table, attributes = self._read_alleles(files, layouts, cell_index)
        self._write_alleles(planned, table, n_cells, attributes)
        _warn_import(attributes, files_text)
        return planned.ref

    def _allele_layouts(
        self,
        paths: Any,
        *,
        format: str | None,
        integrations: Any,
        columns: Mapping[str, Any] | None,
        barcode_prefix: Any,
        barcode_suffix: Any,
        barcode_pattern: Any,
    ) -> tuple[tuple[BarcodeFile, ...], list["AlleleLayout"]]:
        """Check the allele files and their options, and read each one's layout."""
        from ...readers.recorder_alleles import allele_file, allele_layout

        if isinstance(paths, str | os.PathLike):
            listed: list[Any] = [paths]
        elif isinstance(paths, Sequence) and not isinstance(paths, bytes):
            listed = list(paths)
        else:
            raise TypeError("paths must be a path or a sequence of paths")
        for path in listed:
            if not isinstance(path, str | os.PathLike):
                raise TypeError("paths must be a path or a sequence of paths")
        # A CARLIN pair is identified by the file of its cells.
        resolved = [allele_file(path) for path in listed]
        files = barcode_files(
            resolved,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            what="allele file",
            digest=path_sha256,
        )
        # Integrations follow the order of the paths, as given, and each file
        # takes those of its own listing, also when a file is listed twice.
        items = _per_file(integrations, len(resolved))
        layouts = [
            allele_layout(
                file.path,
                format=format,
                columns=columns,
                integrations=items[file.position],
            )
            for file in files
        ]
        return files, layouts

    def _allele_selection_indices(self, cell_selection: ArtifactRef) -> np.ndarray:
        cell_index = cell_selection_indices(self.zw, cell_selection)
        if not len(cell_index):
            raise ValueError("cell_selection selects no cell")
        return cell_index

    def _plan_alleles(
        self,
        arguments: RecorderAlleleImportArguments,
        cell_selection: ArtifactRef,
        n_cells: int,
    ) -> PlannedArtifact:
        record = arguments.to_record()

        def has_records(_ref: ArtifactRef, group: zarr.Group) -> bool:
            cells = group.get(f"{_RECORDS}_cells")
            return (
                isinstance(cells, zarr.Array)
                and len(cells.shape) == 1
                and group.attrs.get("fields") == list(FIELDS)
                and group.attrs.get("states") == list(STATES)
                and all(f"{_RECORDS}_{name}" in group for name in FIELDS)
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
            arrays={name: ((n_cells,), "i") for name in _CELL_ARRAYS},
            invalidate_cache=arguments.invalidate_cache,
            required_attributes=_IMPORT_ATTRIBUTES,
            reuse_validator=has_records,
        )

    def _check_allele_budget(
        self,
        encoder: RecordEncoder,
        *,
        cells: int,
        reading: int = 0,
        merge_text: int = 0,
    ) -> None:
        """Refuse an import whose peak would pass the memory budget.

        The peak is the largest of reading a chunk beside the records held,
        merging the records into one table and writing the table beside the
        file buffers of Zarr's writes, beside what the import holds whatever
        its size.

        Args:
            encoder: The records held so far.
            cells: The bytes held for the selected cells, such as their lookup.
            reading: The bytes held to read a file: its chunk, what it holds
                beside and the barcodes that match no cell.
            merge_text: The bytes that merging takes for the distinct text
                values of the chunks held.
        """
        held = encoder.nbytes
        merging = _MERGE_FACTOR * held + max(
            merge_text,
            _WRITE_BYTES_PER_RECORD * encoder.n_records + zarr_write_buffer_bytes(),
        )
        peak = cells + _FIXED_BYTES + max(held + reading, merging)
        if peak > self.memoryBytes:
            selected = cells + merging
            fewer = ", or pass a smaller cell_selection" if selected >= reading else ""
            raise MemoryError(
                f"Importing alleles holds about {peak} bytes at its peak, {cells} of "
                f"them for the selected cells and {merging} to merge and write "
                f"their {encoder.n_records} records, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget{fewer}."
            )

    def _read_alleles(
        self,
        files: Sequence[BarcodeFile],
        layouts: Sequence["AlleleLayout"],
        cell_index: np.ndarray,
    ) -> tuple[RecordTable, dict[str, Any]]:
        """Read the alleles of the files' rows that name selected cells."""
        n_cells = len(cell_index)
        single = sum(1 for layout in layouts if layout.integration is not None)
        # The per-cell arrays and a mask of each file of one array's cells,
        # which find arrays that share no cell.
        own = (_CELL_BYTES + single) * n_cells
        encoder = RecordEncoder(FIELDS)
        # Files that hold no row, such as a header alone, give no records.
        encoder.add(*empty_allele_chunk(), 0)
        # The selected cells' share, with the join's lookup and the read of
        # their IDs, which decodes a stored chunk of them, is checked before
        # a cell ID is read.
        self._check_allele_budget(
            encoder,
            cells=own
            + LOOKUP_BYTES_PER_CELL * n_cells
            + text_read_bytes(self.cells._get_array("ids"), n_cells),
        )
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )
        join = CellBarcodeJoin(cell_ids, memory_bytes=self.memoryBytes, held_bytes=own)
        del cell_ids
        # From here the import holds what the join counts: its lookup, the
        # cells' IDs as text and the import's own bytes per cell.
        held = join.held_bytes
        named: list[np.ndarray | None] = []
        counted: list[dict[str, int]] = []
        merged = _MergedText()
        # The integrations of contracted tapes that some file's rows name.
        tapes: set[str] = set()
        for position, (file, layout) in enumerate(zip(files, layouts, strict=True)):
            mask = (
                np.zeros(n_cells, dtype=bool)
                if layout.integration is not None
                else None
            )
            counts = self._read_allele_file(
                encoder,
                join,
                file,
                layout,
                position,
                mask=mask,
                held=held,
                merged=merged,
                tapes=tapes,
            )
            named.append(mask)
            counted.append(counts)
        _check_shared_cells(files, layouts, named)
        _check_named_tapes(files, layouts, tapes)
        del named
        self._check_allele_budget(encoder, cells=held, merge_text=merged.nbytes)
        table = encoder.finish()
        if any(layout.rows_hold_sites for layout in layouts):
            table = records_in_line_order(table)
        attributes = {
            "files": [
                {
                    **layout.source(),
                    "lengths": dict(sorted(layout.lengths.items())),
                    "rows": counts["rows"],
                    "records": counts["records"],
                }
                for layout, counts in zip(layouts, counted, strict=True)
            ],
            "n_file_rows": join.n_rows,
            "n_unmatched_rows": join.n_unmatched_rows,
            "n_beyond_length": sum(counts["beyond"] for counts in counted),
            "n_rows_without_integration": sum(
                counts["without_integration"] for counts in counted
            ),
            **join.attributes(),
        }
        return table, attributes

    def _read_allele_file(
        self,
        encoder: RecordEncoder,
        join: CellBarcodeJoin,
        file: BarcodeFile,
        layout: "AlleleLayout",
        position: int,
        *,
        mask: np.ndarray | None,
        held: int,
        merged: "_MergedText",
        tapes: set[str],
    ) -> dict[str, int]:
        """Read one file into the encoder.

        ``merged`` gains the distinct text values of the file's chunks, and
        ``tapes`` the integrations of contracted tapes that its rows name.

        Returns:
            The file's ``rows``, its ``records``, its edits ``beyond`` their
            integration's length and its rows ``without_integration``.
        """
        from ...readers.recorder_alleles import classified_rows, iter_allele_rows

        chunk_rows = layout.chunk_rows(self.memoryBytes // _CHUNK_SHARE)
        reading = layout.held_bytes() + layout.chunk_bytes(chunk_rows)
        # A budget too small for the lookup, the file's names and one chunk
        # refuses before a row is read.
        self._check_allele_budget(
            encoder,
            cells=held,
            reading=reading + join.unmatched_bytes,
            merge_text=merged.nbytes,
        )
        counts = {"rows": 0, "records": 0, "beyond": 0, "without_integration": 0}
        for rows in iter_allele_rows(layout, chunk_rows):
            classified = classified_rows(layout, rows)
            cells = join.cells(file, rows.barcodes)
            if mask is not None:
                mask[cells[cells >= 0]] = True
            frame, record_cells = allele_chunk(rows, cells, classified)
            merged.add(frame)
            encoder.add(frame, record_cells, position)
            counts["rows"] += len(rows)
            counts["records"] += len(record_cells)
            counts["beyond"] += classified.n_beyond
            if rows.integrations is not None:
                counts["without_integration"] += int(
                    np.count_nonzero(pd.isna(rows.integrations))
                )
                if layout.lengths:
                    tapes.update(
                        set(pd.unique(rows.integrations).tolist()) & set(layout.lengths)
                    )
            self._check_allele_budget(
                encoder,
                cells=held,
                reading=reading + join.unmatched_bytes,
                merge_text=merged.nbytes,
            )
        return counts

    def _write_alleles(
        self,
        planned: PlannedArtifact,
        table: RecordTable,
        n_cells: int,
        attributes: Mapping[str, Any],
    ) -> None:
        summaries = cell_summaries(table, n_cells)
        arrays = record_arrays(table, _RECORDS)
        # A state fits a byte.
        arrays[f"{_RECORDS}_state"] = np.asarray(
            arrays[f"{_RECORDS}_state"], dtype=np.int8
        )
        write_cell_data_artifact(
            self.zw,
            planned,
            {**summaries, **arrays},
            chunk_lengths=record_chunk_lengths(arrays),
            attributes={
                "fields": list(FIELDS),
                "states": list(STATES),
                **attributes,
            },
        )
        n_with = int(np.count_nonzero(summaries["n_records"]))
        logger.info(
            f"Imported {table.n_records} allele records of {n_with} of {n_cells} "
            f"cells from {len(attributes['files'])} files"
        )

    def _allele_group(self, alleles: ArtifactRef) -> zarr.Group:
        """Return a complete ``lineage_alleles`` artifact's group."""
        if not isinstance(alleles, ArtifactRef):
            raise TypeError("alleles must be an ArtifactRef")
        if alleles.kind != RecorderAlleleImportArguments.artifact_kind:
            raise ValueError(
                f"alleles must be a lineage_alleles artifact, not {alleles.kind!r}"
            )
        require_complete_artifact(self.zw, alleles)
        return artifact_group(self.zw, alleles)

    def _allele_counts(self, group: zarr.Group) -> dict[str, int]:
        """The sizes of a saved import, read from its arrays' shapes.

        Returns:
            ``records``, ``cells``, the distinct ``integrations``, ``sites``
            and ``alleles``, ``allele_bytes``, the bytes of the distinct
            alleles, ``text_bytes``, those of the distinct integrations,
            sites and alleles, ``text_width``, the most bytes that a
            character of them takes decoded per stored byte, and
            ``decoded_bytes``, what decoding them holds beside their values.
        """

        def length(name: str) -> int:
            return int(as_zarr_array(group[name], name=name).shape[0])

        text_bytes = sum(length(f"{_RECORDS}_{name}_values") for name in _TEXT_FIELDS)
        width = _text_width(group)
        return {
            "records": length(f"{_RECORDS}_cells"),
            "cells": length("n_edits"),
            "integrations": length(f"{_RECORDS}_integration_offsets") - 1,
            "sites": length(f"{_RECORDS}_site_offsets") - 1,
            "alleles": length(f"{_RECORDS}_allele_offsets") - 1,
            "allele_bytes": length(f"{_RECORDS}_allele_values"),
            "text_bytes": text_bytes,
            "text_width": width,
            "decoded_bytes": (_STORED_TEXT_COPIES + _DECODED_TEXT_COPIES * width)
            * text_bytes,
        }

    @validation_scoped
    def _clones_load_alleles(self, alleles: ArtifactRef) -> LineageAlleles:
        """Read saved recorder alleles as tables of cells, records and files.

        Args:
            alleles: A ``lineage_alleles`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_alleles`.

        Returns:
            The alleles, with one row per cell, one row per cell, integration,
            site and allele row, and one row per file.

        Raises:
            TypeError: If ``alleles`` is not an ArtifactRef.
            ValueError: If it is not a ``lineage_alleles`` artifact.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
            MemoryError: If the tables do not fit the memory budget.
        """
        group = self._allele_group(alleles)
        counts = self._allele_counts(group)
        n_records, n_cells = counts["records"], counts["cells"]
        n_text = counts["integrations"] + counts["sites"] + counts["alleles"]
        # The stored arrays and the tables built from them, each distinct
        # text and cell ID as a category, the texts as they are decoded, and
        # the cells' IDs as they are read.
        resident = (
            _FIXED_BYTES
            + _TABLE_BYTES_PER_RECORD * n_records
            + _TABLE_BYTES_PER_TEXT * (n_text + n_cells)
            + counts["decoded_bytes"]
            + _CELL_BYTES * n_cells
            + text_read_bytes(self.cells._get_array("ids"), n_cells)
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"The tables of {n_cells} cells and {n_records} allele records hold "
                f"about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        loaded = self._artifacts_load_values(alleles)
        table = stored_records(group, _RECORDS, FIELDS)
        summaries = {
            "n_edits": np.asarray(loaded.values),
            **{name: read_array(group, name) for name in _CELL_ARRAYS[1:]},
        }
        return lineage_allele_frames(
            loaded.cell_ids, summaries, table, _stored_files(group)
        )

    @validation_scoped
    def _clones_export_characters(
        self,
        alleles: ArtifactRef,
        path: str | os.PathLike[str],
        *,
        format: str = "cassiopeia",
        unknown: str = "state",
    ) -> int:
        """Write saved recorder alleles as a character matrix for tree builders.

        A character is one site of one integration, named
        ``{integration}-{site}``, or after its integration when the site has
        no name, as a DARLIN array's allele does. Its states are integers:
        0 unedited, missing where a site is missing or not applicable, and
        its edits from 1: a mark written as a whole number without leading
        zeros, such as PEtracer's marks ``1`` to ``8``, is its own state, and
        the character's other marks, by their text, then its unknown marks
        that name their allele, take the states after the largest such
        number. So each state names one allele of its character in every
        cell, a character matrix keeps its states through an import and an
        export, and a table names each state's allele. An unknown mark that a
        placeholder writes, such as PEtracer's ``!`` or CARLIN's ``X``, names
        no allele, so cells that hold it share no state: it is missing.
        Where a cell holds several allele rows of one
        integration, the row of most UMIs, then most reads, then most sites
        read is exported, as Cassiopeia keeps each integration's most
        abundant allele, and a warning counts them. Ordered slots export one
        character per slot, so their prefix rule is left to the tree
        builder's model. Cells that read no character and characters that
        no cell reads are left out.

        ``"cassiopeia"`` writes a TreeData ``.zarr`` store, which Cassiopeia
        3.0's solvers read: ``obsm['characters']`` with one int64 column per
        character, ``uns['missing_state']`` -1, ``uns['unmodified_state']``
        0 and ``uns['character_states']``, the ``character``, ``state`` and
        ``allele`` of each edited state. ``"laml"`` writes LAML's CSV, a
        header ``cell,<characters>`` and integer states with ``?`` for
        missing, and beside it ``<stem>_priors.csv``, headerless rows of a
        character's position from 0, a state and its probability, each
        state's share of the exported cells with an edit at its character,
        and ``<stem>_states.csv``, the table of each state's allele. Cell IDs
        are this store's.

        Args:
            alleles: A ``lineage_alleles`` artifact.
            path: For ``"cassiopeia"``, the ``.zarr`` store to create, which
                must not exist; for ``"laml"``, the CSV to write, replacing a
                file of that name, as the files beside it are.
            format: ``"cassiopeia"`` or ``"laml"``.
            unknown: ``"state"`` gives each distinct unknown mark that names
                its allele, such as an insertion outside PEtracer's lookup, a
                state of its own after the marks; ``"missing"`` writes every
                unknown mark as missing.

        Returns:
            The number of cells written.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``alleles`` is not a ``lineage_alleles`` artifact,
                ``format`` or ``unknown`` is unknown, no cell reads a
                character, two integrations give one character name, a
                character's states pass the int32 range, or, for LAML, a cell
                or character name holds a character that its files cannot
                hold.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
            FileExistsError: For ``"cassiopeia"``, if ``path`` exists.
            MemoryError: If the export does not fit the memory budget.
        """
        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path")
        if format not in EXPORT_FORMATS:
            raise ValueError(
                f"format must be one of {', '.join(map(repr, EXPORT_FORMATS))}, not "
                f"{format!r}"
            )
        if unknown not in UNKNOWN_POLICIES:
            raise ValueError(
                f"unknown must be one of {', '.join(map(repr, UNKNOWN_POLICIES))}, "
                f"not {unknown!r}"
            )
        target = os.fspath(path)
        if format == "cassiopeia" and os.path.lexists(target):
            raise FileExistsError(
                f"{target} exists; export the characters to a new path, since a Zarr "
                "store is a folder that an export would have to delete"
            )
        group = self._allele_group(alleles)
        counts = self._allele_counts(group)
        n_records, n_cells = counts["records"], counts["cells"]
        # The records and what the matrix builds from them, each distinct
        # allele as it is ranked, the texts as they are decoded, each
        # character as it is named and ordered (at most every integration at
        # every site), the exported cells' IDs, and a store's column and its
        # alleles as they are written, beside the file buffers of Zarr's
        # writes; the matrix's codes are counted once their size is known.
        resident = (
            _FIXED_BYTES
            + _EXPORT_BYTES_PER_RECORD * n_records
            + _EXPORT_BYTES_PER_ALLELE * counts["alleles"]
            + counts["decoded_bytes"]
            + _EXPORT_BYTES_PER_CHARACTER
            * min(n_records, counts["integrations"] * max(1, counts["sites"]))
            + text_read_bytes(self.cells._get_array("ids"), n_cells)
            + _STORE_BYTES_PER_CELL * n_cells
            + (
                _STORE_BYTES_PER_ALLELE_BYTE * counts["allele_bytes"]
                + zarr_write_buffer_bytes()
                if format == "cassiopeia"
                else 0
            )
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Exporting the characters of {n_records} allele records holds about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        table = stored_records(group, _RECORDS, FIELDS)
        _merge_site_rows(table, _stored_files(group))
        matrix = character_matrix(
            table,
            unknown=unknown,
            memory_bytes=self.memoryBytes,
            held_bytes=resident,
        )
        del table
        if not len(matrix.cells) or not len(matrix.characters):
            raise ValueError(
                "No cell reads a character of these alleles, so there is no character "
                "matrix to export"
            )
        cell_index = cell_selection_indices(
            self.zw, inspect_artifact(self.zw, alleles).input_ref("cell_selection")
        )
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index[matrix.cells]),
            dtype=object,
        )
        if matrix.n_resolved:
            warn(
                f"{matrix.n_resolved} cells and integrations have several allele rows; "
                "the export keeps the row of most UMIs, then most reads, then most "
                "sites read, of each"
            )
        if format == "cassiopeia":
            from ...writers.characters import write_cassiopeia_characters

            write_cassiopeia_characters(target, cell_ids, matrix)
        else:
            from ...writers.characters import laml_row_bytes, write_laml_characters

            # LAML's text is written as many cells at a time as a twentieth of
            # the budget holds, or as the rest of it leaves room for, at least
            # one.
            row_bytes = laml_row_bytes(len(matrix.characters))
            held = resident + int(matrix.codes.nbytes)
            if held + row_bytes > self.memoryBytes:
                raise MemoryError(
                    f"Writing LAML's text of {len(matrix.characters)} characters "
                    f"holds about {held + row_bytes} bytes, more than the memory "
                    f"budget of {self.memoryBytes} bytes. Raise mem_budget."
                )
            write_laml_characters(
                target,
                cell_ids,
                matrix,
                block_cells=max(
                    1, min(self.memoryBytes // 20, self.memoryBytes - held) // row_bytes
                ),
            )
        logger.info(
            f"Exported {len(matrix.cells)} cells by {len(matrix.characters)} "
            f"characters to {target}"
        )
        return int(len(matrix.cells))

    @validation_scoped
    def _clones_allele_clones(
        self,
        alleles: ArtifactRef,
        *,
        method: str,
        frequencies: Mapping[str, pd.DataFrame] | None = None,
        max_frequency: float | None = None,
        max_samples: int | None = None,
        max_alleles: int | None = None,
        fdr: float = 0.05,
        within: str | Sequence[str] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call clones from rare lineage recorder alleles.

        Cells that share an edit descend from the cell in which it arose,
        unless it arose more than once: CRISPR arrays repeat their most
        likely edits in unrelated cells. So cells are linked through rare
        alleles only, and the connected components of the cells that share
        a rare allele, at any character, are the clones. An allele is one
        character's edit, the text of a mark or of an unknown mark that names
        its allele, such as a DARLIN array's ``23_268del,55_248del``; the
        unedited state, placeholders of unknown marks such as CARLIN's ``X``,
        and missing sites never link cells.

        ``method="graph"`` follows DARLIN (Li et al., Cell 2023): an allele is
        rare when its frequency is below ``max_frequency`` and it was found in
        at most ``max_samples`` of its bank's samples; DARLIN used ``2e-5``
        and one mouse. ``method="p_clonal"`` follows CARLIN (Bowling et al.,
        Cell 2020): an allele of bank rate λ among the M cells of its group
        with an edit at its character arose more than once with probability
        p_clonal = (1 - e^(-λM) - λM e^(-λM)) / (1 - e^(-λM)), and is rare
        when that p-value survives the Benjamini-Hochberg procedure at
        ``fdr`` over the alleles of its group, every array's in one family,
        and when it passes ``max_samples``. A clone whose cells hold more than ``max_alleles``
        distinct rare alleles leaves its cells without one, as DARLIN keeps
        clones of at most three: a cell holds at most one allele per array,
        so a clone of more is likely two that a homoplasic allele joined.

        ``frequencies`` gives each character's bank, a DataFrame of each
        allele's ``normalized_count``, its frequency among the bank's edited
        observations, and optionally its ``sample_count``, as DARLIN's banks
        hold them; ``pd.read_csv("allele_bank_Gr_CA.csv.gz")`` reads one. A
        bank fits only alleles annotated by the pipeline that annotated it:
        CARLIN's MATLAB notation and darlin-core's place some events one base
        apart, so a bank in the other notation lists few of the alleles and
        calls most of them rare. The artifact's ``bank_coverage`` attribute
        counts, for each character, its alleles and their observations, the
        pairs of a cell and an allele, and how many of each its bank lists.
        An allele missing from its bank counts as rare, with frequency and
        rate 0, as DARLIN counts it; CARLIN gives such alleles the mean rate
        of the alleles its bank model never observed, which a table of rates
        does not hold. p_clonal grows with M, so among tens of thousands of
        edited cells or more the listed alleles fail and the alleles missing
        from the bank are the ones that survive. Without a bank, an allele's
        frequency is the share of the cells of its group with an edit at its
        character that hold it. Clonal expansion inflates these frequencies,
        so the alleles of large clones look common, and ``"p_clonal"`` needs
        a bank.

        Clones are numbered by size, so ``clone_1`` is the largest, and ties
        follow their first cell. The ``clones`` artifact also holds each
        cell's ``n_alleles``, its distinct rare alleles; its attributes
        record what each rule removed, and its parameters the rules.

        Args:
            alleles: A ``lineage_alleles`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_alleles`.
            method: ``"graph"`` or ``"p_clonal"``.
            frequencies: Each character's bank, keyed by its name: its
                integration, or ``{integration}-{site}`` for a named site, as
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.export_characters`
                names characters. Every character with an allele needs one.
                None takes each allele's frequency from the cells.
            max_frequency: For ``"graph"``, the frequency below which an
                allele is rare, greater than 0 and at most 1, or None for no
                such filter.
            max_samples: The most samples of a bank in which a rare allele is
                found, such as 1 for alleles of at most one bank mouse, or
                None for no such filter; it needs banks with a
                ``sample_count``.
            max_alleles: The most distinct rare alleles of a clone, such as
                3, or None for no such filter.
            fdr: For ``"p_clonal"``, the false discovery rate, greater than 0
                and less than 1.
            within: Cell metadata columns whose labels keep clones apart, such
                as the mouse; alleles are tested within each group, and every
                cell with an edit needs a label.
            invalidate_cache: Call again instead of reusing saved clones.

        Returns:
            A complete datastore-scoped ``clones`` artifact aligned to the
            alleles' cells. Read it with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_clones`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``alleles`` is not a ``lineage_alleles`` artifact,
                the method is unknown, a rule is missing, out of range or
                does not apply to the method, a bank is malformed, names no
                character with an edit or is missing for one, or a within
                column is missing or has no label for a cell with an edit.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If the calling does not fit the memory budget.
        """
        bool_argument(invalidate_cache, "invalidate_cache")
        group = self._allele_group(alleles)
        banks = None
        bank_bytes = 0
        if frequencies is not None:
            checking, bank_bytes = _bank_bytes(frequencies)
            _check_calling(
                checking + _FIXED_BYTES, self.memoryBytes, "the banks of frequencies"
            )
            banks = allele_banks(frequencies)
        settings = allele_call_settings(
            method,
            max_frequency=max_frequency,
            max_samples=max_samples,
            max_alleles=max_alleles,
            fdr=fdr,
            banks=banks,
        )
        columns = group_columns(within, "within", _CLONE_COLUMNS)
        cell_selection = inspect_artifact(self.zw, alleles).input_ref("cell_selection")
        counts = self._allele_counts(group)
        n_cells = counts["cells"]
        # Every cell with an edit needs a label in the within columns: each
        # cell's edits are read, then held as a mask while the cells group.
        _check_calling(
            bank_bytes + 9 * n_cells + _FIXED_BYTES,
            self.memoryBytes,
            f"the edits of {n_cells} cells",
        )
        with_edits = read_array(group, "n_edits") > 0
        group_codes, _keys, within_fingerprint = within_groups(
            self.zw,
            self.cells,
            cell_selection,
            columns,
            with_edits,
            memory_bytes=self.memoryBytes - bank_bytes - n_cells,
            reserved=_CLONE_COLUMNS,
            what="an edit",
        )
        del with_edits
        record = settings.record()
        arguments = AlleleCloneArguments(
            alleles=alleles,
            within_fingerprint=within_fingerprint,
            frequencies_fingerprint=None
            if banks is None
            else _banks_fingerprint(banks),
            method=record["method"],
            max_frequency=record["max_frequency"],
            max_samples=record["max_samples"],
            max_alleles=record["max_alleles"],
            fdr=record["fdr"],
            within=columns,
            invalidate_cache=invalidate_cache,
        )
        planned = plan_clones(
            self.zw,
            arguments,
            cell_selection,
            n_cells,
            with_probability=False,
            required_attributes=("n_clones", "rule_counts", "bank_coverage"),
            source_arrays={"n_alleles": "i"},
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        n_records = counts["records"]
        # Saving the clones is checked first, as if every cell were a clone of
        # its own, the most clones there can be.
        writing = _saved_clone_bytes(n_cells, n_cells)
        held = bank_bytes + _CALL_CELL_BYTES * n_cells
        _check_calling(
            held + writing + _FIXED_BYTES,
            self.memoryBytes,
            f"the clones of {n_cells} cells as they are written",
        )
        # The distinct alleles, integrations and sites are decoded as text
        # before the records are read, and held while the clones are called,
        # each character at the width of the widest.
        n_texts = counts["alleles"] + counts["integrations"] + counts["sites"]
        _check_calling(
            held
            + _DECODE_TEXT_BYTES * n_texts
            + counts["decoded_bytes"]
            + _FIXED_BYTES,
            self.memoryBytes,
            f"the text of {counts['alleles']} distinct alleles as it is decoded",
        )
        texts = _record_texts(group)
        held += _HELD_TEXT_BYTES * n_texts + counts["text_width"] * counts["text_bytes"]
        # The states are read first, so that only the records of edited sites
        # are read on.
        _check_calling(
            held + _STATE_READ_BYTES * n_records + _FIXED_BYTES,
            self.memoryBytes,
            f"the states of {n_records} allele records",
        )
        states = read_array(group, f"{_RECORDS}_state")
        edited = states == MARK
        edited |= states == UNKNOWN_MARK
        del states
        n_edited = int(np.count_nonzero(edited))
        _check_calling(
            held
            + n_records
            + _COLUMN_READ_BYTES * n_records
            + _CALL_EDIT_BYTES * n_edited
            + _FIXED_BYTES,
            self.memoryBytes,
            f"the {n_edited} edited sites of {n_cells} cells as they are read",
        )
        observations = _allele_observations(group, edited, texts)
        del edited, texts
        held += _HELD_EDIT_BYTES * n_edited
        # The calls get what the budget leaves beside what the operation holds
        # and its fixed bytes, which every check of the operation counts.
        called = allele_clone_calls(
            observations,
            n_cells,
            groups=group_codes,
            settings=settings,
            banks=banks,
            max_bytes=self.memoryBytes - held - _FIXED_BYTES,
        )
        del observations, banks
        # The clones are saved beside each cell's group and calls, now that
        # their number is known.
        _check_calling(
            _CALL_CELL_BYTES * n_cells
            + _saved_clone_bytes(n_cells, called.n_clones)
            + _FIXED_BYTES,
            self.memoryBytes,
            f"the {called.n_clones} clones of {n_cells} cells as they are written",
        )
        labels = clone_labels(called.n_clones)
        assignment = CloneAssignment(
            codes=called.codes,
            clone_ids=labels,
            clone_sizes=called.sizes,
            probabilities=None,
            n_cells_in_several_clones=0,
        )
        write_clones(
            self.zw,
            planned,
            assignment,
            {
                "n_clones": called.n_clones,
                "rule_counts": called.counts,
                "bank_coverage": called.coverage,
            },
            source_arrays={"n_alleles": called.n_alleles.astype(np.int32)},
            memory_bytes=self.memoryBytes,
            held_bytes=(_CALL_CELL_BYTES + _CALLED_CELL_BYTES) * n_cells
            + sum(map(sys.getsizeof, labels.tolist()))
            + _FIXED_BYTES,
        )
        for entry in called.coverage:
            share = (
                entry["observations_in_bank"] / entry["observations"]
                if entry["observations"]
                else 0.0
            )
            logger.info(
                f"The bank of {entry['character']} lists {entry['alleles_in_bank']} of "
                f"its {entry['alleles']} alleles, which {share:.1%} of their cells hold"
            )
        n_assigned = int(np.count_nonzero(called.codes >= 0))
        largest = int(called.sizes[0]) if called.n_clones else 0
        logger.info(
            f"Called {called.n_clones} clones of {n_assigned} of {n_cells} cells from "
            f"{called.counts['rare_alleles']} rare alleles; the largest holds "
            f"{largest} cells"
        )
        return planned.ref


def _saved_clone_bytes(n_cells: int, n_clones: int) -> int:
    """What saving the clones of ``n_cells`` holds beside each cell's group and calls.

    The calls' rare alleles of each cell, each clone's label as a Python
    string at the size of the longest, and what saving the clones holds.
    """
    longest = f"clone_{n_clones}"
    return (
        _CALLED_CELL_BYTES * n_cells
        + sys.getsizeof(longest) * n_clones
        + clone_write_bytes(
            n_cells, n_clones, len(longest), source_arrays={"n_alleles": np.int32}
        )
    )


def _check_calling(needed: int, budget: int, what: str) -> None:
    """Raise MemoryError before calling clones holds more than the budget."""
    if needed > budget:
        raise MemoryError(
            f"Calling clones from recorder alleles holds about {needed} bytes for "
            f"{what}, more than the memory budget of {budget} bytes. Raise mem_budget."
        )


def _bank_bytes(frequencies: Any) -> tuple[int, int]:
    """The bytes that checking the banks of ``frequencies`` takes, and holding them.

    Values of the wrong type count nothing, so that the check of the banks
    refuses them by name.
    """
    if not isinstance(frequencies, Mapping):
        return 0, 0
    checking = holding = 0
    for table in frequencies.values():
        if not isinstance(table, pd.DataFrame) or "allele" not in table.columns:
            continue
        lengths = table["allele"].map(
            lambda value: len(value) if isinstance(value, str) else 0
        )
        text = _BANK_CHARACTER_BYTES * int(lengths.sum())
        checking += _BANK_ROW_BYTES * len(table) + text
        holding += _BANK_HELD_ROW_BYTES * len(table) + text
    return checking, holding


def _banks_fingerprint(banks: Mapping[str, AlleleBank]) -> str:
    """Fingerprint each character's bank: its sorted alleles, frequencies and samples."""
    builder = ValueFingerprintBuilder()
    for name in sorted(banks):
        bank = banks[name]
        builder.update_bytes("character", name.encode("utf-8"))
        data, offsets = pack_text(bank.alleles)
        builder.update_array("alleles", data)
        builder.update_array("allele_offsets", offsets)
        builder.update_array("frequencies", np.asarray(bank.frequencies, dtype="<f8"))
        if bank.samples is None:
            builder.update_bytes("samples", b"")
        else:
            builder.update_array("samples", np.asarray(bank.samples, dtype="<i8"))
    return builder.hexdigest()


def _record_texts(group: zarr.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distinct integrations, sites and alleles of saved records, as text."""

    def texts(name: str) -> np.ndarray:
        return unpack_text(
            read_array(group, f"{_RECORDS}_{name}_values"),
            read_array(group, f"{_RECORDS}_{name}_offsets"),
        )

    return texts("integration"), texts("site"), texts("allele")


def _allele_observations(
    group: zarr.Group,
    edited: np.ndarray,
    texts: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> AlleleObservations:
    """Read the records of edited sites: each one's cell, character and allele.

    A character is a pair of an integration and a site, coded as one integer
    key, so the characters need one sort of the edits' keys. The characters
    that hold no edit are found from both full columns a block at a time.
    ``texts`` holds the distinct integrations, sites and alleles as text.
    """
    integration_names, site_names, allele_texts = texts
    n_sites = max(1, len(site_names))
    alleles = read_array(group, f"{_RECORDS}_allele")[edited].astype(np.int64)
    integrations = read_array(group, f"{_RECORDS}_integration")
    sites = read_array(group, f"{_RECORDS}_site")
    every = [
        sorted_unique(
            integrations[start : start + _KEY_BLOCK].astype(np.int64) * n_sites
            + sites[start : start + _KEY_BLOCK]
        )
        for start in range(0, len(integrations), _KEY_BLOCK)
    ]
    integrations = integrations[edited]
    sites = sites[edited]
    keys = integrations.astype(np.int64)
    del integrations
    keys *= n_sites
    keys += sites
    del sites
    distinct, characters = unique_inverse(keys)
    del keys
    cells = read_array(group, f"{_RECORDS}_cells")[edited].astype(np.int64)

    def name(key: int) -> str:
        return character_name(
            str(integration_names[key // n_sites]), str(site_names[key % n_sites])
        )

    edited_names = tuple(name(key) for key in distinct.tolist())
    all_keys = sorted_unique(np.concatenate(every)) if every else distinct
    return AlleleObservations(
        cells=cells,
        characters=characters,
        alleles=alleles,
        character_names=edited_names,
        allele_texts=allele_texts,
        unedited_characters=tuple(
            sorted({name(key) for key in all_keys.tolist()} - set(edited_names))
        ),
    )


def _utf8_width(largest: int) -> int:
    """The most bytes per UTF-8 byte that Python holds a text in, by its largest byte.

    A lead byte from 0xF0 starts a character beyond U+FFFF, which Python
    holds in four bytes, and one from 0xC4 a character beyond U+00FF, which
    it holds in two; ASCII and Latin-1 text takes one.
    """
    for lead, width in _WIDE_LEAD_BYTES:
        if largest >= lead:
            return width
    return 1


def _text_width(group: zarr.Group) -> int:
    """The most bytes that a decoded character takes per stored byte of the texts.

    The stored UTF-8 bytes of the integrations, sites and alleles are read a
    chunk at a time for their largest byte.
    """
    largest = 0
    for name in _TEXT_FIELDS:
        values = as_zarr_array(
            group[f"{_RECORDS}_{name}_values"], name=f"{_RECORDS}_{name}_values"
        )
        block = max(1, int(values.chunks[0]))
        for start in range(0, int(values.shape[0]), block):
            chunk = np.asarray(values[start : start + block])
            if len(chunk):
                largest = max(largest, int(chunk.max()))
    return _utf8_width(largest)


@dataclass(slots=True)
class _MergedText:
    """The distinct text values of the chunks held, which merging joins.

    Merging joins every chunk's distinct values into one text before it
    encodes them, so one value's widest character widens all of them.
    """

    texts: int = 0
    bytes: int = 0
    largest: int = 0

    def add(self, frame: pd.DataFrame) -> None:
        """Count a chunk's distinct integrations, sites and alleles."""
        for name in _TEXT_FIELDS:
            distinct = pd.unique(frame[name].to_numpy(dtype=object))
            for value in distinct.tolist():
                if not isinstance(value, str):
                    continue
                encoded = value.encode("utf-8")
                self.texts += 1
                self.bytes += len(encoded)
                if not value.isascii():
                    self.largest = max(
                        self.largest, int(np.frombuffer(encoded, np.uint8).max())
                    )

    @property
    def nbytes(self) -> int:
        """The bytes that merging takes for these values."""
        return (
            _MERGE_BYTES_PER_TEXT * self.texts
            + (_MERGED_TEXT_BYTES + _MERGED_TEXT_COPIES * _utf8_width(self.largest))
            * self.bytes
        )


def _merge_site_rows(table: RecordTable, files: Sequence[Mapping[str, Any]]) -> None:
    """Give the records of files whose rows each hold one site the row 0.

    The rows of one cell and integration of such a file are one allele, which
    the export keeps or leaves whole.
    """
    hold_sites = [
        position for position, entry in enumerate(files) if entry.get("rows_hold_sites")
    ]
    if hold_sites:
        rows = np.asarray(table.values["row"])
        rows[np.isin(np.asarray(table.sources), hold_sites)] = 0


def _check_shared_cells(
    files: Sequence[BarcodeFile],
    layouts: Sequence["AlleleLayout"],
    named: Sequence[np.ndarray | None],
) -> None:
    """Refuse arrays of one library that share no cell.

    Files that hold one array each and share a barcode prefix, suffix and
    pattern hold the arrays of one library, such as DARLIN's CA, TA and RA,
    so each must share a cell with another array of the library.

    Raises:
        ValueError: If a file's cells are none that another array's file names.
    """
    libraries: dict[tuple[str, str, str], list[int]] = {}
    for position, (file, layout) in enumerate(zip(files, layouts, strict=True)):
        if layout.integration is None:
            continue
        key = (
            file.barcode_prefix or "",
            file.barcode_suffix or "",
            file.barcode_pattern or "",
        )
        libraries.setdefault(key, []).append(position)
    for positions in libraries.values():
        arrays = {layouts[position].integration for position in positions}
        if len(arrays) < 2:
            continue
        for position in positions:
            own = named[position]
            assert own is not None
            others = [
                named[other]
                for other in positions
                if layouts[other].integration != layouts[position].integration
            ]
            union = np.logical_or.reduce([mask for mask in others if mask is not None])
            if own.any() and not (own & union).any():
                raise ValueError(
                    f"The cells of {os.fspath(files[position].path)} ({layouts[position].integration}) "
                    "are none that the other arrays of the import name with the same "
                    "barcode_prefix and barcode_suffix, so they are no arrays of one "
                    "library; give each library its own barcode_prefix"
                )


def _check_named_tapes(
    files: Sequence[BarcodeFile], layouts: Sequence["AlleleLayout"], tapes: set[str]
) -> None:
    """Refuse slots of contracted tapes that no row of the import names.

    A misspelt integration would otherwise leave its tape at the recorder's
    full length without a word.
    """
    given = {
        name
        for layout in layouts
        if layout.lengths and layout.format != "typewriter_chains"
        for name in layout.lengths
    }
    lacking = sorted(given - tapes)
    if lacking:
        paths = ", ".join(
            os.fspath(file.path)
            for file, layout in zip(files, layouts, strict=True)
            if layout.lengths
        )
        raise ValueError(
            f"integrations gives the slots of {', '.join(map(repr, lacking))}, which "
            f"no row of {paths} names; check the names of the contracted tapes"
        )


def _stored_files(group: zarr.Group) -> list[dict[str, Any]]:
    """Read the files that a ``lineage_alleles`` artifact records."""
    files = group.attrs.get("files")
    keys = {"format", "tool", "pipeline", "kind", "integration", "rows", "records"}
    if not isinstance(files, list) or not all(
        isinstance(entry, dict) and keys <= set(entry) for entry in files
    ):
        raise ValueError("The lineage_alleles artifact has no valid list of files")
    return [dict(entry) for entry in files]
