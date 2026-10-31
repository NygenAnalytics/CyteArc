import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...clones.recorders import (
    FIELDS,
    STATES,
    UNKNOWN_POLICIES,
    LineageAlleles,
    allele_chunk,
    cell_summaries,
    character_matrix,
    lineage_allele_frames,
)
from ...metadata.arguments import RecorderAlleleImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.records import RecordEncoder, RecordTable
from ...metadata.rows import read_metadata_rows_chunkwise, text_read_bytes
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
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
# Bytes that merging takes for each distinct text value that a chunk holds,
# beside twice its bytes: its Python string, its entry in the table that codes
# the text, and its rank.
_MERGE_BYTES_PER_TEXT = 150
# Bytes that writing takes per record beside the table: the int32 copies of
# its cells and files, the int8 copy of its states, and the keys that count
# each cell's integrations.
_WRITE_BYTES_PER_RECORD = 40
# Bytes that an import, a load and an export hold whatever their size: a
# block of a file as it is hashed, pandas' parser buffers, and a Zarr chunk of
# 2 MiB as it is filled and encoded.
_FIXED_BYTES = 8 * 2**20
# Bytes that a loaded record takes at the peak of building the tables: its
# stored arrays and its row of the table of records.
_TABLE_BYTES_PER_RECORD = 120
# Bytes that a loaded distinct integration, site or allele, or a cell ID,
# takes as a Python string with its entry in the index of its categories.
_TABLE_BYTES_PER_TEXT = 160
# Bytes that the character matrix takes per record at its peak: the stored
# arrays, the order of the rows, their groups, the keys of the edits and the
# codes of each record.
_EXPORT_BYTES_PER_RECORD = 260
# Bytes that a distinct allele takes while it is ranked, beside twice its
# bytes, and a character while it is named and ordered.
_EXPORT_BYTES_PER_ALLELE = 150
_EXPORT_BYTES_PER_CHARACTER = 1_000
# Bytes that writing a store takes per exported cell: a character's column
# copied as int64 and Zarr's buffer of its chunk.
_STORE_BYTES_PER_CELL = 24


def _files_text(n_files: int) -> str:
    return "the allele file" if n_files == 1 else "the allele files"


def _path_sha256(path: Any) -> str:
    return store_sha256(path) if os.path.isdir(path) else file_sha256(path)


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
        unedited one, and a blank DNA Typewriter slot unedited after the
        last edit. Text that pandas reads as missing is missing only in
        tables read through ``columns``. The clones API reference lists the
        formats and their tokens.

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
                number of slots, such as ``{"TTCACGTA": 2}``. One for every
                file, or a sequence with one per file.
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
                ``"separator"`` (sites joined in one column), ``"umis"``,
                ``"reads"``, ``"kind"`` (``"indel"``, ``"independent"`` or
                ``"ordered"``) and ``"slots"``, and the tokens
                ``"unedited"``, ``"missing"``, ``"not_applicable"`` and
                ``"unknown"``.
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
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
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
            identify=_path_sha256,
        )
        # Integrations follow the order of the paths, as given.
        items = _per_file(integrations, len(resolved))
        positions: dict[str, list[int]] = {}
        for position, path in enumerate(resolved):
            positions.setdefault(os.fspath(path), []).append(position)
        taken: dict[str, int] = {}
        layouts = []
        for file in files:
            key = os.fspath(file.path)
            # A file listed twice with other prefixes takes its listings'
            # options in order.
            position = positions[key][taken.get(key, 0)]
            taken[key] = taken.get(key, 0) + 1
            layouts.append(
                allele_layout(
                    file.path,
                    format=format,
                    columns=columns,
                    integrations=items[position],
                )
            )
        return files, layouts

    def _allele_selection_indices(self, cell_selection: ArtifactRef) -> np.ndarray:
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
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
        merging the records into one table and writing the table, beside
        what the import holds whatever its size.

        Args:
            encoder: The records held so far.
            cells: The bytes held for the selected cells, such as their lookup.
            reading: The bytes held to read a file: its chunk, what it holds
                beside and the barcodes that match no cell.
            merge_text: The bytes that merging takes for the distinct text
                values of the chunks held.
        """
        held = encoder.nbytes
        merging = max(
            _MERGE_FACTOR * held + merge_text,
            held + _WRITE_BYTES_PER_RECORD * encoder.n_records,
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
        ids_bytes = text_read_bytes(self.cells._get_array("ids"), n_cells)
        # The per-cell arrays, the lookup and a mask of each file of one
        # array's cells, which find arrays that share no cell.
        held = (LOOKUP_BYTES_PER_CELL + _CELL_BYTES + single) * n_cells + ids_bytes
        encoder = RecordEncoder(FIELDS)
        # The selected cells' share is checked before a cell ID is read.
        self._check_allele_budget(encoder, cells=held)
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )
        join = CellBarcodeJoin(
            cell_ids,
            memory_bytes=self.memoryBytes,
            held_bytes=(_CELL_BYTES + single) * n_cells + ids_bytes,
        )
        del cell_ids
        named: list[np.ndarray | None] = []
        counted: list[dict[str, int]] = []
        merge_text = 0
        for position, (file, layout) in enumerate(zip(files, layouts, strict=True)):
            mask = (
                np.zeros(n_cells, dtype=bool)
                if layout.integration is not None
                else None
            )
            counts, merge_text = self._read_allele_file(
                encoder,
                join,
                file,
                layout,
                position,
                mask=mask,
                held=held,
                merge_text=merge_text,
            )
            named.append(mask)
            counted.append(counts)
        _check_shared_cells(files, layouts, named)
        del named
        self._check_allele_budget(encoder, cells=held, merge_text=merge_text)
        table = encoder.finish()
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
        merge_text: int,
    ) -> tuple[dict[str, int], int]:
        """Read one file into the encoder.

        Returns:
            The file's ``rows``, its ``records``, its edits ``beyond`` their
            integration's length and its rows ``without_integration``; and
            the bytes that merging takes for the distinct text values of the
            chunks held, these included.
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
            merge_text=merge_text,
        )
        counts = {"rows": 0, "records": 0, "beyond": 0, "without_integration": 0}
        for rows in iter_allele_rows(layout, chunk_rows):
            classified = classified_rows(layout, rows)
            cells = join.cells(file, rows.barcodes)
            if mask is not None:
                mask[cells[cells >= 0]] = True
            frame, record_cells = allele_chunk(rows, cells, classified)
            merge_text += _merge_text_bytes(frame)
            encoder.add(frame, record_cells, position)
            counts["rows"] += len(rows)
            counts["records"] += len(record_cells)
            counts["beyond"] += classified.n_beyond
            if rows.integrations is not None:
                counts["without_integration"] += int(
                    np.count_nonzero(pd.isna(rows.integrations))
                )
            self._check_allele_budget(
                encoder,
                cells=held,
                reading=reading + join.unmatched_bytes,
                merge_text=merge_text,
            )
        return counts, merge_text

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
            and ``alleles``, and ``allele_bytes``, the bytes of the distinct
            alleles.
        """

        def length(name: str) -> int:
            return int(as_zarr_array(group[name], name=name).shape[0])

        return {
            "records": length(f"{_RECORDS}_cells"),
            "cells": length("n_edits"),
            "integrations": length(f"{_RECORDS}_integration_offsets") - 1,
            "sites": length(f"{_RECORDS}_site_offsets") - 1,
            "alleles": length(f"{_RECORDS}_allele_offsets") - 1,
            "allele_bytes": length(f"{_RECORDS}_allele_values"),
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
            ValueError: If it is not a complete ``lineage_alleles`` artifact.
            MemoryError: If the tables do not fit the memory budget.
        """
        group = self._allele_group(alleles)
        counts = self._allele_counts(group)
        n_records, n_cells = counts["records"], counts["cells"]
        n_text = counts["integrations"] + counts["sites"] + counts["alleles"]
        # The stored arrays and the tables built from them, each distinct
        # text and cell ID as a category, and the cells' IDs as they are read.
        resident = (
            _FIXED_BYTES
            + _TABLE_BYTES_PER_RECORD * n_records
            + _TABLE_BYTES_PER_TEXT * (n_text + n_cells)
            + 2 * counts["allele_bytes"]
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
        0 unedited, 1 to k its k distinct edits, the marks first, those
        written as whole numbers by value and the others by their text, then
        the unknown marks, and missing where a site is missing or not
        applicable. A state names one allele of its
        character, whatever cells are exported, and a table names each
        state's allele. Where a cell holds several allele rows of one
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
            unknown: ``"state"`` gives each distinct unknown mark of a
                character a state of its own, after the marks; ``"missing"``
                writes unknown marks as missing.

        Returns:
            The number of cells written.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``alleles`` is not a complete ``lineage_alleles``
                artifact, ``format`` or ``unknown`` is unknown, no cell
                reads a character, or, for LAML, a cell or character name
                holds a character that its files cannot hold.
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
        # allele as it is ranked, each character as it is named and ordered
        # (at most every integration at every site), the exported cells' IDs,
        # and a store's column as it is written; the matrix's codes are
        # counted once their size is known.
        resident = (
            _FIXED_BYTES
            + _EXPORT_BYTES_PER_RECORD * n_records
            + _EXPORT_BYTES_PER_ALLELE * counts["alleles"]
            + 2 * counts["allele_bytes"]
            + _EXPORT_BYTES_PER_CHARACTER
            * min(n_records, counts["integrations"] * max(1, counts["sites"]))
            + text_read_bytes(self.cells._get_array("ids"), n_cells)
            + _STORE_BYTES_PER_CELL * n_cells
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
        cell_index = read_stored_selection_indices(
            self.zw,
            inspect_artifact(self.zw, alleles).input_ref("cell_selection"),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
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

            # LAML's text is written as many cells at a time as the budget
            # leaves room for, at least one.
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
                block_cells=(self.memoryBytes - held) // row_bytes,
            )
        logger.info(
            f"Exported {len(matrix.cells)} cells by {len(matrix.characters)} "
            f"characters to {target}"
        )
        return int(len(matrix.cells))


def _merge_text_bytes(frame: pd.DataFrame) -> int:
    """The bytes that merging takes for a chunk's distinct text values."""
    total = 0
    for name in ("integration", "site", "allele"):
        distinct = pd.unique(frame[name].to_numpy(dtype=object))
        texts = [value for value in distinct.tolist() if isinstance(value, str)]
        total += _MERGE_BYTES_PER_TEXT * len(texts) + 2 * sum(
            len(value.encode("utf-8")) for value in texts
        )
    return total


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


def _stored_files(group: zarr.Group) -> list[dict[str, Any]]:
    """Read the files that a ``lineage_alleles`` artifact records."""
    files = group.attrs.get("files")
    keys = {"format", "tool", "pipeline", "kind", "integration", "rows", "records"}
    if not isinstance(files, list) or not all(
        isinstance(entry, dict) and keys <= set(entry) for entry in files
    ):
        raise ValueError("The lineage_alleles artifact has no valid list of files")
    return [dict(entry) for entry in files]
