import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...clones.barcodes import (
    FIELDS,
    BarcodeCounter,
    CountedBarcodes,
    LineageBarcodes,
    cell_summaries,
    library_summaries,
    lineage_barcode_frames,
    pair_keys,
    repeated_keys,
)
from ...clones.barcode_clones import (
    BarcodeRecords,
    allowlist_texts,
    barcode_call_settings,
    barcode_clone_calls,
)
from ...clones.assignments import CloneAssignment
from ...metadata.arguments import BarcodeCloneArguments, LineageBarcodeImportArguments
from ...metadata.artifacts import (
    cell_data_write_bytes,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.records import pack_text
from ...metadata.rows import read_text_rows, read_whole
from ...metadata.selection import (
    CellSelectionRead,
    plan_cell_selection_read,
    read_cell_selection,
)
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    artifact_group,
    fingerprint_strings,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.diagnostics import warn_diagnostics
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument, optional_bool_argument, quoted_value
from ...utils.logging import logger
from ...utils.memory_budget import MemoryLedger
from .cell_groups import group_columns, within_groups
from .cell_barcodes import (
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    texts_per_file,
    unmatched_barcode_rules,
)
from .clone_artifacts import plan_clones, write_clones
from .clone_steps import files_text, hold_ranked_clone_labels, read_codes
from .record_tables import (
    record_arrays,
    record_chunk_lengths,
    stored_records,
)

if TYPE_CHECKING:
    from ...readers.lineage_barcodes import BarcodeLayout, BarcodeRows
    from ..mapping_datastore import MappingDatastore as _BarcodeOperationsBase
else:
    _BarcodeOperationsBase = object

# The prefix of the stored record arrays, such as record_cells.
_RECORDS = "record"
# The per-cell arrays of a lineage_barcodes artifact; n_barcodes is canonical.
_CELL_ARRAYS = ("n_barcodes", "n_umis", "n_reads", "n_records")
# The attributes that an import saves and that reusing it requires.
_IMPORT_ATTRIBUTES = ("fields", "libraries")
# A chunk of a file may take a twentieth of what the budget leaves.
_CHUNK_SHARE = 20
# Bytes that the key of each row of a TREX molecule table takes at the peak
# of finding a cell and UMI that it lists twice: the key, its sorted copy and
# the comparison of neighbours.
_KEY_BYTES = 40
# Bytes that the per-cell arrays and the arrays written take per selected cell.
_CELL_BYTES = 64
# Each cell's summaries, its barcodes, UMIs, reads and records, as int32,
# int64, int64 and int32, and what counting them holds beside: each record's
# cell and pair as int64, their sorted distinct pairs and the counts' weights,
# and for each cell its sums before they become int64.
_SUMMARY_BYTES_PER_CELL = 24
_COUNTING_BYTES_PER_RECORD = 40
_COUNTING_BYTES_PER_CELL = 8
# Bytes that a loaded record takes at the peak of building the tables: its
# stored arrays and its row of the table of records.
_TABLE_BYTES_PER_RECORD = 120
# Bytes that a loaded distinct barcode, UMI or cell ID takes as a Python
# string with its entry in the index of its categories.
_TABLE_BYTES_PER_TEXT = 160
# Waiting rows merge into the records once they hold about a thousandth of
# the budget in rows, within 1,000 and 250,000 rows.
_MERGE_BUDGET_SHARE = 1_000
# The record arrays that calling clones reads as int64: each field of the
# records, the array it is stored in and what a refusal calls it.
_CALLED_RECORD_ARRAYS = (
    ("cells", "cells", "each record's cell"),
    ("sources", "sources", "each record's library"),
    ("barcodes", "barcode", "each record's barcode"),
    ("umi_codes", "umi", "each record's UMI"),
    ("umis", "umis", "each record's UMIs"),
    ("reads", "reads", "each record's reads"),
)
# Bytes of each allowlisted barcode while its distinct values are sorted and
# fingerprinted: the references to its Python string in the lists and the set
# of distinct barcodes, its packed bytes, and the string itself when a
# fixed-width array gave it.
_ALLOWLIST_BYTES = 300


def _kept_rows(rows: "BarcodeRows", cells: np.ndarray, level: str) -> np.ndarray:
    """Rows of selected cells that hold a barcode: a pair counted 0 holds none."""
    present = ~np.asarray(pd.isna(rows.barcodes), dtype=bool)
    if level == "pairs":
        if rows.umi_counts is not None:
            present &= rows.umi_counts != 0
        elif rows.reads is not None:
            present &= rows.reads != 0
    return np.asarray((cells >= 0) & present, dtype=bool)


class _LineageBarcodeOperationsMixin(_BarcodeOperationsBase):
    @validation_scoped
    def _clones_import_barcodes(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
        *,
        cell_selection: ArtifactRef,
        from_assay: str | None = None,
        barcode_prefix: str | Sequence[str | None] | None = None,
        barcode_suffix: str | Sequence[str | None] | None = None,
        barcode_pattern: str | Sequence[str | None] | None = None,
        format: str | None = None,
        columns: Mapping[str, Any] | None = None,
        library: str | Sequence[str | None] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import the expressed lineage barcodes of a set of cells.

        Lineage barcodes such as LARRY's, CellTag's and TREX's are random
        sequences that cells and their descendants transcribe, so the
        barcodes that cells share mark their common ancestors. Extraction
        tools write each cell's barcodes as reads, molecules (a cell, a UMI
        and a barcode, with the molecule's reads or without) or pairs of a
        cell and a barcode with its UMIs or reads. The import reads any of
        them, joins each row to a selected cell by its cell barcode and saves
        the barcodes as a ``lineage_barcodes`` artifact, which
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_barcodes`
        reads back.

        Each file's format is recognised from its contents: TREX's
        ``molecules.txt`` and ``molecules_corrected.txt`` of every release,
        pycashier's per-read table with or without its header, CellTag-multi's
        and the Biddy et al. workflow's per-read tables, FateMap's
        ``stepThreeStarcodeShavedReads.txt``, QuiCAT's per-sample tables and
        merged table, BARtab's ``counts.tsv`` and cell barcode annotation,
        Cell Ranger's ``molecule_info.h5``, and AnnData files of cells by
        barcodes, such as QuiCAT's ``barcodes_output.h5ad``. ``format`` set
        to ``"reads"``, ``"molecules"`` or ``"pairs"`` reads any other table
        by the columns that ``columns`` names. ``from_assay`` imports the
        counts of an assay whose features are barcodes, such as a Cell Ranger
        feature barcode library, instead of files. The clones API reference
        lists the formats and their columns.

        Rows that are reads collapse into molecules whose reads are their
        rows, or, without UMIs, into pairs of a cell and a barcode. Rows that
        name one molecule or pair of a library merge, summing their reads. A
        pair counted 0 and a row without a barcode hold none, and are counted
        in the ``emptyRows`` diagnostic.

        A cell's ID is its file's ``barcode_prefix``, the cell barcode and
        ``barcode_suffix``, matched exactly; ``barcode_pattern`` keeps the
        part of each cell barcode that its ``barcode`` group matches first.
        Cell barcodes that match no selected cell are counted in the join's
        diagnostics, which ``ds.artifacts.inspect(ref).diagnostics`` returns,
        apart from those of cells outside the selection, beside
        ``fileRows``, the files' rows, and ``unmatchedRows``, those of no
        selected cell, which count each item of a list field and each nonzero
        count of a merged table or a matrix. Barcodes of no cell of the
        store, such as droplets that cell calling rejected, are named in a
        warning, which a reused import gives again. A file is identified by
        the SHA-256 of its bytes and by its library, the file's name unless
        ``library`` names it, so a file moved to another folder reuses the
        saved barcodes, as does a renamed one whose library is named, an
        edited one gives new ones, and the order of the files does not
        matter.

        The artifact's canonical ``n_barcodes`` holds each cell's distinct
        barcodes, and ``n_umis``, ``n_reads`` and ``n_records`` its UMIs,
        reads and records, -1 where its library counts no UMIs or reads. The
        records, one per molecule or pair, are stored in the ``record_``
        arrays, ordered by cell, library, barcode and UMI.

        Args:
            paths: One barcode file, or several, plain or gzip-compressed; or
                None with ``from_assay``.
            cell_selection: The cells that the barcodes belong to.
            from_assay: An assay whose features are barcodes and whose counts
                are UMIs, read instead of files, or None.
            barcode_prefix: Text before each cell barcode in the cell IDs, such
                as ``"sample1__"`` in a merged store: one for every file, or one
                per file. None adds nothing.
            barcode_suffix: Text after each cell barcode in the cell IDs, such
                as ``"-1"``: one for every file, or one per file.
            barcode_pattern: A regular expression whose named group
                ``barcode`` is the part of each cell barcode that names the
                cell, found as ``re.search`` finds it; a cell barcode it does
                not match names no cell. One for every file, or one per file.
            format: One of ``"trex"``, ``"pycashier"``, ``"celltag"``,
                ``"fatemap"``, ``"quicat"``, ``"quicat_wide"``, ``"bartab"``,
                ``"bartab_annotation"``, ``"molecule_info"``, ``"h5ad"``,
                ``"reads"``, ``"molecules"`` and ``"pairs"``, or None to
                recognise each file's format.
            columns: For ``format="reads"``, ``"molecules"`` or ``"pairs"``,
                the columns to read: ``"cell"`` and ``"barcode"`` (a column,
                or a list of columns whose values are joined by ``"__"``, such
                as the two halves of a CloneTracker XP barcode), ``"umi"``,
                ``"umis"``, ``"reads"`` and ``"library"``, and ``"separator"``,
                one of ``";"``, ``","``, ``":"`` and ``"|"``, which splits the
                barcode and count fields of pairs into lists. For
                ``format="h5ad"``, ``{"reads": "X"}`` reads the counts as
                reads, and ``"library"`` names an ``obs`` column.
            library: The library of each file's rows, such as ``"RNA"`` and
                ``"ATAC"``: one for every file, or one per file. None names a
                file's library after its name before the first ``.``, and the
                assay after its name. For ``molecule_info``, the type of the
                library to read, such as ``"Custom"``.
            invalidate_cache: Import again instead of reusing saved barcodes.

        Returns:
            A complete datastore-scoped ``lineage_barcodes`` artifact.

        Raises:
            TypeError: If an argument has the wrong type, or the assay's
                counts are not integers.
            ValueError: If a file's format is unknown or malformed, a column
                is missing, an option lists a value for another number of
                files, a file is listed twice with the same prefix and suffix,
                a library holds molecules in one file and pairs in another,
                rows of a TREX molecule table repeat a cell and UMI, paths and
                ``from_assay`` are both given or neither, or the selection
                holds no cell.
            FileNotFoundError: If a file does not exist.
            PermissionError: If the store is read-only and the barcodes are not
                saved yet.
            MemoryError: If the import does not fit the memory budget.
        """
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if format is not None and not isinstance(format, str):
            raise TypeError("format must be a format name or None")
        files: tuple[BarcodeFile, ...] = ()
        layouts: list[BarcodeLayout] = []
        assay: Any = None
        if from_assay is None:
            files, layouts = self._barcode_layouts(
                paths,
                barcode_prefix=barcode_prefix,
                barcode_suffix=barcode_suffix,
                barcode_pattern=barcode_pattern,
                format=format,
                columns=columns,
                library=library,
            )
            sources = tuple(
                {**layout.record(), **file.record()}
                for file, layout in zip(files, layouts, strict=True)
            )
        else:
            _refuse_file_options(
                paths,
                barcode_prefix=barcode_prefix,
                barcode_suffix=barcode_suffix,
                barcode_pattern=barcode_pattern,
                format=format,
                columns=columns,
            )
            assay = self._barcode_assay(from_assay)
            sources = (self._assay_source(assay, library),)
        selection = self._barcode_selection_read(cell_selection)
        n_cells = selection.cells
        arguments = LineageBarcodeImportArguments(
            cell_selection=cell_selection,
            sources=sources,
            invalidate_cache=invalidate_cache,
        )
        if assay is not None:
            self._require_measured_cells(
                assay.name, cell_selection, operation=arguments.operation
            )
        warnings = unmatched_barcode_rules(files_text("barcode", len(files)))
        planned = self._plan_barcodes(arguments, cell_selection, n_cells)
        if planned.reused:
            # Saved barcodes warn as they did when they were imported.
            warn_diagnostics(
                arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
            )
            return planned.ref
        self._require_writable(arguments.operation)
        if assay is None:
            counted, diagnostics, ledger = self._count_files(files, layouts, selection)
        else:
            counted, diagnostics, ledger = self._count_assay(
                assay, str(sources[0]["library"]), selection
            )
        self._write_barcodes(planned, counted, n_cells, diagnostics, ledger=ledger)
        warn_diagnostics(
            arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
        )
        return planned.ref

    def _barcode_layouts(
        self,
        paths: Any,
        *,
        barcode_prefix: Any,
        barcode_suffix: Any,
        barcode_pattern: Any,
        format: str | None,
        columns: Mapping[str, Any] | None,
        library: Any,
    ) -> tuple[tuple[BarcodeFile, ...], list["BarcodeLayout"]]:
        """Check the barcode files and their options, and read each one's layout."""
        from ...readers.lineage_barcodes import barcode_layout

        if paths is None:
            raise ValueError(
                "paths must name the barcode files, or from_assay an assay whose "
                "features are barcodes"
            )
        files = barcode_files(
            paths,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            what="barcode file",
        )
        n_listed = 1 if isinstance(paths, str | os.PathLike) else len(paths)
        if library == "" or (
            isinstance(library, Sequence)
            and not isinstance(library, str)
            and any(name == "" for name in library)
        ):
            raise ValueError("library must name a library, not be empty")
        # Libraries follow the order of the paths, as given.
        libraries = texts_per_file(library, n_listed, "library", "barcode file")
        layouts = [
            barcode_layout(
                file.path,
                format=format,
                columns=columns,
                library=libraries[file.position],
            )
            for file in files
        ]
        return files, layouts

    def _count_files(
        self,
        files: Sequence[BarcodeFile],
        layouts: Sequence["BarcodeLayout"],
        selection: CellSelectionRead,
    ) -> tuple[CountedBarcodes, dict[str, Any], MemoryLedger]:
        """Count the barcodes of the files' rows that name selected cells.

        One ledger holds what the import keeps for the selected cells, their
        positions and IDs as text, the join, a chunk of each file and the
        counter's records at their merge, and is returned for the save.
        """
        n_cells = selection.cells
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Importing barcodes for {n_cells} selected cells",
            remedy="pass a smaller cell_selection",
            codec_threads=self.nthreads,
        )
        ledger.hold("each selected cell's arrays", _CELL_BYTES * n_cells)
        cell_index = read_cell_selection(
            ledger, self.zw, selection, item="the selected cells' positions"
        )
        # Only the selected cells' IDs are read, a sample's in a merged store.
        cell_ids = read_text_rows(
            ledger,
            self.cells._get_array("ids"),
            cell_index,
            item="the selected cells' IDs",
        )
        join = CellBarcodeJoin(
            cell_ids, ledger=ledger, store_ids=self.cells._get_array("ids")
        )
        counter = BarcodeCounter(merge_rows=self._merge_rows())
        n_empty = 0
        counted = 0
        for file, layout in zip(files, layouts, strict=True):
            empty, counted = self._count_file(
                counter, join, file, layout, ledger=ledger, counted=counted
            )
            n_empty += empty
        records = counter.finish()
        # The finished table replaces the records at their merge.
        ledger.hold(f"the {records.table.n_records} records", records.table.nbytes)
        ledger.release(counted)
        return (
            records,
            {
                **join.diagnostics(),
                "fileRows": join.n_rows,
                "unmatchedRows": join.n_unmatched_rows,
                "emptyRows": n_empty,
            },
            ledger,
        )

    def _barcode_selection_read(self, cell_selection: ArtifactRef) -> CellSelectionRead:
        """Check and count a cell selection, whose cells the import reads within its ledger."""
        planned = plan_cell_selection_read(self.zw, cell_selection)
        if not planned.cells:
            raise ValueError("cell_selection selects no cell")
        return planned

    def _merge_rows(self) -> int:
        return int(min(250_000, max(1_000, self.memoryBytes // _MERGE_BUDGET_SHARE)))

    def _plan_barcodes(
        self,
        arguments: LineageBarcodeImportArguments,
        cell_selection: ArtifactRef,
        n_cells: int,
    ) -> PlannedArtifact:
        record = arguments.to_record()

        def has_records(_ref: ArtifactRef, group: zarr.Group) -> bool:
            cells = group.get(f"{_RECORDS}_cells")
            fields = group.attrs.get("fields")
            return (
                isinstance(cells, zarr.Array)
                and len(cells.shape) == 1
                and fields == list(FIELDS)
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

    @staticmethod
    def _held_records(
        ledger: MemoryLedger, counter: BarcodeCounter, counted: int
    ) -> int:
        """Charge the counter's records at the peak of their merge in place of ``counted``.

        Returns:
            The bytes now charged for them.
        """
        ledger.release(counted)
        return ledger.hold(
            f"the {counter.n_rows} molecules and pairs at their merge",
            counter.peak_bytes,
        )

    def _count_file(
        self,
        counter: BarcodeCounter,
        join: CellBarcodeJoin,
        file: BarcodeFile,
        layout: "BarcodeLayout",
        *,
        ledger: MemoryLedger,
        counted: int,
    ) -> tuple[int, int]:
        """Count the barcodes of one file.

        The file's chunks take a twentieth of the ledger's room when the file
        is planned. The file's names and a chunk are held while the file is
        read, and the counter's records at their merge, ``counted`` bytes
        when the file starts, are charged again after each chunk. A file that
        lists each molecule once, as TREX's molecule tables do, is refused if
        a cell and UMI repeat on its rows.

        Returns:
            The file's rows without a barcode, and the bytes then charged for
            the counter's records.
        """
        from ...readers.lineage_barcodes import iter_barcode_rows

        # A chunk takes a twentieth of the room left when the file is planned.
        room = ledger.room
        chunk_rows = layout.chunk_rows(
            self.memoryBytes // _CHUNK_SHARE if room is None else room // _CHUNK_SHARE
        )
        # A smaller selection holds the file's names and chunks all the same.
        reading = ledger.hold(
            f"the barcode file's names and a chunk of {chunk_rows} rows",
            layout.held_bytes() + layout.chunk_bytes(chunk_rows),
            remedy="",
        )
        if layout.library_column is None:
            # The file's library is listed even if it holds no row of a cell.
            counter.add_library(layout.library, molecules=layout.umi is not None)
        keys: list[np.ndarray] | None = [] if layout.one_row_per_molecule else None
        n_keys = 0
        n_empty = 0
        for rows in iter_barcode_rows(layout, chunk_rows):
            if keys is not None:
                ledger.hold(
                    "the keys of the file's molecules",
                    _KEY_BYTES * len(rows.cells),
                    remedy="",
                )
                keys.append(pair_keys(rows.cells, rows.umis))
                n_keys += len(rows.cells)
            cells = join.cells(file, rows.cells)
            keep = _kept_rows(rows, cells, layout.level)
            n_empty += int(np.count_nonzero((cells >= 0) & ~keep))
            n_kept = int(np.count_nonzero(keep))
            libraries = (
                np.full(n_kept, layout.library, dtype=object)
                if rows.libraries is None
                else rows.libraries[keep]
            )
            counter.add(
                cells[keep],
                libraries,
                rows.barcodes[keep],
                umis=None if rows.umis is None else rows.umis[keep],
                umi_counts=None if rows.umi_counts is None else rows.umi_counts[keep],
                reads=None if rows.reads is None else rows.reads[keep],
                rows_are_reads=layout.level == "reads",
            )
            del rows, cells, keep, libraries
            counted = self._held_records(ledger, counter, counted)
        if keys is not None:
            repeated = repeated_keys(keys)
            if repeated:
                raise ValueError(
                    f"{os.fspath(file.path)} lists {repeated} cells and UMIs on "
                    "several rows, as TREX's reads.txt lists each read; import "
                    "molecules.txt or molecules_corrected.txt, which list each "
                    "molecule once"
                )
            del keys
            ledger.release(_KEY_BYTES * n_keys)
        ledger.release(reading)
        return n_empty, counted

    def _barcode_assay(self, from_assay: Any) -> Any:
        """Return the assay whose features are barcodes, after checking its counts."""
        if not isinstance(from_assay, str):
            raise TypeError("from_assay must be an assay name or None")
        assay = self._get_assay(from_assay)
        count_dtype = np.dtype(assay.rawData.dtype)
        if count_dtype.kind not in "iu":
            raise TypeError(
                f"The barcode counts of assay {quoted_value(assay.name)} must be integers, not "
                f"{count_dtype}"
            )
        return assay

    def _assay_source(self, assay: Any, library: Any) -> dict[str, Any]:
        """Identify an assay's barcode counts: its name, data and feature IDs."""
        if library is not None and not isinstance(library, str):
            raise TypeError("library must be text or None with from_assay")
        if library == "":
            raise ValueError("library must name a library, not be empty")
        feature_ids = np.asarray(assay.feats.fetch_all("ids"), dtype=object)
        return {
            "assay": assay.name,
            "library": library or assay.name,
            "dataset_fingerprint": self._ensure_dataset_fingerprint(assay.name),
            "feature_ids_fingerprint": fingerprint_strings(feature_ids.astype(str)),
        }

    def _count_assay(
        self, assay: Any, library: str, selection: CellSelectionRead
    ) -> tuple[CountedBarcodes, dict[str, Any], MemoryLedger]:
        """Count the barcodes that an assay counts as its features, as pairs, and return the ledger for the save."""
        feature_ids = np.asarray(assay.feats.fetch_all("ids"), dtype=object)
        counter = BarcodeCounter(merge_rows=self._merge_rows())
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Importing the barcodes of {selection.cells} selected cells",
            remedy="pass a smaller cell_selection",
            codec_threads=self.nthreads,
        )
        ledger.hold("each selected cell's arrays", _CELL_BYTES * selection.cells)
        cell_index = read_cell_selection(
            ledger, self.zw, selection, item="the selected cells' positions"
        )
        counted = 0
        blocks = assay.rawData[cell_index, :]._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Reading barcode counts",
            prefetch=None,
            row_mask=None,
            resident_bytes=ledger.held_bytes,
            memory_bytes=self.memoryBytes,
        )
        counter.add_library(library, molecules=False)
        start = 0
        n_rows = 0
        for raw in blocks:
            block = np.asarray(raw)
            rows, columns = np.nonzero(block)
            n_rows += len(rows)
            counter.add(
                start + rows,
                np.full(len(rows), library, dtype=object),
                feature_ids[columns],
                umi_counts=block[rows, columns].astype(np.int64),
            )
            counted = self._held_records(ledger, counter, counted)
            start += block.shape[0]
        records = counter.finish()
        # The finished table replaces the records at their merge.
        ledger.hold(f"the {records.table.n_records} records", records.table.nbytes)
        ledger.release(counted)
        return (
            records,
            # An assay counts the barcodes of the store's own cells.
            {
                "fileBarcodes": 0,
                "unmatchedBarcodes": 0,
                "unselectedBarcodes": 0,
                "unmatchedBarcodeExamples": [],
                "unselectedBarcodeExamples": [],
                "fileRows": n_rows,
                "unmatchedRows": 0,
                "emptyRows": 0,
            },
            ledger,
        )

    def _write_barcodes(
        self,
        planned: PlannedArtifact,
        counted: CountedBarcodes,
        n_cells: int,
        diagnostics: Mapping[str, Any],
        *,
        ledger: MemoryLedger,
    ) -> None:
        """Save the counted records and each cell's summaries, charging both and the write to ``ledger``."""
        table = counted.table
        ledger.hold("each cell's summaries", _SUMMARY_BYTES_PER_CELL * n_cells)
        ledger.peak(
            f"counting each cell's summaries of {table.n_records} records",
            _COUNTING_BYTES_PER_RECORD * table.n_records
            + _COUNTING_BYTES_PER_CELL * n_cells,
        )
        summaries = cell_summaries(table, n_cells)
        libraries = library_summaries(counted)
        arrays = {
            "n_barcodes": summaries["n_barcodes"],
            "n_umis": summaries["n_umis"],
            "n_reads": summaries["n_reads"],
            "n_records": summaries["n_records"],
            **record_arrays(table, _RECORDS),
        }
        chunk_lengths = record_chunk_lengths(
            {
                name: values
                for name, values in arrays.items()
                if name.startswith(_RECORDS)
            }
        )
        ledger.peak(
            f"writing {table.n_records} records and {n_cells} cells",
            cell_data_write_bytes(
                {name: (values.shape, values.dtype) for name, values in arrays.items()},
                chunk_lengths=chunk_lengths,
            ),
        )
        write_cell_data_artifact(
            self.zw,
            planned,
            arrays,
            chunk_lengths=chunk_lengths,
            attributes={"fields": list(FIELDS), "libraries": libraries},
            diagnostics=diagnostics,
        )
        n_with = int(np.count_nonzero(summaries["n_barcodes"]))
        logger.info(
            f"Imported {table.n_records} molecules and barcode pairs of {n_with} of "
            f"{n_cells} cells, in {len(libraries)} libraries"
        )

    @validation_scoped
    def _clones_load_barcodes(self, barcodes: ArtifactRef) -> LineageBarcodes:
        """Read saved lineage barcodes as tables of cells, records and libraries.

        Args:
            barcodes: A ``lineage_barcodes`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_barcodes`.

        Returns:
            The barcodes, with one row per cell, one row per molecule or pair,
            and one row per library.

        Raises:
            TypeError: If ``barcodes`` is not an ArtifactRef.
            ValueError: If it is not a ``lineage_barcodes`` artifact.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
            MemoryError: If the tables, about 120 bytes per record, 160 per
                cell and per distinct barcode and UMI, and 64 per cell, or the
                read of the cells' IDs and values, which
                ``artifacts.load_values`` checks, do not fit the memory budget.
        """
        group = self._barcode_group(barcodes)
        n_records = int(
            as_zarr_array(group[f"{_RECORDS}_cells"], name="record_cells").shape[0]
        )
        n_cells = int(as_zarr_array(group["n_barcodes"], name="n_barcodes").shape[0])
        n_text = sum(
            int(as_zarr_array(group[f"{_RECORDS}_{name}_offsets"], name=name).shape[0])
            for name in ("barcode", "umi")
        )
        MemoryLedger(
            self.memoryBytes,
            f"Loading the barcodes of {n_cells} cells",
            codec_threads=self.nthreads,
        ).peak(
            f"the tables of {n_cells} cells and {n_records} molecules and pairs",
            _TABLE_BYTES_PER_RECORD * n_records
            + _TABLE_BYTES_PER_TEXT * (n_text + n_cells)
            + _CELL_BYTES * n_cells,
        )
        loaded = self._artifacts_load_values(barcodes)
        table = stored_records(group, _RECORDS, FIELDS)
        summaries = {
            "n_barcodes": np.asarray(loaded.values),
            **{name: read_array(group, name) for name in _CELL_ARRAYS[1:]},
        }
        return lineage_barcode_frames(
            loaded.cell_ids, summaries, table, _stored_libraries(group)
        )

    @validation_scoped
    def _clones_barcode_clones(
        self,
        barcodes: ArtifactRef,
        *,
        preset: str | None = None,
        min_reads: int | None = None,
        max_distance: int | None = None,
        metric: str | None = None,
        allowlist: Sequence[str] | bool | None = None,
        min_umis: int | None = None,
        min_fraction: float | None = None,
        barcodes_per_cell: tuple[int, int | None] | None = None,
        method: str | None = None,
        similarity: float | tuple[float, float] | None = None,
        inclusive: bool | None = None,
        prune: bool | None = None,
        doublets: bool | None = None,
        trex_rules: bool | None = None,
        min_size: int | None = None,
        within: str | Sequence[str] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call clones from imported lineage barcodes.

        Cells that carry the same lineage barcodes descend from one labelled
        cell. The barcodes of each cell pass the rules of the method that
        made them, in this order: molecules with fewer than ``min_reads``
        reads are dropped; barcodes within ``max_distance`` of each other by
        ``metric`` merge, within each group and library, into the most
        abundant of their connected component, ties going to the barcode
        that sorts last; barcodes outside ``allowlist`` are dropped; a cell's
        barcodes of fewer than ``min_umis`` UMIs, or below ``min_fraction`` of
        its top barcode's UMIs, are dropped, in each library apart; cells
        with fewer or more barcodes than ``barcodes_per_cell`` allows get no
        clone. Then ``method="sets"`` makes the cells of identical barcode
        sets a clone, and ``method="jaccard"`` links cells whose Jaccard
        index of barcode sets exceeds ``similarity``, or reaches it with
        ``inclusive``; ``prune`` removes TREX's bridges and ``doublets`` its
        doublets, and the connected components are the clones. Clones of
        fewer than ``min_size`` cells leave their cells without one.

        ``preset`` sets every rule to a method's published values: ``"larry"``
        (LARRY), ``"trex"`` (TREX), ``"celltag"`` (CellTagR) or
        ``"celltag_multi"`` (CellTag-multi). An argument given replaces the
        preset's value, and 0 turns a count filter off. Without a preset,
        ``method`` is required, ``"jaccard"`` needs ``similarity`` and
        ``inclusive``, and ``max_distance`` needs a ``metric``. The clones API
        reference lists each preset's rules, which
        ``cytearc.clones.BARCODE_PRESETS`` holds.

        ``preset="trex"`` also applies TREX's rules for its partial cloneIDs:
        molecules of low complexity are dropped, cloneIDs are compared where
        both hold a base and need 20 such bases, and a one-UMI barcode is
        dropped where another cell of its group holds it, as is a barcode
        whose only molecule has one read. ``trex_rules`` left None follows
        the preset, and True or False turns these rules on or off in any
        call, so a call with and without them shows what they remove.

        Counts are read in each library's unit: UMIs, or reads in a library
        that counts no UMIs, such as scATAC-seq CellTags. A rule that needs a
        count that a library does not hold refuses rather than guess.

        Clones are numbered by size, so ``clone_1`` is the largest, and ties
        follow their first cell. The ``clones`` artifact also holds each
        cell's ``n_barcodes``, its barcodes after the pair filters, and
        ``doublet``, whether TREX's rule removed it as a doublet, and its
        parameters the resolved rules. Its diagnostics, which
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_clones` and
        ``ds.artifacts.inspect(clones).diagnostics`` return, show when
        correction merges unrelated barcodes: ``distinctBarcodes``, the
        distinct barcodes of each group and library that correction
        compares; ``expectedChanceMatches``, the pairs of them expected
        within ``max_distance`` by chance, None under ``levenshtein``
        beyond 8 edits, whose count takes too long; ``mergedBarcodes``,
        those that correction merged into another; then what each rule
        removed or found, None for a rule that did not run, which the clones
        API reference lists; and, as every source's clones,
        ``cellsWithClone``, the cells given a clone, and
        ``largestCloneShare``, the share of them that the largest clone
        holds.

        Args:
            barcodes: A ``lineage_barcodes`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.ClonesAccessor.import_barcodes`.
            preset: ``"larry"``, ``"trex"``, ``"celltag"``,
                ``"celltag_multi"`` or None.
            min_reads: The fewest reads of a molecule.
            max_distance: The largest distance at which barcodes merge, 0 for
                no correction.
            metric: ``"hamming"`` or ``"levenshtein"``.
            allowlist: The barcodes of the library, a sequence of text; the
                CellTag presets require one, and False calls them without.
            min_umis: The fewest UMIs, or reads, of a cell's barcode.
            min_fraction: The share of the UMIs of a cell's top barcode that
                its other barcodes must reach, from 0 to 1.
            barcodes_per_cell: The fewest and the most barcodes of a called
                cell, the most None for no limit.
            method: ``"sets"`` or ``"jaccard"``.
            similarity: The Jaccard index that links two cells, from 0 to 1;
                or two, within one library and across libraries.
            inclusive: Whether an index equal to the similarity links.
            prune: Remove the edges whose ends share no neighbour.
            doublets: Remove the cells whose neighbours fall apart without
                them.
            trex_rules: Apply TREX's rules for partial cloneIDs: None with
                ``preset="trex"`` alone, or True or False in any call.
            min_size: The fewest cells of a clone.
            within: Cell metadata columns whose labels keep clones apart, such
                as the mouse; every cell with a barcode needs a label.
            invalidate_cache: Call again instead of reusing saved clones.

        Returns:
            A complete datastore-scoped ``clones`` artifact aligned to the
            barcodes' cells. Read it with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_clones`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``barcodes`` is not a ``lineage_barcodes`` artifact,
                the preset is unknown, a required rule is missing, a value is
                out of range, TREX's rules meet another metric than Hamming's, an
                argument does not apply to the method, a rule needs counts
                that a library does not hold, or a within column is missing,
                is named twice or has no label for a cell with a barcode.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If the calling does not fit the memory budget.
        """
        inclusive = optional_bool_argument(inclusive, "inclusive")
        prune = optional_bool_argument(prune, "prune")
        doublets = optional_bool_argument(doublets, "doublets")
        trex_rules = optional_bool_argument(trex_rules, "trex_rules")
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        group = self._barcode_group(barcodes)
        stored_counts = as_zarr_array(group["n_barcodes"], name="n_barcodes")
        n_cells = int(stored_counts.shape[0])
        # The cells' selection is checked once, outside the budget, before
        # anything is charged; the grouping and saving reuse the check.
        plan_cell_selection_read(
            self.zw, inspect_artifact(self.zw, barcodes).input_ref("cell_selection")
        )
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Calling clones from the barcodes of {n_cells} cells",
            codec_threads=self.nthreads,
        )
        listed = None
        allowlist_bytes = 0
        if allowlist is not None and not isinstance(allowlist, bool | np.bool_):
            if isinstance(allowlist, str | bytes) or not hasattr(allowlist, "__len__"):
                raise TypeError(
                    "allowlist must be a sequence of barcodes, False or None"
                )
            # Its distinct barcodes are sorted and fingerprinted; the barcodes of
            # a fixed-width array become Python strings first.
            characters = (
                len(allowlist) * (allowlist.dtype.itemsize // 4)
                if isinstance(allowlist, np.ndarray) and allowlist.dtype.kind == "U"
                else 0
            )
            ledger.peak(
                f"an allowlist of {len(allowlist)} barcodes",
                _ALLOWLIST_BYTES * len(allowlist) + characters,
            )
            listed = allowlist_texts(allowlist)
            allowlist_bytes = ledger.hold(
                "the allowlist", _ALLOWLIST_BYTES * len(listed)
            )
        settings = barcode_call_settings(
            preset,
            min_reads=min_reads,
            max_distance=max_distance,
            metric=metric,
            allowlist=allowlist,
            min_umis=min_umis,
            min_fraction=min_fraction,
            barcodes_per_cell=barcodes_per_cell,
            method=method,
            similarity=similarity,
            inclusive=inclusive,
            prune=prune,
            doublets=doublets,
            min_size=min_size,
            trex_rules=trex_rules,
        )
        columns = group_columns(within, "within")
        cell_selection = inspect_artifact(self.zw, barcodes).input_ref("cell_selection")
        allowed = None
        allowlist_fingerprint = None
        if settings.allowlist:
            assert listed is not None
            allowed = listed
            builder = ValueFingerprintBuilder()
            data, offsets = pack_text(allowed)
            builder.update_array("data", data)
            builder.update_array("offsets", offsets)
            allowlist_fingerprint = builder.hexdigest()
            del data, offsets, builder
        # Every cell with a barcode needs a label in the within columns.
        counts = read_whole(ledger, group, "n_barcodes", item="each cell's barcodes")
        flags = ledger.hold("the cells with a barcode", n_cells)
        with_barcodes = counts > 0
        del counts
        ledger.release(n_cells * stored_counts.dtype.itemsize)
        # Each cell's group and the groups' keys stay held until the clones
        # are saved.
        group_codes, _keys, within_fingerprint = within_groups(
            self.zw,
            self.cells,
            cell_selection,
            columns,
            with_barcodes,
            ledger=ledger,
            what="a barcode",
        )
        del with_barcodes
        ledger.release(flags)
        record = settings.record()
        arguments = BarcodeCloneArguments(
            barcodes=barcodes,
            within_fingerprint=within_fingerprint,
            allowlist_fingerprint=allowlist_fingerprint,
            preset=record["preset"],
            min_reads=record["min_reads"],
            max_distance=record["max_distance"],
            metric=record["metric"],
            min_umis=record["min_umis"],
            min_fraction=record["min_fraction"],
            barcodes_per_cell=settings.barcodes_per_cell,
            method=record["method"],
            similarity=(
                settings.similarity
                if settings.across is None or settings.similarity is None
                else (settings.similarity, settings.across)
            ),
            inclusive=record["inclusive"],
            prune=record["prune"],
            doublets=record["doublets"],
            min_size=record["min_size"],
            within=columns,
            trex_rules=record["trex_rules"],
            invalidate_cache=invalidate_cache,
        )
        planned = plan_clones(
            self.zw,
            arguments,
            cell_selection,
            n_cells,
            with_probability=False,
            required_attributes=("n_clones", "library_units"),
            source_arrays={"n_barcodes": "i", "doublet": "b"},
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        records = BarcodeRecords(
            **{
                field: read_codes(ledger, group, f"{_RECORDS}_{name}", item=item)
                for field, name, item in _CALLED_RECORD_ARRAYS
            },
            barcode_data=read_whole(
                ledger,
                group,
                f"{_RECORDS}_barcode_values",
                item="the bytes of the distinct barcodes",
            ).astype(np.uint8, copy=False),
            barcode_offsets=read_codes(
                ledger,
                group,
                f"{_RECORDS}_barcode_offsets",
                item="the offsets of the distinct barcodes",
            ),
            libraries=tuple(
                (str(entry["library"]), str(entry["level"]))
                for entry in _stored_libraries(group)
            ),
        )
        called = barcode_clone_calls(
            records,
            n_cells,
            groups=group_codes,
            settings=settings,
            allowlist=allowed,
            ledger=ledger,
        )
        record_bytes = records.nbytes
        del records, allowed, listed
        ledger.release(record_bytes + allowlist_bytes)
        calls = called.calls
        labels = hold_ranked_clone_labels(ledger, calls.n_clones)
        assignment = CloneAssignment(
            codes=calls.codes,
            clone_ids=labels,
            clone_sizes=calls.sizes,
            probabilities=None,
            n_cells_in_several_clones=0,
        )
        write_clones(
            self.zw,
            planned,
            assignment,
            {
                "n_clones": calls.n_clones,
                "library_units": [
                    {"library": name, "unit": unit}
                    for (name, _level), unit in zip(
                        (
                            (str(entry["library"]), str(entry["level"]))
                            for entry in _stored_libraries(group)
                        ),
                        called.units,
                        strict=True,
                    )
                ],
            },
            source_arrays={
                "n_barcodes": called.n_barcodes,
                "doublet": calls.doublets,
            },
            ledger=ledger,
            diagnostics=called.diagnostics,
        )
        n_assigned = int(np.count_nonzero(calls.codes >= 0))
        largest = int(calls.sizes[0]) if calls.n_clones else 0
        logger.info(
            f"Called {calls.n_clones} clones of {n_assigned} of {n_cells} cells from "
            f"their lineage barcodes; the largest holds {largest} cells"
        )
        return planned.ref

    def _barcode_group(self, barcodes: ArtifactRef) -> zarr.Group:
        """Return a complete ``lineage_barcodes`` artifact's group."""
        if not isinstance(barcodes, ArtifactRef):
            raise TypeError("barcodes must be an ArtifactRef")
        if barcodes.kind != LineageBarcodeImportArguments.artifact_kind:
            raise ValueError(
                f"barcodes must be a lineage_barcodes artifact, not {quoted_value(barcodes.kind)}"
            )
        require_complete_artifact(self.zw, barcodes)
        return artifact_group(self.zw, barcodes)


def _refuse_file_options(paths: Any, **options: Any) -> None:
    """Refuse the options of files beside ``from_assay``, which reads none."""
    if paths is not None:
        raise ValueError(
            "Pass paths to import barcode files, or from_assay to import an "
            "assay's barcode counts, not both"
        )
    given = sorted(name for name, value in options.items() if value is not None)
    if given:
        raise ValueError(
            "from_assay reads the store's own cells and features, so "
            + ", ".join(given)
            + " would change nothing; pass them with paths"
        )


def _stored_libraries(group: zarr.Group) -> list[dict[str, Any]]:
    """Read the libraries that a ``lineage_barcodes`` artifact records."""
    libraries = group.attrs.get("libraries")
    keys = {"library", "level", "records", "has_umis", "has_reads"}
    if not isinstance(libraries, list) or not all(
        isinstance(entry, dict) and keys <= set(entry) for entry in libraries
    ):
        raise ValueError("The lineage_barcodes artifact has no valid list of libraries")
    return [dict(entry) for entry in libraries]
