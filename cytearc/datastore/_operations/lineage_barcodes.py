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
)
from ...metadata.arguments import LineageBarcodeImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.rows import read_metadata_rows_chunkwise
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_strings,
    require_complete_artifact,
)
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.logging import logger
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    texts_per_file,
    warn_unmatched_barcodes,
)
from .record_tables import (
    read_array,
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
_IMPORT_ATTRIBUTES = (
    "fields",
    "libraries",
    "n_file_rows",
    "n_unmatched_rows",
    "n_empty_rows",
    "n_file_barcodes",
    "n_unmatched_barcodes",
    "unmatched_barcode_examples",
)
# Bytes that parsing holds for each field of a chunk's rows: the raw text, its
# stripped copy and the parsed values.
_PARSE_BYTES_PER_FIELD = 1_000
# A chunk may take a twentieth of the budget, within 1 and 50,000 rows.
_MAX_CHUNK_ROWS = 50_000
# Bytes that the per-cell arrays and the arrays written take per selected cell.
_CELL_BYTES = 64
# Bytes that a loaded record takes at the peak of building the tables: its
# stored arrays and its row of the table of records, as measured on 5,000,000
# pairs and 2,000,000 molecules.
_TABLE_BYTES_PER_RECORD = 120
# Bytes that a loaded distinct barcode, UMI or cell ID takes as a Python
# string with its entry in the index of its categories.
_TABLE_BYTES_PER_TEXT = 160
# Waiting rows merge into the records once they hold about a thousandth of
# the budget in rows, within 1,000 and 250,000 rows.
_MERGE_BUDGET_SHARE = 1_000


def _chunk_rows(memory_bytes: int, n_columns: int) -> int:
    rows = int(memory_bytes) // (20 * _PARSE_BYTES_PER_FIELD * max(1, n_columns))
    return int(min(_MAX_CHUNK_ROWS, max(1, rows)))


def _files_text(n_files: int) -> str:
    return "the barcode file" if n_files == 1 else "the barcode files"


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
        in the ``n_empty_rows`` attribute.

        A cell's ID is its file's ``barcode_prefix``, the cell barcode and
        ``barcode_suffix``, matched exactly; ``barcode_pattern`` keeps the
        part of each cell barcode that its ``barcode`` group matches first.
        Cell barcodes that match no selected cell, such as droplets that cell
        calling rejected, are counted, saved in the artifact's
        ``n_file_barcodes``, ``n_unmatched_barcodes`` and
        ``unmatched_barcode_examples`` attributes beside ``n_file_rows`` and
        ``n_unmatched_rows``, and named in a warning, which a reused import
        gives again. A file is identified by the SHA-256 of its bytes, so a
        moved file reuses the saved barcodes, an edited one gives new ones,
        and the order of the files does not matter.

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
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
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
        cell_index = self._barcode_selection_indices(cell_selection)
        n_cells = len(cell_index)
        arguments = LineageBarcodeImportArguments(
            cell_selection=cell_selection,
            sources=sources,
            invalidate_cache=invalidate_cache,
        )
        if assay is not None:
            self._require_measured_cells(
                assay.name, cell_selection, operation=arguments.operation
            )
        files_text = _files_text(len(files))
        planned = self._plan_barcodes(arguments, cell_selection, n_cells)
        if planned.reused:
            # Saved barcodes warn as they did when they were imported.
            warn_unmatched_barcodes(
                artifact_group(self.zw, planned.ref).attrs, files_text
            )
            return planned.ref
        self._require_writable(arguments.operation)
        if assay is None:
            counted, attributes = self._count_files(files, layouts, cell_index)
        else:
            counted, attributes = self._count_assay(
                assay, str(sources[0]["library"]), cell_index
            )
        self._write_barcodes(planned, counted, n_cells, attributes)
        warn_unmatched_barcodes(attributes, files_text)
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
        cell_index: np.ndarray,
    ) -> tuple[CountedBarcodes, dict[str, Any]]:
        """Count the barcodes of the files' rows that name selected cells."""
        n_cells = len(cell_index)
        # Only the selected cells' IDs are read, a sample's in a merged store.
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )
        held = (LOOKUP_BYTES_PER_CELL + _CELL_BYTES) * n_cells
        join = CellBarcodeJoin(
            cell_ids, memory_bytes=self.memoryBytes, held_bytes=_CELL_BYTES * n_cells
        )
        counter = BarcodeCounter(merge_rows=self._merge_rows())
        n_empty = 0
        molecule_tables: list[str] = []
        for file, layout in zip(files, layouts, strict=True):
            n_empty += self._count_file(counter, join, file, layout, held=held)
            if layout.one_row_per_molecule:
                molecule_tables.append(layout.library)
        counted = counter.finish()
        _refuse_repeated_molecules(counted, molecule_tables)
        return counted, {
            "n_file_rows": join.n_rows,
            "n_unmatched_rows": join.n_unmatched_rows,
            "n_empty_rows": n_empty,
            **join.attributes(),
        }

    def _barcode_selection_indices(self, cell_selection: ArtifactRef) -> np.ndarray:
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

    def _check_barcode_budget(self, held: int, counter: BarcodeCounter) -> None:
        peak = held + counter.peak_bytes
        if peak > self.memoryBytes:
            raise MemoryError(
                f"Importing barcodes holds {counter.n_rows} molecules and pairs, "
                f"about {peak} bytes at their peak, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget, or pass a smaller "
                "cell_selection."
            )

    def _count_file(
        self,
        counter: BarcodeCounter,
        join: CellBarcodeJoin,
        file: BarcodeFile,
        layout: "BarcodeLayout",
        *,
        held: int,
    ) -> int:
        """Count the barcodes of one file and return its rows without a barcode."""
        from ...readers.lineage_barcodes import iter_barcode_rows

        chunk_rows = _chunk_rows(self.memoryBytes, layout.n_columns)
        chunk_bytes = _PARSE_BYTES_PER_FIELD * layout.n_columns * chunk_rows
        if layout.library_column is None:
            # The file's library is listed even if it holds no row of a cell.
            counter.add_library(layout.library, molecules=layout.umi is not None)
        n_empty = 0
        for rows in iter_barcode_rows(layout, chunk_rows):
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
            self._check_barcode_budget(held + chunk_bytes, counter)
        return n_empty

    def _barcode_assay(self, from_assay: Any) -> Any:
        """Return the assay whose features are barcodes, after checking its counts."""
        if not isinstance(from_assay, str):
            raise TypeError("from_assay must be an assay name or None")
        assay = self._get_assay(from_assay)
        count_dtype = np.dtype(assay.rawData.dtype)
        if count_dtype.kind not in "iu":
            raise TypeError(
                f"The barcode counts of assay {assay.name!r} must be integers, not "
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
        self, assay: Any, library: str, cell_index: np.ndarray
    ) -> tuple[CountedBarcodes, dict[str, Any]]:
        """Count the barcodes that an assay counts as its features, as pairs."""
        feature_ids = np.asarray(assay.feats.fetch_all("ids"), dtype=object)
        counter = BarcodeCounter(merge_rows=self._merge_rows())
        held = _CELL_BYTES * len(cell_index)
        blocks = assay.rawData[cell_index, :]._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Reading barcode counts",
            prefetch=None,
            row_mask=None,
            resident_bytes=held,
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
            self._check_barcode_budget(held, counter)
            start += block.shape[0]
        return counter.finish(), {
            "n_file_rows": n_rows,
            "n_unmatched_rows": 0,
            "n_empty_rows": 0,
            "n_file_barcodes": 0,
            "n_unmatched_barcodes": 0,
            "unmatched_barcode_examples": [],
        }

    def _write_barcodes(
        self,
        planned: PlannedArtifact,
        counted: CountedBarcodes,
        n_cells: int,
        attributes: Mapping[str, Any],
    ) -> None:
        table = counted.table
        summaries = cell_summaries(table, n_cells)
        libraries = library_summaries(counted)
        arrays = record_arrays(table, _RECORDS)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "n_barcodes": summaries["n_barcodes"],
                "n_umis": summaries["n_umis"],
                "n_reads": summaries["n_reads"],
                "n_records": summaries["n_records"],
                **arrays,
            },
            chunk_lengths=record_chunk_lengths(arrays),
            attributes={
                "fields": list(FIELDS),
                "libraries": libraries,
                **attributes,
            },
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
            ValueError: If it is not a complete ``lineage_barcodes`` artifact.
            MemoryError: If the tables do not fit the memory budget.
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
        resident = (
            _TABLE_BYTES_PER_RECORD * n_records
            + _TABLE_BYTES_PER_TEXT * (n_text + n_cells)
            + _CELL_BYTES * n_cells
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"The tables of {n_cells} cells and {n_records} molecules and pairs "
                f"hold about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
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

    def _barcode_group(self, barcodes: ArtifactRef) -> zarr.Group:
        """Return a complete ``lineage_barcodes`` artifact's group."""
        if not isinstance(barcodes, ArtifactRef):
            raise TypeError("barcodes must be an ArtifactRef")
        if barcodes.kind != LineageBarcodeImportArguments.artifact_kind:
            raise ValueError(
                f"barcodes must be a lineage_barcodes artifact, not {barcodes.kind!r}"
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


def _refuse_repeated_molecules(counted: CountedBarcodes, libraries: list[str]) -> None:
    """Refuse molecule tables, such as TREX's, that list one cell and UMI twice.

    TREX's ``reads.txt`` has the header of its early ``molecules.txt`` but
    lists each read, so its cells repeat their UMIs.
    """
    if not libraries:
        return
    table = counted.table
    positions = [
        position
        for position, name in enumerate(counted.libraries)
        if name in set(libraries)
    ]
    held = np.isin(np.asarray(table.sources), positions)
    if not held.any():
        return
    cells = np.asarray(table.cells)[held]
    sources = np.asarray(table.sources)[held]
    umis = np.asarray(table.text["umi"].codes)[held]
    keys = np.stack([cells, sources, umis])
    _unique, counts = np.unique(keys, axis=1, return_counts=True)
    repeated = int(np.count_nonzero(counts > 1)) + int(
        np.count_nonzero(counted.rows[held] > 1)
    )
    if repeated:
        raise ValueError(
            f"{repeated} cells and UMIs of a TREX molecule table appear on several "
            "rows, as each read of TREX's reads.txt does; import molecules.txt or "
            "molecules_corrected.txt, which list each molecule once"
        )
