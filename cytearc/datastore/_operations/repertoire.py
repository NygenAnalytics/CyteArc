import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import ImmuneReceptorImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.records import (
    RAGGED,
    TEXT,
    EncodedText,
    RecordTable,
    chunk_rows_for,
    decode_ragged,
    decode_text,
)
from ...metadata.rows import read_metadata_rows_chunkwise, text_read_bytes
from ...metadata.selection import cell_selection_indices
from ...repertoire.chains import RECEPTOR_TYPES, index_chains, receptor_quality
from ...repertoire.contigs import FIELDS, contig_encoder
from ...repertoire.receptors import (
    SLOT_FIELDS,
    ImmuneReceptors,
    contig_frame,
    receptor_cell_frame,
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
from ...utils.logging import logger
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    warn_unmatched_barcodes,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _RepertoireOperationsBase
else:
    _RepertoireOperationsBase = object

# The per-cell arrays of an immune_receptors artifact, besides chain_slots.
_CELL_ARRAYS = (
    "values",
    "receptor_subtype",
    "chain_pairing",
    "multichain",
    "n_vj",
    "n_vdj",
    "n_contigs",
)
# The attributes that an import saves and that reusing it requires: the
# stored fields, the contigs of the files and those that match no selected
# cell, and what the warning about unmatched barcodes names.
_IMPORT_ATTRIBUTES = (
    "fields",
    "n_file_contigs",
    "n_unmatched_contigs",
    "n_file_barcodes",
    "n_unmatched_barcodes",
    "unmatched_barcode_examples",
)
# At its peak, while it orders the contigs, ranks the chains and writes the
# artifact, an import holds up to three times the bytes of its encoded
# contigs beside what it holds for each selected cell, as measured on
# Cell Ranger files with and without region sequences and on AIRR files with
# sequences and alignments.
_PEAK_PER_ENCODED_BYTE = 3
# Bytes that an import holds for each selected cell beside the join's lookup
# of cell IDs and their text, which the join counts, and the read of the IDs,
# which text_read_bytes counts: the selection, each cell's chain slots,
# counts and labels, and the fixed-width text of the labels it writes.
# With the lookup and the IDs of 12 characters, up to 290 bytes per cell were
# resident at the peak on 200,000 and 1,000,000 cells, and tracemalloc traces
# about 320.
_IMPORT_BYTES_PER_CELL = 300
# Bytes that a decoded contig field takes in a table, per contig, besides the
# text of ragged fields.
_TABLE_BYTES_PER_FIELD = 64
# Bytes per cell of the table of cells without its slot fields: the cell IDs,
# the labels and chain counts as read and as table columns, and the slots.
# Up to 560 bytes per cell were resident at the peak, on 1,000,000 cells.
_CELL_TABLE_BYTES_PER_CELL = 700
# The bytes of each chunk of a contig array. Packed text and per-contig codes
# are read and written whole, which chunks of megabytes make fast.
_CONTIG_CHUNK_BYTES = 2 * 2**20


@dataclass(frozen=True, slots=True)
class _ContigSource:
    """One contig file of an import, its format and the columns of its header."""

    file: BarcodeFile
    format: str
    n_columns: int

    def record(self) -> dict[str, Any]:
        """Return the file's identity: its format, bytes and barcode options."""
        return {"format": self.format, **self.file.record()}


def _contig_sources(
    paths: Any,
    *,
    barcode_prefix: Any,
    barcode_suffix: Any,
    barcode_pattern: Any,
    format: Any,
) -> tuple[_ContigSource, ...]:
    """Check the files of an import and their options, in canonical order.

    Each file's format is ``format``, or is read from its header when
    ``format`` is None.
    """
    from ...readers.airr import AIRR_FORMAT, airr_columns, contig_file_format
    from ...readers.cellranger import CELLRANGER_VDJ_FORMAT, vdj_contig_columns

    formats = (CELLRANGER_VDJ_FORMAT, AIRR_FORMAT)
    if format is not None and not isinstance(format, str):
        raise TypeError("format must be a format name or None")
    if format is not None and format not in formats:
        raise ValueError(
            "format must be "
            + " or ".join(repr(name) for name in formats)
            + f", or None to read each file's format from its header, not {format!r}"
        )
    files = barcode_files(
        paths,
        barcode_prefix=barcode_prefix,
        barcode_suffix=barcode_suffix,
        barcode_pattern=barcode_pattern,
        what="contig file",
    )
    sources = []
    for file in files:
        file_format = format or contig_file_format(file.path)
        header = (
            vdj_contig_columns(file.path)
            if file_format == CELLRANGER_VDJ_FORMAT
            else airr_columns(file.path)
        )
        sources.append(
            _ContigSource(file=file, format=file_format, n_columns=len(header))
        )
    return tuple(sources)


def _iter_source(
    source: _ContigSource, chunk_rows: int
) -> Iterator[tuple[int, pd.DataFrame]]:
    from ...readers.cellranger import CELLRANGER_VDJ_FORMAT, iter_vdj_contigs

    if source.format == CELLRANGER_VDJ_FORMAT:
        return iter_vdj_contigs(source.file.path, chunk_rows=chunk_rows)
    from ...readers.airr import iter_airr_rearrangements

    return iter_airr_rearrangements(source.file.path, chunk_rows=chunk_rows)


def _encode_contigs(
    sources: Sequence[_ContigSource],
    join: CellBarcodeJoin,
    *,
    memory_bytes: int,
    held_bytes: int,
) -> RecordTable:
    """Read the contig files and encode the contigs of the selected cells.

    Each contig joins the selected cell that its barcode names, through
    ``join``, which counts the contigs and barcodes of the files and those
    that match no selected cell. ``held_bytes`` are held beside the encoded
    contigs.

    Raises:
        MemoryError: If the import's peak would exceed ``memory_bytes``.
        ValueError: If a contig of a selected cell has no ID, or an ID repeats
            among the contigs of selected cells within one file.
    """
    encoder = contig_encoder()
    for position, source in enumerate(sources):
        chunk_rows = chunk_rows_for(memory_bytes, columns=source.n_columns)
        for _first_line, chunk in _iter_source(source, chunk_rows):
            rows = join.cells(source.file, chunk["cell_id"].to_numpy(dtype=object))
            matched = rows >= 0
            encoder.add(
                chunk.loc[matched].drop(columns="cell_id"), rows[matched], position
            )
            peak = _PEAK_PER_ENCODED_BYTE * encoder.nbytes + held_bytes
            if peak > memory_bytes:
                raise MemoryError(
                    f"Importing the contigs of {encoder.n_records} matched rows "
                    f"holds about {peak} bytes at its peak, more than the memory "
                    f"budget of {memory_bytes} bytes. Raise mem_budget."
                )
    if encoder.missing_ids:
        raise ValueError(
            f"{encoder.missing_ids} contigs of the selected cells have no "
            "sequence_id (contig_id in Cell Ranger's files)"
        )
    repeated = encoder.repeated_ids()
    if repeated:
        position, name = repeated[0]
        raise ValueError(
            f"Contig ID {name!r} repeats in "
            f"{os.fspath(sources[position].file.path)!r}; each contig of a file "
            "needs its own ID"
        )
    return encoder.finish()


def _contig_arrays(table: RecordTable) -> dict[str, np.ndarray]:
    """Return the stored arrays of encoded contigs, keyed by array name.

    A text field ``f`` is stored as ``contig_f``, each contig's code, with the
    UTF-8 bytes of its distinct values in ``contig_f_values`` and their
    offsets in ``contig_f_offsets``. A ragged field is stored as
    ``contig_f``, the bytes of each contig's value, with ``contig_f_offsets``.
    Other fields are stored as ``contig_f``.
    """
    arrays: dict[str, np.ndarray] = {
        "contig_cells": np.asarray(table.cells, dtype=np.int32),
        "contig_sources": np.asarray(table.sources, dtype=np.int32),
    }
    # contig_cells, contig_sources and contig_chain are not fields.
    for name, kind in table.kinds.items():
        if kind == TEXT:
            encoded = table.text[name]
            arrays[f"contig_{name}"] = np.asarray(encoded.codes, dtype=np.int32)
            arrays[f"contig_{name}_values"] = encoded.data
            arrays[f"contig_{name}_offsets"] = encoded.offsets
        elif kind == RAGGED:
            data, offsets = table.ragged[name]
            arrays[f"contig_{name}"] = data
            arrays[f"contig_{name}_offsets"] = offsets
        else:
            arrays[f"contig_{name}"] = np.asarray(table.values[name])
    return arrays


def _contig_chunk_lengths(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    """Chunk each contig array in about 2 MiB, for whole-array reads and writes."""
    return {
        name: max(1, _CONTIG_CHUNK_BYTES // max(1, values.dtype.itemsize))
        for name, values in arrays.items()
    }


def _read(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[:])


def _stored_text(group: zarr.Group, name: str) -> EncodedText:
    return EncodedText(
        codes=_read(group, f"contig_{name}"),
        data=_read(group, f"contig_{name}_values"),
        offsets=_read(group, f"contig_{name}_offsets"),
    )


def _stored_column(group: zarr.Group, name: str) -> np.ndarray:
    """Decode one stored contig field: text, None where missing."""
    kind = FIELDS[name]
    if kind == TEXT:
        encoded = _stored_text(group, name)
        return decode_text(encoded.codes, encoded.categories)
    if kind == RAGGED:
        return decode_ragged(
            _read(group, f"contig_{name}"), _read(group, f"contig_{name}_offsets")
        )
    return _read(group, f"contig_{name}")


def _stored_fields(group: zarr.Group) -> tuple[str, ...]:
    fields = group.attrs["fields"]
    if not isinstance(fields, list) or not all(name in FIELDS for name in fields):
        raise ValueError("The immune_receptors artifact has no valid list of fields")
    return tuple(str(name) for name in fields)


def _files_text(n_files: int) -> str:
    return "the contig file" if n_files == 1 else "the contig files"


class _RepertoireOperationsMixin(_RepertoireOperationsBase):
    @validation_scoped
    def _repertoire_import_receptors(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        cell_selection: ArtifactRef,
        barcode_prefix: str | Sequence[str | None] | None = None,
        barcode_suffix: str | Sequence[str | None] | None = None,
        barcode_pattern: str | Sequence[str | None] | None = None,
        format: str | None = None,
        productive_only: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import the immune receptor contigs of a set of cells.

        A V(D)J library, such as 10x Genomics Chromium Single Cell Immune
        Profiling, assembles each cell's T or B cell receptor transcripts into
        contigs. The import reads Cell Ranger's contig annotations,
        ``filtered_contig_annotations.csv`` or ``all_contig_annotations.csv``,
        or AIRR Rearrangement files with a ``cell_id`` field, such as Cell
        Ranger's ``airr_rearrangement.tsv``; each file's format is read from
        its header unless ``format`` names it. It joins each contig to a
        selected cell by barcode, keeps every contig of the selected cells,
        and saves them as an ``immune_receptors`` artifact. Pass the TCR and
        BCR files of one library together, so a cell with both kinds of chain
        is found.

        Each cell's chains are ranked as scirpy ranks them, as the repertoire
        API reference describes: a contig counts as a chain when it has a
        locus of a receptor chain and a junction amino-acid sequence, is
        productive unless ``productive_only`` is False, and is not flagged by
        Cell Ranger as outside a cell or of low confidence. The first two
        chains of each arm fill the cell's ``VJ_1``, ``VJ_2``, ``VDJ_1`` and
        ``VDJ_2`` slots, and the chains give the cell's receptor type, subtype
        and chain pairing. A selected cell without a chain is ``"no IR"``.

        A contig's cell ID is its file's ``barcode_prefix``, its barcode and
        ``barcode_suffix``, matched exactly; ``barcode_pattern`` keeps the
        part of each barcode that its ``barcode`` group matches first. A
        merged store prefixes each sample's barcodes with its name and two
        underscores, so pass each sample's files with that sample's prefix.
        Contigs and barcodes that match no selected cell are counted, saved
        in the artifact's ``n_unmatched_contigs``, ``n_unmatched_barcodes``
        and ``unmatched_barcode_examples`` attributes beside
        ``n_file_contigs`` and ``n_file_barcodes``, and named in a warning,
        which a reused import gives again. A file is identified by the
        SHA-256 of its bytes, so a moved file reuses the saved receptors and
        an edited file gives new ones, and the order of the files does not
        matter.

        The artifact's canonical ``values`` hold each cell's receptor type,
        and its other per-cell arrays are ``receptor_subtype``,
        ``chain_pairing``, ``multichain``, ``n_vj`` and ``n_vdj`` (its chains
        of each arm) and ``n_contigs``. ``select_cells`` reads them, so
        ``ds.select_cells(receptors, include=["TCR"])`` keeps the T cells
        with a receptor.

        Args:
            paths: One contig file, or several, plain or gzip-compressed.
            cell_selection: The cells that the contigs belong to.
            barcode_prefix: Text before each barcode in the cell IDs, such as
                ``"sample1__"`` in a merged store: one for every file, or one
                per file. None adds nothing.
            barcode_suffix: Text after each barcode in the cell IDs, such as
                ``"-1"``: one for every file, or one per file.
            barcode_pattern: A regular expression whose named group
                ``barcode`` is the part of each barcode that names the cell,
                found as ``re.search`` finds it, such as
                ``r"^(?P<barcode>[ACGT]{16})"``; a barcode it does not match
                names no cell. One for every file, or one per file.
            format: ``"cellranger"`` or ``"airr"`` for every file, or None to
                read each file's format from its header.
            productive_only: Count only productive contigs as chains.
            invalidate_cache: Import again instead of reusing saved receptors.

        Returns:
            A complete datastore-scoped ``immune_receptors`` artifact. Read it
            with
            :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.load_receptors`
            or plot it with ``ds.plots.receptor_qc``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a file is neither format or not the named one, is
                malformed, repeats a contig ID, or is listed twice with the
                same prefix and suffix, an option lists a value for another
                number of files, a pattern has no ``barcode`` group, or the
                selection holds no cell.
            FileNotFoundError: If a file does not exist.
            PermissionError: If the store is read-only and the receptors are
                not saved yet.
            MemoryError: If the import would not fit the memory budget: at
                its peak it holds about three times the bytes of its encoded
                contigs, 400 bytes per selected cell and each cell's ID as
                text, and reading the IDs decodes a stored chunk of them.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        bool_argument(productive_only, "productive_only")
        bool_argument(invalidate_cache, "invalidate_cache")
        sources = _contig_sources(
            paths,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            format=format,
        )
        cell_index = cell_selection_indices(self.zw, cell_selection)
        n_cells = len(cell_index)
        if not n_cells:
            raise ValueError("cell_selection selects no cell")
        arguments = ImmuneReceptorImportArguments(
            cell_selection=cell_selection,
            sources=tuple(source.record() for source in sources),
            productive_only=productive_only,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()

        def has_contigs(_ref: ArtifactRef, group: zarr.Group) -> bool:
            cells = group.get("contig_cells")
            fields = group.attrs.get("fields")
            return (
                isinstance(cells, zarr.Array)
                and len(cells.shape) == 1
                and isinstance(fields, list)
                and all(f"contig_{name}" in group for name in fields)
            )

        planned = plan_cell_data_artifact(
            self.zw,
            scope="datastore",
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={
                "values": ((n_cells,), None),
                "receptor_subtype": ((n_cells,), None),
                "chain_pairing": ((n_cells,), None),
                "multichain": ((n_cells,), "b"),
                "n_vj": ((n_cells,), "i"),
                "n_vdj": ((n_cells,), "i"),
                "n_contigs": ((n_cells,), "i"),
                "chain_slots": ((n_cells, 4), "i"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=_IMPORT_ATTRIBUTES,
            reuse_validator=has_contigs,
        )
        files_text = _files_text(len(sources))
        if planned.reused:
            # Saved receptors warn as they did when they were imported.
            warn_unmatched_barcodes(
                artifact_group(self.zw, planned.ref).attrs, files_text
            )
            return planned.ref
        self._require_writable(arguments.operation)
        # What the import holds for the selected cells, and the read of their
        # IDs as text, are checked before the IDs are read.
        reading = (
            LOOKUP_BYTES_PER_CELL + _IMPORT_BYTES_PER_CELL
        ) * n_cells + text_read_bytes(self.cells._get_array("ids"), n_cells)
        if reading > self.memoryBytes:
            raise MemoryError(
                f"Importing contigs for {n_cells} selected cells holds about "
                f"{reading} bytes before the first contig is read, more than the "
                f"memory budget of {self.memoryBytes} bytes. Raise mem_budget, or "
                "pass a smaller cell_selection."
            )
        # Only the selected cells' IDs are read, a sample's in a merged store.
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )
        join = CellBarcodeJoin(
            cell_ids,
            memory_bytes=self.memoryBytes,
            held_bytes=_IMPORT_BYTES_PER_CELL * n_cells,
        )
        # Beside the encoded contigs, the import holds what the join counts:
        # its lookup, the cells' IDs as text and the bytes held for each cell.
        table = _encode_contigs(
            sources, join, memory_bytes=self.memoryBytes, held_bytes=join.held_bytes
        )
        # Files without a contig row give no locus field and no contig.
        locus = (
            table.text["locus"]
            if "locus" in table.text
            else EncodedText.from_categories(np.empty(0, dtype=np.int32), [])
        )
        index = index_chains(
            table, n_cells, locus=locus, productive_only=productive_only
        )
        quality = receptor_quality(index, locus)
        n_contigs = np.bincount(table.cells, minlength=n_cells).astype(np.int32)
        contig_arrays = {"contig_chain": index.eligible, **_contig_arrays(table)}
        attributes = {
            "fields": list(table.fields),
            "n_file_contigs": join.n_rows,
            "n_unmatched_contigs": join.n_unmatched_rows,
            **join.attributes(),
        }
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": quality.receptor_type,
                "receptor_subtype": quality.receptor_subtype,
                "chain_pairing": quality.chain_pairing,
                "multichain": index.multichain,
                "n_vj": index.n_vj,
                "n_vdj": index.n_vdj,
                "n_contigs": n_contigs,
                "chain_slots": index.slots,
                **contig_arrays,
            },
            chunk_lengths=_contig_chunk_lengths(contig_arrays),
            attributes=attributes,
        )
        warn_unmatched_barcodes(attributes, files_text)
        counts = pd.Series(quality.receptor_type).value_counts()
        logger.info(
            f"Imported {table.n_records} contigs of {n_cells} cells: "
            + ", ".join(f"{int(counts.get(name, 0))} {name}" for name in RECEPTOR_TYPES)
        )
        return planned.ref

    @validation_scoped
    def _repertoire_load_receptors(
        self,
        receptors: ArtifactRef,
        *,
        fields: Sequence[str] | None = None,
    ) -> ImmuneReceptors:
        """Read saved immune receptors as a table of cells and a table of contigs.

        Args:
            receptors: An ``immune_receptors`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.import_receptors`.
            fields: The contig fields to load, by AIRR name, or None for
                every stored field. The table of contigs holds them, and the
                table of cells holds those of ``SLOT_FIELDS`` for each slot;
                ``fields=[]`` loads the labels and chain counts of each cell
                alone.

        Returns:
            The receptors, with one row per cell and one row per contig.

        Raises:
            TypeError: If ``receptors`` is not an ArtifactRef or ``fields`` is
                not a sequence of names.
            ValueError: If it is not an ``immune_receptors`` artifact, or a
                field is not stored.
            MemoryError: If the tables do not fit the datastore memory budget.
        """
        group, stored = self._receptor_group(receptors)
        if fields is None:
            chosen = stored
        else:
            if isinstance(fields, str) or not isinstance(fields, Sequence):
                raise TypeError("fields must be a sequence of field names or None")
            unknown = [name for name in fields if name not in stored]
            if unknown:
                raise ValueError(
                    "The receptors store no field "
                    + ", ".join(repr(str(name)) for name in unknown)
                    + "; they store "
                    + ", ".join(stored)
                )
            chosen = tuple(name for name in stored if name in set(fields))
        contig_cells = _read(group, "contig_cells")
        n_contigs = len(contig_cells)
        n_cells = int(as_zarr_array(group["values"], name="values").shape[0])
        sequence_bytes = sum(
            int(as_zarr_array(group[f"contig_{name}"], name=name).shape[0])
            for name in chosen
            if FIELDS[name] == RAGGED
        )
        slot_columns = 4 * sum(name in SLOT_FIELDS for name in chosen)
        resident = (
            _TABLE_BYTES_PER_FIELD * n_contigs * (len(chosen) + 3)
            + 2 * sequence_bytes
            + (_CELL_TABLE_BYTES_PER_CELL + _TABLE_BYTES_PER_FIELD * slot_columns)
            * n_cells
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"The tables of {n_cells} cells and {n_contigs} contigs with "
                f"{len(chosen)} fields hold about {resident} bytes, more than the "
                f"memory budget of {self.memoryBytes} bytes. Pass fields= with "
                "fewer fields, or raise mem_budget."
            )
        # The canonical values come with the cell IDs; the other per-cell
        # arrays are aligned to the same cells.
        loaded = self._artifacts_load_values(receptors)
        cell_ids = loaded.cell_ids
        arrays = {name: _read(group, name) for name in _CELL_ARRAYS[1:]}
        slots = _read(group, "chain_slots")
        columns = {name: _stored_column(group, name) for name in chosen}
        chains = contig_frame(
            cell_ids,
            contig_cells=contig_cells,
            slots=slots,
            eligible=_read(group, "contig_chain").astype(bool),
            columns={name: columns[name] for name in chosen},
        )
        cells = receptor_cell_frame(
            cell_ids,
            receptor_type=np.asarray(loaded.values),
            receptor_subtype=arrays["receptor_subtype"],
            chain_pairing=arrays["chain_pairing"],
            n_vj=arrays["n_vj"],
            n_vdj=arrays["n_vdj"],
            n_contigs=arrays["n_contigs"],
            slots=slots,
            columns=columns,
            fields=chosen,
        )
        parameters = inspect_artifact(self.zw, receptors).parameters or {}
        return ImmuneReceptors(
            cells=cells,
            chains=chains,
            fields=stored,
            productive_only=bool(parameters["productive_only"]),
        )

    def _receptor_group(
        self, receptors: ArtifactRef
    ) -> tuple[zarr.Group, tuple[str, ...]]:
        """Return a complete ``immune_receptors`` artifact's group and its fields."""
        if not isinstance(receptors, ArtifactRef):
            raise TypeError("receptors must be an ArtifactRef")
        if receptors.kind != ImmuneReceptorImportArguments.artifact_kind:
            raise ValueError(
                "receptors must be an immune_receptors artifact, not "
                f"{receptors.kind!r}"
            )
        require_complete_artifact(self.zw, receptors)
        group = artifact_group(self.zw, receptors)
        return group, _stored_fields(group)
