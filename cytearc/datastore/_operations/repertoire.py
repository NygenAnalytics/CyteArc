import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import ClonotypeArguments, ImmuneReceptorImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.records import (
    RAGGED,
    TEXT,
    EncodedText,
    RecordTable,
    chunk_rows_for,
    decode_ragged,
    decode_text,
    take_text,
    unpack_text,
)
from ...metadata.rows import (
    array_row_selection_parts,
    read_array_rows_chunkwise,
    read_metadata_rows_chunkwise,
    text_read_bytes,
)
from ...metadata.selection import cell_selection_indices
from ...repertoire.chains import (
    NO_RECEPTOR,
    RECEPTOR_TYPES,
    SLOTS,
    index_chains,
    receptor_quality,
)
from ...repertoire.clonotypes import (
    CLUSTER_SEARCH_BYTES,
    DUAL_IR,
    METRICS,
    RECEPTOR_ARMS,
    REPRESENTATIVE_FIELDS,
    SEQUENCES,
    Clonotypes,
    assign_clonotype_clusters,
    assign_clonotypes,
    clonotype_frames,
    clonotype_labels,
    representative_cells,
    shared_text,
)
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
from ...storage.geometry import array_geometry
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument, integer_argument
from ...utils.logging import logger
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    warn_unmatched_barcodes,
)
from .cell_groups import cell_groups, group_columns

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
# Reading the receptor types holds each cell's stored text and, while it is
# coded, a Python string and a code for each cell, 92 bytes beside the text as
# traced on 200,000 cells.
_RECEPTOR_TYPE_CODING_BYTES = 100
# Defining clonotypes holds the chains of every slot, their distinct
# configurations and keys, and the labels: 340 bytes per cell at the peak, as
# traced on 1,000,000 simulated cells of about two chains each. Cells with more
# chains give more keys, so the check reserves 1,000 bytes per cell, beside the
# stored code and its int64 copy of each contig's compared field.
_CLONOTYPE_BYTES_PER_CELL = 1_000
_CODE_BYTES = 8
# Loading clonotypes holds, beside the cells' IDs as text, up to 300 bytes per
# cell: each cell's codes and sizes as read and in the table, and the
# receptors' slots and types that the representatives are taken from.
_CLONOTYPE_TABLE_BYTES_PER_CELL = 300
# Each clonotype holds its label and its row of the tables, 348 bytes while the
# tables are built, as traced on 50,000 clonotypes of one cell each with two
# and with four chains, and the codes of its four slots' contigs in each
# representative field, 32 bytes as held and 96 while the field is decoded.
_CLONOTYPE_TABLE_BYTES_PER_CLONOTYPE = 700
# Each representative field's packed text and offsets are read whole, with the
# bounds of every value, and each value that a representative holds becomes a
# string: its header and pointers, and up to four bytes per character as held
# and three more copies of its bytes while it is decoded.
_FIELD_BYTES_PER_VALUE = 25
_FIELD_BYTES_PER_DECODED_VALUE = 64
_FIELD_BYTES_PER_PACKED_BYTE = 7
# Values are decoded a batch of this many at a time, each holding about 160
# bytes of Python objects while its batch is decoded.
_DECODE_BATCH_VALUES = 2**16
_DECODE_BYTES_PER_BATCH_VALUE = 160
# Each junction that is decoded, such as one that names a clonotype in a
# figure or one that clusters compare, while it is decoded: its bounds as read
# and as Python integers, its object and its string's header; and each of its
# bytes as read, joined and decoded, and up to four bytes per character in its
# string. Once decoded, each keeps its bounds, its object and its string.
_JUNCTION_BYTES_PER_VALUE = 200
_JUNCTION_BYTES_PER_BYTE = 7
_KEPT_JUNCTION_BYTES_PER_VALUE = 100
_KEPT_JUNCTION_BYTES_PER_BYTE = 4
# The columns of the tables of clonotypes, which a within column may not share.
_CLONOTYPE_COLUMNS = frozenset(
    {"clonotype", "clone_size", "n_cells", "fraction", "n_configurations"}
    | {"receptor_type"}
    | {f"{slot}_{name}" for slot in SLOTS for name in REPRESENTATIVE_FIELDS}
)


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


def _stored_text_at(group: zarr.Group, name: str, positions: np.ndarray) -> np.ndarray:
    """Decode a text field of some contigs, reading only their distinct values.

    Args:
        group: An ``immune_receptors`` artifact's group.
        name: A text field that the artifact stores.
        positions: Contig positions, -1 for none.

    Returns:
        The text of each position, None for -1 or a missing value.
    """
    positions = np.asarray(positions, dtype=np.int64)
    codes = np.full(len(positions), -1, dtype=np.int64)
    present = positions >= 0
    if present.any():
        stored = as_zarr_array(group[f"contig_{name}"], name=name)
        codes[present] = np.asarray(stored.get_coordinate_selection(positions[present]))
    values = np.full(len(positions), None, dtype=object)
    needed, inverse = np.unique(codes[codes >= 0], return_inverse=True)
    if len(needed):
        # The packed bytes are read once; only the needed values become text,
        # a batch at a time, so the objects that decoding makes for each value
        # are held for one batch.
        data = _read(group, f"contig_{name}_values")
        offsets = _read(group, f"contig_{name}_offsets")
        text = np.empty(len(needed), dtype=object)
        for start in range(0, len(needed), _DECODE_BATCH_VALUES):
            batch = needed[start : start + _DECODE_BATCH_VALUES]
            text[start : start + len(batch)] = take_text(data, offsets, batch)
        values[codes >= 0] = text[inverse.reshape(-1)]
    return values


def _slot_codes(group: zarr.Group, name: str, slots: np.ndarray) -> np.ndarray:
    """Return a text field's code in each cell's slots, -1 for an empty slot."""
    codes = _read(group, f"contig_{name}").astype(np.int64)
    taken = np.full(slots.shape, -1, dtype=np.int64)
    filled = slots >= 0
    taken[filled] = codes[slots[filled]]
    return taken


@dataclass(slots=True)
class _Reads:
    """What a sequence of reads holds, checked against the budget before each read.

    Attributes:
        limit: The memory budget.
        held: The bytes held so far, by the caller and by earlier reads.
        what: What the reads are for, named in errors.
        remedy: What to do when a read does not fit, named in errors.
    """

    limit: int
    held: int
    what: str
    remedy: str

    def reserve(self, needed: int) -> None:
        """Raise MemoryError if ``needed`` more bytes would exceed the budget."""
        if self.held + needed > self.limit:
            raise MemoryError(
                f"{self.what} holds about {self.held + needed} bytes, more than the "
                f"memory budget of {self.limit} bytes. {self.remedy}"
            )

    def rows(self, source: zarr.Group, name: str, wanted: np.ndarray) -> np.ndarray:
        """Read distinct rows of an array a chunk at a time, keeping them held."""
        array = as_zarr_array(source[name], name=name)
        fixed, per_row = array_row_selection_parts(array)
        self.reserve(fixed + len(wanted) * per_row)
        values = np.asarray(read_array_rows_chunkwise(array, wanted))
        self.held += int(values.nbytes)
        return values


def _decoded_values(
    reads: _Reads, group: zarr.Group, name: str, codes: np.ndarray
) -> np.ndarray:
    """Decode the stored values of some codes of a contig text field.

    The bounds of the values are read a chunk at a time, or whole where that
    holds less, and their bytes a chunk at a time, so values that no code
    names are never decoded. Each read, and the decoding, is checked first,
    and the decoded values stay counted in ``reads``.

    Args:
        reads: What the caller holds, and the budget.
        group: An ``immune_receptors`` artifact's group.
        name: The text field, such as ``junction_aa``.
        codes: Distinct codes of the field in ascending order.

    Returns:
        The value of each code, as an object array of text.
    """
    offsets = as_zarr_array(group[f"contig_{name}_offsets"], name=name)
    values = as_zarr_array(group[f"contig_{name}_values"], name=name)
    bound_rows = np.union1d(codes, codes + 1)
    fixed, per_row = array_row_selection_parts(offsets)
    serial = fixed + len(bound_rows) * per_row
    whole = 2 * int(offsets.nbytes) + fixed
    if whole < serial:
        # Most values are named: reading the bounds whole holds less.
        reads.reserve(whole)
        stored = np.asarray(offsets[:], dtype=np.int64)
        starts, ends = stored[codes], stored[codes + 1]
        del stored
        reads.held += int(starts.nbytes + ends.nbytes)
    else:
        bounds = reads.rows(group, f"contig_{name}_offsets", bound_rows)
        starts = bounds[np.searchsorted(bound_rows, codes)].astype(np.int64)
        ends = bounds[np.searchsorted(bound_rows, codes + 1)].astype(np.int64)
    n_bytes = int((ends - starts).sum())
    geometry = array_geometry(values)
    chunk = 0 if geometry is None else geometry.nominalChunkBytes()
    reads.reserve(
        array_row_selection_parts(values)[0]
        + chunk
        + _JUNCTION_BYTES_PER_VALUE * len(codes)
        + _JUNCTION_BYTES_PER_BYTE * n_bytes
    )
    lengths = np.zeros(len(codes) + 1, dtype=np.int64)
    np.cumsum(ends - starts, out=lengths[1:])
    text = unpack_text(_byte_ranges(values, starts, ends), lengths)
    reads.held += (
        _KEPT_JUNCTION_BYTES_PER_VALUE * len(codes)
        + _KEPT_JUNCTION_BYTES_PER_BYTE * n_bytes
    )
    return text


def _byte_ranges(array: Any, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """The bytes of ascending, disjoint ranges of a one-dimensional array, joined.

    Each physical chunk that a range touches is read once, as a slice, so the
    read holds one chunk and its slice beside the bytes it returns.
    """
    length = int(array.shape[0])
    geometry = array_geometry(array)
    step = max(1, length) if geometry is None else geometry.axisChunk(0)
    joined = np.empty(int((ends - starts).sum()), dtype=np.uint8)
    block = np.empty(0, dtype=np.uint8)
    block_start = -1
    at = 0
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        position = start
        while position < end:
            chunk_start = position // step * step
            if chunk_start != block_start:
                del block
                block = np.asarray(
                    array[chunk_start : min(chunk_start + step, length)],
                    dtype=np.uint8,
                )
                block_start = chunk_start
            taken = min(end, chunk_start + step) - position
            offset = position - chunk_start
            joined[at : at + taken] = block[offset : offset + taken]
            at += taken
            position += taken
    return joined


def _sorted_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted distinct labels and each label's code among them.

    The codes equal those of ``np.unique(values, return_inverse=True)``, by
    hashing rather than by sorting every label.
    """
    codes, distinct = pd.factorize(np.asarray(values), sort=True)
    return np.asarray(distinct), np.asarray(codes, dtype=np.int64)


def _labels_at(labels: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Each code's label, None for -1, one string for each distinct label."""
    taken = np.full(len(codes), None, dtype=object)
    present = codes >= 0
    taken[present] = shared_text(labels[codes[present]])
    return taken


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

    def _within_groups(
        self,
        cell_selection: ArtifactRef,
        columns: tuple[str, ...],
        with_receptor: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, str]], str | None]:
        """Group the cells with a receptor by their labels in the within columns.

        The groups are those of :func:`cell_groups`, in the natural order of
        their labels, and are identified by its fingerprint of each cell's
        group.

        Returns:
            Each cell's group, 0 for a cell without a receptor; each group's
            labels by column; and the fingerprint of the cells' groups.
            Without columns, every cell is in one group without labels, and
            the fingerprint is None.

        Raises:
            ValueError: If a column is not a cell metadata column, or a cell
                with a receptor has no label in one.
            MemoryError: If grouping the cells does not fit the memory budget.
        """
        n_cells = len(with_receptor)
        if not columns:
            return np.zeros(n_cells, dtype=np.int64), [{}], None
        for column in columns:
            if column not in self.cells.columns:
                raise ValueError(
                    f"within names {column!r}, which is not a cell metadata column"
                )
        grouped = cell_groups(
            self.zw,
            self.cells,
            cell_selection,
            with_receptor,
            group_by=columns,
            groups=None,
            cell_selection=None,
            memory_bytes=self.memoryBytes,
            reserved=_CLONOTYPE_COLUMNS,
        )
        unlabelled = int(np.count_nonzero(with_receptor & (grouped.codes < 0)))
        if unlabelled:
            raise ValueError(
                f"The within columns have no label for {unlabelled} cells with a "
                "receptor. Every cell with a receptor needs a label in the within "
                "columns."
            )
        return np.maximum(grouped.codes, 0), grouped.keys, grouped.fingerprint()

    def _artifact_cell_ids(self, artifact: ArtifactRef) -> np.ndarray:
        """The IDs of a cell-aligned artifact's cells, in the order of its arrays."""
        selection = inspect_artifact(self.zw, artifact).input_ref("cell_selection")
        cell_index = cell_selection_indices(self.zw, selection)
        return np.asarray(read_metadata_rows_chunkwise(self.cells, "ids", cell_index))

    @validation_scoped
    def _repertoire_clonotypes(
        self,
        receptors: ArtifactRef,
        *,
        sequence: str = "nt",
        receptor_arms: str = "all",
        dual_ir: str = "any",
        same_v_gene: bool = False,
        same_j_gene: bool = False,
        within: str | Sequence[str] | None = None,
        within_receptor_type: bool = True,
        metric: str = "identity",
        cutoff: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Group the cells of imported receptors into clonotypes.

        Two cells share a clonotype when the junctions of their chains are
        identical, as scirpy's ``define_clonotypes`` decides, and clonotypes
        are the connected groups that these links form. ``sequence`` compares
        the junctions' nucleotides (``"nt"``) or residues (``"aa"``);
        ``receptor_arms`` the arms that must agree: ``"all"`` (both arms when
        both cells have both, else the one arm that each has), ``"any"``,
        ``"VJ"`` or ``"VDJ"``; ``dual_ir`` the chains of an arm: ``"any"`` (one
        chain of each cell), ``"all"`` (every chain, so both cells need the
        same number) or ``"primary_only"``. ``same_v_gene`` and ``same_j_gene``
        also require matched chains to share their gene calls.

        Cells of different receptor types, such as T cells and B cells, never
        share a clonotype unless ``within_receptor_type`` is False, and cells
        with different labels in the ``within`` cell metadata columns, such as
        the donor, never do. A cell without a receptor, or without a compared
        sequence among the chains that take part, has no clonotype.

        ``metric`` links chains whose junctions are similar instead, as
        scirpy's ``define_clonotype_clusters`` does: ``"hamming"`` (at most
        ``cutoff`` substitutions between junctions of one length),
        ``"normalized_hamming"`` (substitutions as a percentage of the length,
        rounded half up, at most ``cutoff``) or ``"levenshtein"`` (at most
        ``cutoff`` substitutions, insertions and deletions), under the same
        rules for arms, dual chains, genes and groups. The resulting
        clonotype clusters group T cells that likely share their epitope, or
        B cells of one clone despite somatic mutations. ``cutoff`` has no
        default for these metrics, since it decides the clusters, and a
        cutoff of 0 links identical junctions alone, as in scirpy.

        Clonotypes are numbered by size, so ``clonotype_1`` is the largest;
        ties follow the receptor type in sorted order, then the ``within``
        labels in natural order, then the sorted chains, so the numbers do
        not depend on the order of the cells. The artifact's canonical ``values``
        hold each cell's clonotype label, empty for a cell without one, which
        groupings treat as missing; ``clone_size`` holds the cells of each
        cell's clonotype. The clone summaries of ``ds.clones``, such as
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.diversity`, read
        the artifact as they read clones, and measure the cells of each
        receptor type apart.

        Args:
            receptors: An ``immune_receptors`` artifact.
            sequence: ``"nt"`` or ``"aa"``.
            receptor_arms: ``"all"``, ``"any"``, ``"VJ"`` or ``"VDJ"``.
            dual_ir: ``"any"``, ``"all"`` or ``"primary_only"``.
            same_v_gene: Require matched chains to share their V gene call.
            same_j_gene: Require matched chains to share their J gene call.
            within: Cell metadata columns whose labels keep clonotypes apart,
                such as the donor; every cell with a receptor needs a label.
            within_receptor_type: Keep the clonotypes of each receptor type
                apart.
            metric: ``"identity"``, ``"hamming"``, ``"normalized_hamming"`` or
                ``"levenshtein"``.
            cutoff: The largest distance of linked chains for the distance
                metrics: substitutions, percent, or edits; None with
                ``"identity"``.
            invalidate_cache: Define again instead of reusing saved clonotypes.

        Returns:
            A complete datastore-scoped ``clonotypes`` artifact aligned to the
            receptors' cells. Read it with
            :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.load_clonotypes`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``receptors`` is not an ``immune_receptors``
                artifact, an option is unknown, a distance metric has no
                cutoff, the receptors store no junction of the compared kind,
                or a ``within`` column is not a cell metadata column or has
                no label for a cell with a receptor.
            KeyError: If the store holds no such receptors.
            RuntimeError: If the receptors are incomplete.
            PermissionError: If the store is read-only and the clonotypes are
                not saved yet.
            MemoryError: If reading the receptor types, grouping the cells
                by their ``within`` labels, or the definition, 1,000 bytes per
                cell and 12 per contig and with a distance metric the 512 MiB
                of the search, does not fit the memory budget; or, with a
                distance metric, the junctions that the cells' chains hold,
                decoded for the search, or the pairs of chains within the
                cutoff and the links between the chains or arms that match, 32
                bytes each, do not fit what is left.
        """
        group, fields = self._receptor_group(receptors)
        if sequence not in SEQUENCES:
            raise ValueError(f"sequence must be one of {', '.join(SEQUENCES)}")
        if receptor_arms not in RECEPTOR_ARMS:
            raise ValueError(f"receptor_arms must be one of {', '.join(RECEPTOR_ARMS)}")
        if dual_ir not in DUAL_IR:
            raise ValueError(f"dual_ir must be one of {', '.join(DUAL_IR)}")
        for name, flag in (
            ("same_v_gene", same_v_gene),
            ("same_j_gene", same_j_gene),
            ("within_receptor_type", within_receptor_type),
            ("invalidate_cache", invalidate_cache),
        ):
            bool_argument(flag, name)
        columns = group_columns(within, "within", _CLONOTYPE_COLUMNS)
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}")
        if metric == "identity":
            # Only the 0 that the artifact records stands for no cutoff.
            if cutoff is not None and (
                isinstance(cutoff, bool | np.bool_)
                or not isinstance(cutoff, int | np.integer)
                or cutoff != 0
            ):
                raise ValueError("metric='identity' takes no cutoff")
            limit = 0
        else:
            if cutoff is None:
                raise ValueError(
                    f"metric={metric!r} needs a cutoff, such as 1 or 2 "
                    "substitutions or edits, or 15 percent"
                )
            limit = integer_argument(cutoff, "cutoff", minimum=0)
            if metric == "normalized_hamming" and limit > 100:
                raise ValueError(
                    "A normalized_hamming cutoff is a percentage of the length, "
                    "at most 100"
                )
        field_name = "junction" if sequence == "nt" else "junction_aa"
        if field_name not in fields:
            raise ValueError(
                f"The receptors store no {field_name}, so they cannot be compared "
                f"by sequence={sequence!r}; pass sequence="
                f"{'aa' if sequence == 'nt' else 'nt'!r}"
            )
        for flag, gene in ((same_v_gene, "v_call"), (same_j_gene, "j_call")):
            if flag and gene not in fields:
                raise ValueError(f"The receptors store no {gene} to match")
        status = inspect_artifact(self.zw, receptors)
        cell_selection = status.input_ref("cell_selection")
        stored_types = as_zarr_array(group["values"], name="values")
        n_cells = int(stored_types.shape[0])
        held = n_cells * (stored_types.dtype.itemsize + _RECEPTOR_TYPE_CODING_BYTES)
        if held > self.memoryBytes:
            raise MemoryError(
                f"Reading the receptor types of {n_cells} cells holds about {held} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        receptor_type = np.asarray(stored_types[:])
        with_receptor = receptor_type != NO_RECEPTOR
        if within_receptor_type:
            _types, type_codes = _sorted_codes(receptor_type)
        else:
            type_codes = np.zeros(n_cells, dtype=np.int64)
        del receptor_type
        within_codes, within_labels, fingerprint = self._within_groups(
            cell_selection, columns, with_receptor
        )
        groups = np.where(
            with_receptor, type_codes * max(1, len(within_labels)) + within_codes, -1
        )
        arguments = ClonotypeArguments(
            receptors=receptors,
            within_fingerprint=fingerprint,
            sequence=sequence,
            receptor_arms=receptor_arms,
            dual_ir=dual_ir,
            same_v_gene=same_v_gene,
            same_j_gene=same_j_gene,
            within=columns,
            within_receptor_type=within_receptor_type,
            metric=metric,
            cutoff=limit,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
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
                "codes": ((n_cells,), "i"),
                "clone_size": ((n_cells,), "i"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("n_clonotypes",),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        # Each compared field's code of every contig is read, one field at a
        # time.
        contig_codes = as_zarr_array(group[f"contig_{field_name}"], name=field_name)
        resident = _CLONOTYPE_BYTES_PER_CELL * n_cells + int(contig_codes.shape[0]) * (
            contig_codes.dtype.itemsize + _CODE_BYTES
        )
        if metric != "identity":
            # A cluster search holds its encoded junctions, one pass of keys
            # with their candidate pairs, and one chunk of candidate cells.
            resident += CLUSTER_SEARCH_BYTES
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Defining the clonotypes of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        slots = _read(group, "chain_slots")
        compared = _slot_codes(group, field_name, slots)
        v_genes = _slot_codes(group, "v_call", slots) if same_v_gene else None
        j_genes = _slot_codes(group, "j_call", slots) if same_j_gene else None
        if metric == "identity":
            assignment = assign_clonotypes(
                compared,
                groups=groups,
                v_genes=v_genes,
                j_genes=j_genes,
                receptor_arms=receptor_arms,
                dual_ir=dual_ir,
            )
        else:
            # Only the junctions that the slots compare are decoded, so the
            # contigs that no slot holds, such as non-productive ones, cost
            # nothing, and the slots are coded by their position among them.
            # The decoded junctions are held while the search and the
            # linking run, which get what the budget leaves beside them.
            present = compared >= 0
            compared_codes, local = np.unique(compared[present], return_inverse=True)
            compared[present] = local.reshape(-1)
            reads = _Reads(
                limit=self.memoryBytes,
                held=resident - CLUSTER_SEARCH_BYTES,
                what=(
                    f"Decoding the {len(compared_codes)} junctions that the clonotype "
                    f"clusters of {n_cells} cells compare"
                ),
                remedy="Raise mem_budget.",
            )
            texts = _decoded_values(reads, group, field_name, compared_codes)
            held = reads.held + CLUSTER_SEARCH_BYTES
            if held > self.memoryBytes:
                raise MemoryError(
                    f"Defining the clonotype clusters of {n_cells} cells holds about "
                    f"{held} bytes with the {len(compared_codes)} junctions they "
                    f"compare, more than the memory budget of {self.memoryBytes} "
                    "bytes. Raise mem_budget."
                )
            assignment = assign_clonotype_clusters(
                compared,
                texts,
                groups=groups,
                metric=metric,
                cutoff=limit,
                v_genes=v_genes,
                j_genes=j_genes,
                receptor_arms=receptor_arms,
                dual_ir=dual_ir,
                max_bytes=self.memoryBytes - held,
            )
        labels = clonotype_labels(assignment.n_clonotypes)
        has = assignment.codes >= 0
        values = np.full(n_cells, "", dtype=object)
        values[has] = labels[assignment.codes[has]]
        sizes = np.zeros(n_cells, dtype=np.int64)
        sizes[has] = assignment.sizes[assignment.codes[has]]
        representatives, n_configurations = representative_cells(assignment)
        clonotype_within = (
            within_codes[representatives]
            if columns
            else np.full(assignment.n_clonotypes, -1, dtype=np.int64)
        )
        arrays = {
            "values": values,
            "codes": assignment.codes.astype(np.int32),
            "clone_size": sizes,
            "clonotype_ids": labels,
            "clonotype_sizes": assignment.sizes,
            "representatives": representatives,
            "n_configurations": n_configurations,
            "clonotype_within": clonotype_within,
        }
        if columns:
            # Each within group's label in each column, row by row, as an
            # array, so opening the artifact reads none of them.
            arrays["within_labels"] = np.asarray(
                [key[column] for key in within_labels for column in columns],
                dtype=object,
            )
        write_cell_data_artifact(
            self.zw,
            planned,
            arrays,
            attributes={"n_clonotypes": assignment.n_clonotypes},
        )
        defined = "clonotypes" if metric == "identity" else "clonotype clusters"
        logger.info(
            f"Defined {assignment.n_clonotypes} {defined} of {int(has.sum())} of "
            f"{n_cells} cells; the largest holds "
            f"{int(assignment.sizes[0]) if assignment.n_clonotypes else 0} cells"
        )
        return planned.ref

    def _clonotype_group(self, clonotypes: ArtifactRef) -> zarr.Group:
        """Return a complete ``clonotypes`` artifact's group."""
        if not isinstance(clonotypes, ArtifactRef):
            raise TypeError("clonotypes must be an ArtifactRef")
        if clonotypes.kind != ClonotypeArguments.artifact_kind:
            raise ValueError(
                f"clonotypes must be a clonotypes artifact, not {clonotypes.kind!r}"
            )
        require_complete_artifact(self.zw, clonotypes)
        return artifact_group(self.zw, clonotypes)

    def _representative_junctions(
        self, clonotypes: ArtifactRef, codes: np.ndarray, *, held_bytes: int = 0
    ) -> np.ndarray:
        """The junction amino acids that name some clonotypes in figures.

        A clonotype is named by its representative cell's ``VDJ_1`` chain, or
        its ``VJ_1`` chain without one; clonotypes defined by
        ``receptor_arms="VJ"`` by the ``VJ_1`` chain first, which their
        cells share. Of the receptors, only the representatives' slots and
        junction codes, the bounds of the junctions they name, a chunk at a
        time or whole where that holds less, and those junctions' bytes are
        read, each array a physical chunk at a time. Before each read, what it
        holds is checked against the memory budget beside ``held_bytes`` and
        what earlier reads keep: one decoded chunk and its compressed bytes,
        with three copies and the index arrays of each row it reads, and for
        the junctions' bytes the chunk's slice, the bytes and each junction
        while it is decoded.

        Args:
            clonotypes: A complete ``clonotypes`` artifact.
            codes: The positions of some of its clonotypes, such as 0 for
                ``clonotype_1``.
            held_bytes: What the caller holds beside the reads.

        Returns:
            Each clonotype's junction, None where the chain that names it
            has no junction amino-acid sequence, as an object array.

        Raises:
            MemoryError: If a read does not fit the memory budget.
        """
        group = self._clonotype_group(clonotypes)
        status = inspect_artifact(self.zw, clonotypes)
        receptor_group = artifact_group(self.zw, status.input_ref("receptors"))
        naming = ("VJ_1", "VDJ_1")
        if (status.parameters or {}).get("receptor_arms") != "VJ":
            naming = ("VDJ_1", "VJ_1")
        reads = _Reads(
            limit=self.memoryBytes,
            held=int(held_bytes),
            what="Labelling clonotypes by their junctions",
            remedy="Raise mem_budget, or label fewer clonotypes.",
        )
        wanted, back = np.unique(np.asarray(codes, dtype=np.int64), return_inverse=True)
        cells = reads.rows(group, "representatives", wanted).astype(np.int64)
        distinct_cells, cell_back = np.unique(cells, return_inverse=True)
        stored_slots = reads.rows(receptor_group, "chain_slots", distinct_cells)
        slots = np.asarray(
            stored_slots[:, [SLOTS.index(slot) for slot in naming]], dtype=np.int64
        )[cell_back.reshape(-1)]
        contigs = np.where(slots[:, 0] >= 0, slots[:, 0], slots[:, 1])
        junction_codes = np.full(len(contigs), -1, dtype=np.int64)
        present = contigs >= 0
        if present.any():
            distinct_contigs, contig_back = np.unique(
                contigs[present], return_inverse=True
            )
            junction_codes[present] = reads.rows(
                receptor_group, "contig_junction_aa", distinct_contigs
            ).astype(np.int64)[contig_back.reshape(-1)]
        junctions = np.full(len(wanted), None, dtype=object)
        named = junction_codes >= 0
        if named.any():
            # Only the bytes of the junctions needed are read and decoded.
            needed, inverse = np.unique(junction_codes[named], return_inverse=True)
            text = _decoded_values(reads, receptor_group, "junction_aa", needed)
            junctions[np.flatnonzero(named)] = text[inverse.reshape(-1)]
        return junctions[back.reshape(-1)]

    @validation_scoped
    def _repertoire_load_clonotypes(self, clonotypes: ArtifactRef) -> Clonotypes:
        """Read saved clonotypes as a table of cells and a table of clonotypes.

        Args:
            clonotypes: A ``clonotypes`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.clonotypes`.

        Returns:
            The clonotypes, with one row per cell and one row per clonotype,
            largest first, with the chains of each clonotype's representative
            cell.

        Raises:
            TypeError: If ``clonotypes`` is not an ArtifactRef.
            ValueError: If it is not a ``clonotypes`` artifact.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete.
            MemoryError: If the tables, 300 bytes per cell and 700 per
                clonotype beside the cells' IDs, the representatives' chains
                and the ``within`` labels that they read as text, do not fit
                the memory budget.
        """
        group = self._clonotype_group(clonotypes)
        n_cells = int(as_zarr_array(group["codes"], name="codes").shape[0])
        status = inspect_artifact(self.zw, clonotypes)
        parameters = dict(status.parameters or {})
        receptors = status.input_ref("receptors")
        receptor_group = artifact_group(self.zw, receptors)
        fields = _stored_fields(receptor_group)
        within = tuple(str(column) for column in parameters.get("within") or ())
        n_clonotypes = int(
            as_zarr_array(group["clonotype_ids"], name="clonotype_ids").shape[0]
        )
        # Beside the tables of the cells and the clonotypes: the cells' IDs and
        # the within groups' labels as text, and each representative field.
        resident = (
            _CLONOTYPE_TABLE_BYTES_PER_CELL * n_cells
            + _CLONOTYPE_TABLE_BYTES_PER_CLONOTYPE * n_clonotypes
            + text_read_bytes(self.cells._get_array("ids"), n_cells)
        )
        if within:
            stored = as_zarr_array(group["within_labels"], name="within_labels")
            resident += text_read_bytes(stored, int(stored.shape[0]))
        decoded = 0
        for name in REPRESENTATIVE_FIELDS:
            if name in fields:
                packed = as_zarr_array(
                    receptor_group[f"contig_{name}_values"], name=name
                )
                offsets = as_zarr_array(
                    receptor_group[f"contig_{name}_offsets"], name=name
                )
                n_values = int(offsets.shape[0]) - 1
                # The representatives' four slots hold at most four values.
                n_decoded = min(n_values, 4 * n_clonotypes)
                decoded = max(decoded, n_decoded)
                resident += (
                    _FIELD_BYTES_PER_VALUE * n_values
                    + _FIELD_BYTES_PER_DECODED_VALUE * n_decoded
                    + _FIELD_BYTES_PER_PACKED_BYTE * int(packed.nbytes)
                )
        resident += _DECODE_BYTES_PER_BATCH_VALUE * min(decoded, _DECODE_BATCH_VALUES)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Loading the clonotypes of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        representatives = _read(group, "representatives").astype(np.int64)
        slots = _read(receptor_group, "chain_slots")[representatives]
        # Only the representatives' contigs are decoded.
        contigs = slots.reshape(-1)
        local = np.where(contigs >= 0, np.arange(len(contigs)), -1)
        chains = {
            name: _stored_text_at(receptor_group, name, contigs)
            for name in REPRESENTATIVE_FIELDS
            if name in fields
        }
        within_codes = _read(group, "clonotype_within").astype(np.int64)
        within_labels = (
            _read(group, "within_labels").reshape(-1, len(within))
            if within
            else np.empty((0, 0), dtype=object)
        )
        cells, table = clonotype_frames(
            self._artifact_cell_ids(clonotypes),
            codes=_read(group, "codes"),
            clone_size=_read(group, "clone_size"),
            clonotype_ids=_read(group, "clonotype_ids"),
            clonotype_sizes=_read(group, "clonotype_sizes"),
            n_configurations=_read(group, "n_configurations"),
            receptor_type=_read(receptor_group, "values")[representatives],
            within={
                column: _labels_at(within_labels[:, position], within_codes)
                for position, column in enumerate(within)
            },
            slots=local.reshape(slots.shape),
            chains=chains,
        )
        return Clonotypes(
            cells=cells,
            clonotypes=table,
            receptors=receptors,
            sequence=str(parameters["sequence"]),
            receptor_arms=str(parameters["receptor_arms"]),
            dual_ir=str(parameters["dual_ir"]),
            same_v_gene=bool(parameters["same_v_gene"]),
            same_j_gene=bool(parameters["same_j_gene"]),
            within=within,
            within_receptor_type=bool(parameters["within_receptor_type"]),
            metric=str(parameters["metric"]),
            cutoff=int(parameters["cutoff"]),
        )
