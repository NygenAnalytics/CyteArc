import io
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...clones.groups import group_key_table
from ...metadata.arguments import ClonotypeArguments, ImmuneReceptorImportArguments
from ...metadata.artifacts import (
    cell_data_write_bytes,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.records import (
    FLAG,
    INTEGER,
    NUMBER,
    RAGGED,
    TEXT,
    EncodedText,
    RecordTable,
    decode_ragged,
    pack_text,
    take_text,
    unpack_text,
)
from ...metadata.rows import (
    read_rows,
    read_text_rows,
    read_whole,
    row_read_bytes,
    text_read_bytes,
)
from ...metadata.selection import (
    plan_cell_selection_read,
    read_cell_selection,
)
from ...storage.selections import validate_cell_selection
from ...repertoire.chains import (
    CHAIN_PAIRINGS,
    NO_RECEPTOR,
    RECEPTOR_SUBTYPES,
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
    ClonotypeAssignment,
    Clonotypes,
    assign_clonotype_clusters,
    assign_clonotypes,
    clonotype_frames,
    clonotype_labels,
    clonotype_nouns,
    representative_cells,
)
from ...repertoire.contigs import FIELDS, contig_encoder
from ...repertoire.receptors import (
    SLOT_FIELDS,
    ImmuneReceptors,
    contig_frame,
    receptor_cell_frame,
)
from ...repertoire.usage import (
    GENE_FIELDS,
    GeneUsage,
    Spectratype,
    count_chains,
    gene_codes,
    gene_names,
    text_lengths,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.async_execution import zarr_decoding_threads, zarr_io_concurrency
from ...storage.diagnostics import ExportSummary, export_summary, warn_diagnostics
from ...storage.geometry import ROW_INDEX_BYTES, array_geometry
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument, integer_argument, quoted_value
from ...utils.logging import logger
from ...utils.memory_budget import (
    MemoryLedger,
    hash_table_bytes,
    object_array_bytes,
    python_text_bytes,
)
from .cell_barcodes import (
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    unmatched_barcode_rules,
)
from .cell_groups import cell_groups, group_columns, within_groups
from .clone_artifacts import clone_diagnostics
from .clone_steps import (
    CODES_WITHIN_TYPE,
    artifact_cell_ids,
    files_text,
    measured_within_types,
    plan_artifact_cells,
)
from .record_tables import (
    record_arrays,
    record_chunk_lengths,
    stored_column,
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
# stored fields.
_IMPORT_ATTRIBUTES = ("fields",)
# At its peak, while it orders the contigs, ranks the chains and writes the
# artifact, an import holds up to three times the bytes of its encoded
# contigs beside what it holds for each selected cell, as measured on
# Cell Ranger files with and without region sequences and on AIRR files with
# sequences and alignments.
_PEAK_PER_ENCODED_BYTE = 3
# Parsing a chunk of a contig file holds its raw text, its split fields and
# the parsed table, about 55 to 59 bytes for each field of a row as traced on
# chunks of 10,000 to 50,000 rows of 18 and 31 columns, and objects of about
# 0.8 MB whatever the chunk's size. Chunks hold 1,000 to 50,000 rows.
_PARSE_BYTES_PER_FIELD = 64
_PARSE_FIXED_BYTES = 1 << 20
_LEAST_PARSED_ROWS = 1_000
_MOST_PARSED_ROWS = 50_000
# Bytes that an import holds for each selected cell beside the join's lookup
# of cell IDs and their text, which the join counts, the read of the IDs,
# which text_read_bytes counts, and what writing the cells' arrays holds,
# which _receptor_write_bytes counts: the selection, and each cell's chain
# slots, counts and labels, about 70 bytes as their arrays add up and 44 as
# traced beside the rest at the peak of 50,000 cells of which 1,000 have
# contigs.
_IMPORT_BYTES_PER_CELL = 100
# Bytes that a decoded contig field takes in a table, per contig, besides the
# text of ragged fields.
_TABLE_BYTES_PER_FIELD = 64
# Bytes per cell of the table of cells without its slot fields: the cell IDs,
# the labels and chain counts as read and as table columns, and the slots.
# Up to 560 bytes per cell were resident at the peak, on 1,000,000 cells.
_CELL_TABLE_BYTES_PER_CELL = 700
# The prefix of the stored contig arrays, such as contig_cells.
_CONTIGS = "contig"
# Defining clonotypes holds the chains of every slot, their distinct
# configurations and keys, and the labels: 340 bytes per cell at the peak, as
# traced on 1,000,000 simulated cells of about two chains each. Cells with more
# chains give more keys, so the definition reserves 1,000 bytes per cell,
# beside the stored code and its int64 copy of each contig's compared field.
_CLONOTYPE_BYTES_PER_CELL = 1_000
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
# string. Once decoded, each keeps its string.
_JUNCTION_BYTES_PER_VALUE = 200
_JUNCTION_BYTES_PER_BYTE = 7
# The parameters that this version records for clonotypes, which tell an
# artifact that an earlier version saved from one it reads.
_CLONOTYPE_PARAMETERS = frozenset(
    field.name
    for field in fields(ClonotypeArguments)
    if field.metadata.get("argument_role") == "parameter"
)
# Finding the chains of the chosen slots that grouped cells hold holds, for
# each, its cell, group and clonotype and the codes of its locus and of each
# counted field, as int64, and while they are found, counted and ordered,
# sorted and combined copies of them: about 40 bytes and 16 more for each
# field, the locus and the clonotype among them, as traced on 200,000 cells
# of two chains each, counting cells and clonotypes.
_CHAIN_BYTES = 64
_CHAIN_BYTES_PER_FIELD = 24
# The table of the groups' labels holds each group's name and its label in
# each key column, beside the groups' labels, which the grouping holds: 75,
# 190 and 505 bytes at its peak for groups of one, three and thirteen key
# columns, as traced on 20,000 and 200,000 groups.
_KEY_BYTES_PER_GROUP = 100
_KEY_BYTES_PER_GROUP_COLUMN = 40
# Each row of the table, while it is built: its group, locus and values as
# codes and as pointers to shared text, its count and fraction, and the
# order of the rows, and 16 bytes for each column of the table; 114 to 130
# bytes beside the chains, as traced on 400,000 rows of one and three key
# columns.
_ROW_BYTES = 120
_ROW_BYTES_PER_COLUMN = 16
# Counting each clonotype once holds, for each chain, its row of group,
# locus, values and clonotype among the distinct rows: 80 bytes beside the
# counts of cells with one field and 16 more for each field, as traced on
# 400,000 chains of clonotypes of their own, checked at 16 bytes for each
# column of the row and 32 more.
_ONCE_BYTES_PER_CHAIN = 32
_ONCE_BYTES_PER_COLUMN = 16
# Ranking the codes of a field among its stored values holds, for each
# value, whether a chain names it and its rank.
_RANK_BYTES_PER_VALUE = 17
# Measuring a junction holds its bounds, its length and the bounds of its
# part of each chunk it spans, and text beyond ASCII also a mask and a running
# count of its continuation bytes, as int32, for each byte of the chunk read.
# On 5,000 to 400,000 junctions of 5 to 906 characters, ASCII or of four bytes
# each, the traced peak stayed at 0.35 to 0.88 of what the checks count.
_LENGTH_BYTES_PER_VALUE = 48
_CONTINUED_BYTES_PER_BYTE = 5


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
            + " or ".join(quoted_value(name) for name in formats)
            + f", or None to read each file's format from its header, not {quoted_value(format)}"
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
) -> Iterator[tuple[np.ndarray, pd.DataFrame]]:
    from ...readers.cellranger import CELLRANGER_VDJ_FORMAT, iter_vdj_contigs

    if source.format == CELLRANGER_VDJ_FORMAT:
        return iter_vdj_contigs(source.file.path, chunk_rows=chunk_rows)
    from ...readers.airr import iter_airr_rearrangements

    return iter_airr_rearrangements(source.file.path, chunk_rows=chunk_rows)


def _encode_contigs(
    sources: Sequence[_ContigSource], join: CellBarcodeJoin, ledger: MemoryLedger
) -> RecordTable:
    """Read the contig files and encode the contigs of the selected cells.

    Each contig joins the selected cell that its barcode names, through
    ``join``, which counts the contigs and barcodes of the files and those
    that match no selected cell. Each file is parsed in chunks of as many
    rows as half the ledger's room holds, from 1,000 to 50,000, so that the
    encoded contigs keep the other half; a chunk's parse is charged beside
    the contigs encoded so far before the file is read and held until it is
    read, since the reader keeps each chunk's text until it parses the next,
    and after each chunk, the peak that the encoded contigs reach while the
    import orders them, ranks the chains and writes the artifact.

    Raises:
        MemoryBudgetError: If a chunk's parse or the import's peak would pass
            the ledger's budget.
        ValueError: If a contig of a selected cell has no ID, or an ID repeats
            among the contigs of selected cells within one file.
    """
    encoder = contig_encoder(ledger)
    for position, source in enumerate(sources):
        per_row = _PARSE_BYTES_PER_FIELD * source.n_columns
        # The encoder holds the contigs encoded so far, which the room leaves
        # out; a chunk takes half of what the room and they leave together.
        room = ledger.room
        fitting = (
            _MOST_PARSED_ROWS
            if room is None
            else ((room - encoder.nbytes) // 2 - _PARSE_FIXED_BYTES) // per_row
        )
        chunk_rows = min(_MOST_PARSED_ROWS, max(_LEAST_PARSED_ROWS, fitting))
        chunks = _iter_source(source, chunk_rows)
        # The reader keeps a chunk's text until it parses the next, so a
        # chunk's parse stays charged until the file is read.
        parsing = _PARSE_FIXED_BYTES + chunk_rows * per_row
        ledger.hold(f"a chunk of {chunk_rows} contig rows as parsed", parsing)
        while True:
            parsed = next(chunks, None)
            if parsed is None:
                break
            chunk = parsed[1]
            rows = join.cells(source.file, chunk["cell_id"].to_numpy(dtype=object))
            matched = rows >= 0
            encoder.add(
                chunk.loc[matched].drop(columns="cell_id"), rows[matched], position
            )
            del chunk, parsed, rows, matched
            # Beside the contigs encoded, which the encoder holds; the parse is
            # let go before the import's peak, so only what the copies of the
            # contigs take beyond it counts beside what is held now.
            ledger.peak(
                f"the {encoder.n_records} contigs of the selected cells at the "
                "import's peak",
                max(0, (_PEAK_PER_ENCODED_BYTE - 1) * encoder.nbytes - parsing),
            )
        ledger.release(parsing)
    if encoder.missing_ids:
        raise ValueError(
            f"{encoder.missing_ids} contigs of the selected cells have no "
            "sequence_id (contig_id in Cell Ranger's files)"
        )
    repeated = encoder.repeated_ids()
    if repeated:
        position, name = repeated[0]
        raise ValueError(
            f"Contig ID {quoted_value(name)} repeats in "
            f"{quoted_value(os.fspath(sources[position].file.path))}; each contig of a file "
            "needs its own ID"
        )
    return encoder.finish()


def _receptor_write_bytes(n_cells: int) -> int:
    """What writing the per-cell arrays of imported receptors holds.

    Each label is counted at the width of the longest label of its kind.
    """
    return cell_data_write_bytes(
        {
            "values": ((n_cells,), f"U{max(map(len, RECEPTOR_TYPES))}"),
            "receptor_subtype": ((n_cells,), f"U{max(map(len, RECEPTOR_SUBTYPES))}"),
            "chain_pairing": ((n_cells,), f"U{max(map(len, CHAIN_PAIRINGS))}"),
            "multichain": ((n_cells,), bool),
            "n_vj": ((n_cells,), np.int32),
            "n_vdj": ((n_cells,), np.int32),
            "n_contigs": ((n_cells,), np.int32),
            "chain_slots": ((n_cells, len(SLOTS)), np.int32),
        }
    )


def _clonotype_write_bytes(
    n_cells: int,
    columns: tuple[str, ...],
    within_labels: list[dict[str, str]],
    *,
    within_types: bool = False,
) -> int:
    """What writing the arrays of clonotypes holds, as many as there are cells at most.

    Each label is counted at the width of the label of the last of as many
    clonotypes as cells, and each ``within`` label at the widest of them.
    ``within_types`` counts each cell's clonotype within its receptor type,
    which clonotypes across receptor types write.
    """
    width = len(f"clonotype_{max(1, n_cells)}")
    arrays: dict[str, tuple[tuple[int, ...], Any]] = {
        "values": ((n_cells,), f"U{width}"),
        "codes": ((n_cells,), np.int32),
        "clone_size": ((n_cells,), np.int64),
        "clonotype_ids": ((n_cells,), f"U{width}"),
        "clonotype_sizes": ((n_cells,), np.int64),
        "representatives": ((n_cells,), np.int64),
        "n_configurations": ((n_cells,), np.int64),
        "clonotype_within": ((n_cells,), np.int64),
    }
    if within_types:
        arrays[CODES_WITHIN_TYPE] = ((n_cells,), np.int32)
    if columns:
        widest = max(
            (len(key[column]) for key in within_labels for column in columns), default=1
        )
        arrays["within_labels"] = (
            (len(within_labels) * len(columns),),
            f"U{max(1, widest)}",
        )
    return cell_data_write_bytes(arrays)


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
        data = read_array(group, f"contig_{name}_values")
        offsets = read_array(group, f"contig_{name}_offsets")
        text = np.empty(len(needed), dtype=object)
        for start in range(0, len(needed), _DECODE_BATCH_VALUES):
            batch = needed[start : start + _DECODE_BATCH_VALUES]
            text[start : start + len(batch)] = take_text(data, offsets, batch)
        values[codes >= 0] = text[inverse.reshape(-1)]
    return values


def _slot_codes(
    ledger: MemoryLedger, group: zarr.Group, name: str, slots: np.ndarray
) -> np.ndarray:
    """Return a text field's code in each cell's slots, -1 for an empty slot.

    Every contig's stored code is read whole and copied as int64, each read
    and copy charged to ``ledger`` and let go once the slots' codes are
    taken. The caller charges the slots' codes it keeps.
    """
    stored = read_whole(ledger, group, f"contig_{name}", item=f"each contig's {name}")
    # The codes as int64, and the slots' codes with the mask of the filled
    # slots, their contigs and their codes as gathered.
    ledger.peak(f"each contig's {name} as int64", 8 * len(stored) + 25 * slots.size)
    codes = stored.astype(np.int64)
    stored_bytes = stored.nbytes
    del stored
    ledger.release(stored_bytes)
    taken = np.full(slots.shape, -1, dtype=np.int64)
    filled = slots >= 0
    taken[filled] = codes[slots[filled]]
    return taken


def _value_bounds(
    ledger: MemoryLedger, group: zarr.Group, name: str, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Read where the stored values of some codes of a contig text field lie.

    The bounds are read a chunk at a time, or whole where that holds less;
    each read is charged to ``ledger`` first, and the bounds of the codes
    stay charged.

    Args:
        ledger: What the caller holds, and the budget.
        group: An ``immune_receptors`` artifact's group.
        name: The text field, such as ``junction_aa``.
        codes: Distinct codes of the field in ascending order.

    Returns:
        The start and end of each code's value among the packed bytes.
    """
    item = f"the bounds of the {len(codes)} {name} values"
    offsets_name = f"contig_{name}_offsets"
    offsets = as_zarr_array(group[offsets_name], name=name)
    # Each code's bound and the next, sorted and joined.
    ledger.peak(item, 48 * len(codes))
    bound_rows = np.union1d(codes, codes + 1)
    geometry = array_geometry(offsets)
    whole = geometry is None or geometry.wholeReadBytes(1) < row_read_bytes(
        geometry, bound_rows
    )
    # The bounds as read, or every bound where most values are named, which
    # holds less, then each code's start and end as int64.
    if whole:
        bounds = read_whole(ledger, group, offsets_name, item=item)
        ledger.peak(item, 24 * len(codes))
        starts = bounds[codes].astype(np.int64)
        ends = bounds[codes + 1].astype(np.int64)
    else:
        bounds = read_rows(ledger, offsets, bound_rows, item=item)
        ledger.peak(item, 32 * len(codes))
        starts = bounds[np.searchsorted(bound_rows, codes)].astype(np.int64)
        ends = bounds[np.searchsorted(bound_rows, codes + 1)].astype(np.int64)
    bounds_bytes = bounds.nbytes
    del bounds, bound_rows
    ledger.release(bounds_bytes)
    ledger.hold(item, starts.nbytes + ends.nbytes)
    return starts, ends


def _decoded_values(
    ledger: MemoryLedger, group: zarr.Group, name: str, codes: np.ndarray
) -> np.ndarray:
    """Decode the stored values of some codes of a contig text field.

    The bounds of the values are read as :func:`_value_bounds` reads them,
    and their bytes a chunk at a time, so values that no code names are
    never decoded. Each read, and the decoding, is charged to ``ledger``
    first; the decoded values stay charged as the Python strings they are,
    and the bounds are let go.

    Args:
        ledger: What the caller holds, and the budget.
        group: An ``immune_receptors`` artifact's group.
        name: The text field, such as ``junction_aa``.
        codes: Distinct codes of the field in ascending order.

    Returns:
        The value of each code, as an object array of text.
    """
    values = as_zarr_array(group[f"contig_{name}_values"], name=name)
    starts, ends = _value_bounds(ledger, group, name, codes)
    n_bytes = int((ends - starts).sum())
    geometry = array_geometry(values)
    # A stored chunk of the bytes and its slice as read, with the chunk read
    # before, the bytes joined, and each value while it is decoded.
    reading = (
        0
        if geometry is None
        else geometry.bandReadBytes(geometry.axisChunk(0), 1, bands=2)
    )
    ledger.peak(
        f"decoding the {len(codes)} {name} values",
        reading
        + _JUNCTION_BYTES_PER_VALUE * len(codes)
        + _JUNCTION_BYTES_PER_BYTE * n_bytes,
    )
    lengths = np.zeros(len(codes) + 1, dtype=np.int64)
    np.cumsum(ends - starts, out=lengths[1:])
    with zarr_decoding_threads(ledger.codec_threads):
        joined = _byte_ranges(values, starts, ends)
    if geometry is not None and len(joined):
        # The chunks from the first range to the last, each read once, leave
        # one on each of the ledger's codec threads.
        step = geometry.axisChunk(0)
        spanned = (int(ends[-1]) - 1) // step - int(starts[0]) // step + 1
        ledger.leave(geometry.heldReadBytes(ledger.holding_threads(spanned)))
    # Each value is a string of one byte a character when every byte is
    # ASCII, and of up to four otherwise.
    widest = int(joined.max()) if len(joined) else 0
    text = unpack_text(joined, lengths)
    del joined, lengths
    ledger.hold(
        f"the {len(codes)} {name} values as text",
        object_array_bytes(len(codes))
        + python_text_bytes(
            len(codes), n_bytes, widest=widest if widest < 0x80 else 0x10FFFF
        ),
    )
    bounds_bytes = starts.nbytes + ends.nbytes
    del starts, ends
    ledger.release(bounds_bytes)
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


def _shares_an_arm(slots: tuple[str, ...]) -> bool:
    """Whether two of the slots hold chains of one arm, which may share values."""
    chosen = set(slots)
    return {"VJ_1", "VJ_2"} <= chosen or {"VDJ_1", "VDJ_2"} <= chosen


def _value_ranks(codes: np.ndarray, n_values: int) -> tuple[np.ndarray, np.ndarray]:
    """The distinct stored values that some codes name, and each code's rank.

    Args:
        codes: Codes of a contig text field, -1 for none.
        n_values: The values that the field stores.

    Returns:
        Each code's position among the distinct codes, -1 for none, and the
        distinct codes in ascending order.
    """
    known = codes >= 0
    present = np.zeros(n_values, dtype=bool)
    present[codes[known]] = True
    needed = np.flatnonzero(present)
    rank = np.cumsum(present, dtype=np.int64) - 1
    local = np.full(len(codes), -1, dtype=np.int64)
    local[known] = rank[codes[known]]
    return local, needed


def _value_lengths(
    ledger: MemoryLedger, group: zarr.Group, name: str, codes: np.ndarray
) -> np.ndarray:
    """Measure the stored values of some codes of a contig text field.

    The bounds of the values are read as :func:`_value_bounds` reads them,
    and their bytes a stored chunk at a time, only the chunks that hold
    them, so the read holds one chunk whatever the field stores. Each value
    is measured in characters without being decoded: in a chunk beyond
    ASCII, the bytes that continue a character are counted, which adds up
    over the chunks that one value spans. Each read is charged to ``ledger``
    first; the lengths stay charged, and the bounds are let go.

    Args:
        ledger: What the caller holds, and the budget.
        group: An ``immune_receptors`` artifact's group.
        name: The text field, such as ``junction_aa``.
        codes: Distinct codes of the field in ascending order.

    Returns:
        The length of each code's value, in characters.
    """
    packed = as_zarr_array(group[f"contig_{name}_values"], name=name)
    n_bytes = int(packed.shape[0])
    geometry = array_geometry(packed)
    step = max(1, n_bytes if geometry is None else geometry.axisChunk(0))
    starts, ends = _value_bounds(ledger, group, name, codes)
    # One chunk as it is read, with the chunk read before, and for text beyond
    # ASCII the block's mask and running count as int32, beside each value's
    # bounds and length.
    reading = (
        0
        if geometry is None
        else geometry.bandReadBytes(min(step, n_bytes), 1, bands=2)
    )
    ledger.peak(
        f"measuring the {len(codes)} {name} values",
        reading
        + _CONTINUED_BYTES_PER_BYTE * min(step, n_bytes)
        + _LENGTH_BYTES_PER_VALUE * len(codes),
    )
    lengths: np.ndarray = ends - starts
    filled = np.flatnonzero(lengths > 0)
    first = int(starts[filled[0]]) // step if len(filled) else 0
    last = int(ends[filled[-1]] - 1) // step if len(filled) else -1
    for chunk in range(first, last + 1):
        low, high = chunk * step, min((chunk + 1) * step, n_bytes)
        # The values that end after the chunk starts and start before it ends.
        begin = int(np.searchsorted(ends, low, side="right"))
        stop = int(np.searchsorted(starts, high, side="left"))
        if begin >= stop:
            continue
        with zarr_decoding_threads(ledger.codec_threads):
            block = np.asarray(packed[low:high], dtype=np.uint8)
        parts = np.empty(2 * (stop - begin), dtype=np.int64)
        parts[0::2] = np.clip(starts[begin:stop], low, high) - low
        parts[1::2] = np.clip(ends[begin:stop], low, high) - low
        # Each part's characters, its bytes less those that continue one.
        characters = text_lengths(block, parts, np.arange(0, len(parts), 2))
        lengths[begin:stop] -= np.diff(parts)[0::2] - characters
        del block
    if geometry is not None and len(filled):
        ledger.leave(geometry.heldReadBytes(ledger.holding_threads(last - first + 1)))
    bounds_bytes = starts.nbytes + ends.nbytes
    del starts, ends, filled
    ledger.release(bounds_bytes)
    ledger.hold(f"the lengths of the {len(codes)} {name} values", lengths.nbytes)
    return lengths


def _coding_bytes(n_values: int, width: int, n_distinct: int) -> int:
    """What coding ``n_values`` stored labels of up to ``width`` ASCII characters with :func:`_sorted_codes` holds.

    pandas hashes a Python string for each label, as an object array, into a
    table sized for every label, and gives each its code beside a reference
    to each label; sorting the ``n_distinct`` distinct labels renumbers the
    codes in place.
    """
    return (
        object_array_bytes(n_values)
        + python_text_bytes(n_values, n_values * width)
        + hash_table_bytes(n_values, n_distinct=n_distinct)
        + 16 * n_values
    )


def _sorted_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted distinct labels and each label's code among them.

    The codes equal those of ``np.unique(values, return_inverse=True)``, by
    hashing rather than by sorting every label.
    """
    codes, distinct = pd.factorize(np.asarray(values), sort=True)
    return np.asarray(distinct), np.asarray(codes, dtype=np.int64)


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
        The artifact's diagnostics, which
        ``ds.artifacts.inspect(ref).diagnostics`` returns, hold the join's
        counts of the files' barcodes, those that match no cell of the store
        and those of cells outside the selection, with examples of each,
        then ``fileContigs``, the files' contigs, ``unmatchedContigs``, those
        that match no selected cell, and ``cellsWithChain``, the selected
        cells with a chain. Barcodes that match no cell of the store are
        named in a warning, which a reused import gives again. A file is identified by the
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
        productive_only = bool_argument(productive_only, "productive_only")
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        sources = _contig_sources(
            paths,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            format=format,
        )
        selection = plan_cell_selection_read(self.zw, cell_selection)
        n_cells = selection.cells
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
        warnings = unmatched_barcode_rules(files_text("contig", len(sources)))
        if planned.reused:
            # Saved receptors warn as they did when they were imported.
            warn_diagnostics(
                arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
            )
            return planned.ref
        self._require_writable(arguments.operation)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Importing contigs for {n_cells} selected cells",
            remedy="pass a smaller cell_selection",
            codec_threads=self.nthreads,
        )
        # What the import holds for each selected cell, what writing their
        # labels and counts holds, and the join's lookup of their IDs are
        # charged before the IDs are read.
        ledger.hold(
            "each selected cell's chain slots, counts and labels",
            _IMPORT_BYTES_PER_CELL * n_cells,
        )
        ledger.hold(
            "writing each cell's labels and counts", _receptor_write_bytes(n_cells)
        )
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
        # The join charges its lookup of the IDs and the hashes of barcodes
        # that match no cell as they grow.
        join = CellBarcodeJoin(
            cell_ids, ledger=ledger, store_ids=self.cells._get_array("ids")
        )
        table = _encode_contigs(sources, join, ledger)
        contigs = record_arrays(table, _CONTIGS)
        # Ordering the contigs, ranking the chains and writing the artifact
        # hold, beside the table that the ledger holds, the copies of the
        # contigs that the import's peak counts, or what writing the contigs'
        # arrays holds where that is more, as for a few contigs.
        ledger.peak(
            f"ordering and writing the {table.n_records} contigs",
            max(
                (_PEAK_PER_ENCODED_BYTE - 1) * table.nbytes,
                cell_data_write_bytes(
                    {
                        name: (values.shape, values.dtype)
                        for name, values in contigs.items()
                    },
                    chunk_lengths=record_chunk_lengths(contigs),
                ),
            ),
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
        contig_arrays = {"contig_chain": index.eligible, **contigs}
        attributes = {"fields": list(table.fields)}
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
            chunk_lengths=record_chunk_lengths(contig_arrays),
            attributes=attributes,
            diagnostics={
                **join.diagnostics(),
                "fileContigs": join.n_rows,
                "unmatchedContigs": join.n_unmatched_rows,
                "cellsWithChain": int(np.count_nonzero(index.has_receptor)),
            },
        )
        warn_diagnostics(
            arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
        )
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
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as a call that
                stopped before it finished leaves it.
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
                    + ", ".join(quoted_value(str(name)) for name in unknown)
                    + "; they store "
                    + ", ".join(stored)
                )
            chosen = tuple(name for name in stored if name in set(fields))
        contig_cells = read_array(group, "contig_cells")
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
        MemoryLedger(
            self.memoryBytes,
            f"Loading the receptors of {n_cells} cells and {n_contigs} contigs "
            f"with {len(chosen)} fields",
            remedy="pass fields= with fewer fields",
        ).peak("the tables of the cells and the contigs", resident)
        # The canonical values come with the cell IDs; the other per-cell
        # arrays are aligned to the same cells.
        loaded = self._artifacts_load_values(receptors)
        cell_ids = loaded.cell_ids
        arrays = {name: read_array(group, name) for name in _CELL_ARRAYS[1:]}
        slots = read_array(group, "chain_slots")
        columns = {
            name: stored_column(group, _CONTIGS, name, FIELDS[name]) for name in chosen
        }
        chains = contig_frame(
            cell_ids,
            contig_cells=contig_cells,
            slots=slots,
            eligible=read_array(group, "contig_chain").astype(bool),
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
                f"{quoted_value(receptors.kind)}"
            )
        require_complete_artifact(self.zw, receptors)
        group = artifact_group(self.zw, receptors)
        return group, _stored_fields(group)

    def _within_groups(
        self,
        cell_selection: ArtifactRef,
        columns: tuple[str, ...],
        with_receptor: np.ndarray,
        ledger: MemoryLedger,
    ) -> tuple[np.ndarray, list[dict[str, str]], str | None]:
        """Group the cells with a receptor by their labels in the within columns, charging ``ledger``."""
        return within_groups(
            self.zw,
            self.cells,
            cell_selection,
            columns,
            with_receptor,
            ledger=ledger,
            what="a receptor",
        )

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
        max_distance: int | None = None,
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
        sequence among the chains that take part, has no clonotype. Defined
        across receptor types, the artifact also saves each cell's clonotype
        within its type, as ``within_receptor_type=True`` defines it, in
        ``codes_within_type``, which every clone summary measures, so the
        summaries of clonotypes across types equal those of clonotypes within
        types, under labels such as ``clonotype_1 (TCR 2)``.

        ``metric`` links chains whose junctions are similar instead, as
        scirpy's ``define_clonotype_clusters`` does: ``"hamming"`` (at most
        ``max_distance`` substitutions between junctions of one length),
        ``"normalized_hamming"`` (substitutions as a percentage of the length,
        rounded half up, at most ``max_distance``) or ``"levenshtein"`` (at
        most ``max_distance`` substitutions, insertions and deletions), under
        the same rules for arms, dual chains, genes and groups. The resulting
        clonotype clusters group T cells that likely share their epitope, or
        B cells of one clone despite somatic mutations. ``max_distance`` has
        no default for these metrics, since it decides the clusters, and a
        ``max_distance`` of 0 links identical junctions alone, as scirpy's
        cutoff of 0 does.

        Clonotypes are numbered by size, so ``clonotype_1`` is the largest;
        ties follow the receptor type in sorted order, then the ``within``
        labels in natural order, then the sorted chains, so the numbers do
        not depend on the order of the cells. The artifact's canonical ``values``
        hold each cell's clonotype label, empty for a cell without one, which
        groupings treat as missing; ``clone_size`` holds the cells of each
        cell's clonotype. The clone summaries of ``ds.clones``, such as
        :meth:`~cytearc.datastore.namespaces.ClonesAccessor.diversity`, read
        the artifact as they read clones, and measure the cells of each
        receptor type apart. Its diagnostics, which
        ``ds.artifacts.inspect(clonotypes).diagnostics`` returns, are those
        of every source's clones: ``cellsWithClone``, the cells given a
        clonotype, and ``largestCloneShare``, the share of them that the
        largest holds.

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
            max_distance: The largest distance of linked chains for the
                distance metrics: substitutions, percent, or edits; None with
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
                ``max_distance``, the receptors store no junction of the
                compared kind,
                or a ``within`` column is not a cell metadata column or has
                no label for a cell with a receptor.
            KeyError: If the store holds no such receptors.
            RuntimeError: If the receptors are incomplete.
            PermissionError: If the store is read-only and the clonotypes are
                not saved yet.
            MemoryError: If reading the receptor types, grouping the cells
                by their ``within`` labels, or the definition, 1,000 bytes per
                cell and 12 per contig and with a distance metric the 512 MiB
                of the search, does not fit the memory budget, across
                receptor types beside the clonotypes across types while each
                cell's clonotype within its type is defined; or, with a
                distance metric, the junctions that the cells' chains hold,
                decoded for the search, or the pairs of chains within
                ``max_distance`` and the links between the chains or arms that
                match, 32
                bytes each, do not fit what is left.
        """
        same_v_gene = bool_argument(same_v_gene, "same_v_gene")
        same_j_gene = bool_argument(same_j_gene, "same_j_gene")
        within_receptor_type = bool_argument(
            within_receptor_type, "within_receptor_type"
        )
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        group, fields = self._receptor_group(receptors)
        if sequence not in SEQUENCES:
            raise ValueError(f"sequence must be one of {', '.join(SEQUENCES)}")
        if receptor_arms not in RECEPTOR_ARMS:
            raise ValueError(f"receptor_arms must be one of {', '.join(RECEPTOR_ARMS)}")
        if dual_ir not in DUAL_IR:
            raise ValueError(f"dual_ir must be one of {', '.join(DUAL_IR)}")
        columns = group_columns(within, "within")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}")
        if metric == "identity":
            # Only the 0 that the artifact records stands for no distance.
            if max_distance is not None and (
                isinstance(max_distance, bool | np.bool_)
                or not isinstance(max_distance, int | np.integer)
                or max_distance != 0
            ):
                raise ValueError("metric='identity' takes no max_distance")
            limit = 0
        else:
            if max_distance is None:
                raise ValueError(
                    f"metric={quoted_value(metric)} needs a max_distance, such as 1 or 2 "
                    "substitutions or edits, or 15 percent"
                )
            limit = integer_argument(max_distance, "max_distance", minimum=0)
            if metric == "normalized_hamming" and limit > 100:
                raise ValueError(
                    "A normalized_hamming max_distance is a percentage of the "
                    "length, at most 100"
                )
        field_name = "junction" if sequence == "nt" else "junction_aa"
        if field_name not in fields:
            raise ValueError(
                f"The receptors store no {field_name}, so they cannot be compared "
                f"by sequence={quoted_value(sequence)}; pass sequence="
                f"{quoted_value('aa' if sequence == 'nt' else 'nt')}"
            )
        for name, flag, gene in (
            ("same_v_gene", same_v_gene, "v_call"),
            ("same_j_gene", same_j_gene, "j_call"),
        ):
            if flag and gene not in fields:
                raise ValueError(
                    f"{name} compares each chain's {gene}, which these receptors "
                    f"do not store; leave {name} False"
                )
        status = inspect_artifact(self.zw, receptors)
        cell_selection = status.input_ref("cell_selection")
        stored_types = as_zarr_array(group["values"], name="values")
        n_cells = int(stored_types.shape[0])
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Defining the clonotypes of {n_cells} cells",
            codec_threads=self.nthreads,
        )
        receptor_type = read_whole(
            ledger, group, "values", item="the cells' receptor types"
        )
        # Whether each cell has a receptor, and its type's code, as hashed and
        # sorted from a Python string for each cell. Clonotypes across
        # receptor types code the types too, to link each type's cells apart
        # as well.
        ledger.hold("whether each cell has a receptor", n_cells)
        with_receptor = receptor_type != NO_RECEPTOR
        ledger.hold("each cell's receptor type code", 8 * n_cells)
        ledger.peak(
            "coding the cells' receptor types",
            _coding_bytes(
                n_cells, stored_types.dtype.itemsize // 4, len(RECEPTOR_TYPES)
            ),
        )
        _types, type_codes = _sorted_codes(receptor_type)
        types_bytes = receptor_type.nbytes
        del receptor_type
        ledger.release(types_bytes)
        within_codes, within_labels, fingerprint = self._within_groups(
            cell_selection, columns, with_receptor, ledger
        )
        # Each cell's group among the receptor types and the within groups,
        # beside its within group, which the grouping holds; across receptor
        # types, also its group among the within groups alone.
        ledger.hold("each cell's group", (8 if within_receptor_type else 16) * n_cells)
        typed_groups = np.where(
            with_receptor, type_codes * max(1, len(within_labels)) + within_codes, -1
        )
        groups = (
            typed_groups
            if within_receptor_type
            else np.where(with_receptor, within_codes, -1)
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
            max_distance=limit,
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
                # Clonotypes across receptor types save each cell's clonotype
                # within its type, as within_receptor_type=True defines it,
                # which every clone summary measures.
                **(
                    {}
                    if within_receptor_type
                    else {CODES_WITHIN_TYPE: ((n_cells,), "i")}
                ),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("n_clonotypes",),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        ledger.hold(
            "writing the clonotypes",
            _clonotype_write_bytes(
                n_cells, columns, within_labels, within_types=not within_receptor_type
            ),
        )
        ledger.hold(
            "the chains and clonotypes of each cell",
            _CLONOTYPE_BYTES_PER_CELL * n_cells,
        )
        if metric != "identity":
            # A cluster search holds its encoded junctions, one pass of keys
            # with their candidate pairs, and one chunk of candidate cells.
            ledger.hold("the search for near junctions", CLUSTER_SEARCH_BYTES)
        slots = read_whole(ledger, group, "chain_slots", item="each cell's chain slots")
        # Each compared field's code of every contig is read, one field at a
        # time.
        compared = _slot_codes(ledger, group, field_name, slots)
        v_genes = _slot_codes(ledger, group, "v_call", slots) if same_v_gene else None
        j_genes = _slot_codes(ledger, group, "j_call", slots) if same_j_gene else None
        if metric != "identity":
            # Only the junctions that the slots compare are decoded, so the
            # contigs that no slot holds, such as non-productive ones, cost
            # nothing, and the slots are coded by their position among them.
            # The decoded junctions are held while the search and the
            # linking run, which get what the budget leaves beside them.
            present = compared >= 0
            compared_codes, local = np.unique(compared[present], return_inverse=True)
            compared[present] = local.reshape(-1)
            texts = _decoded_values(ledger, group, field_name, compared_codes)

        def assigned(grouping: np.ndarray) -> ClonotypeAssignment:
            if metric == "identity":
                return assign_clonotypes(
                    compared,
                    groups=grouping,
                    v_genes=v_genes,
                    j_genes=j_genes,
                    receptor_arms=receptor_arms,
                    dual_ir=dual_ir,
                )
            return assign_clonotype_clusters(
                compared,
                texts,
                groups=grouping,
                metric=metric,
                max_distance=limit,
                v_genes=v_genes,
                j_genes=j_genes,
                receptor_arms=receptor_arms,
                dual_ir=dual_ir,
                ledger=ledger,
            )

        assignment = assigned(groups)
        within_type = None
        if not within_receptor_type:
            # The clonotypes across receptor types are held while each type's
            # cells are linked apart, as within_receptor_type=True links them,
            # so every summary can measure each cell's clonotype within its
            # type, whose code is kept to be written.
            ledger.hold(
                "the clonotypes across receptor types",
                assignment.codes.nbytes
                + assignment.sizes.nbytes
                + assignment.configurations.nbytes
                + assignment.configuration_clonotypes.nbytes,
            )
            within_type = assigned(typed_groups).codes
            ledger.hold(
                "each cell's clonotype within its receptor type", within_type.nbytes
            )
        del typed_groups
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
        if within_type is not None:
            arrays[CODES_WITHIN_TYPE] = within_type.astype(np.int32)
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
            # Clonotypes are clones, so they save what every source's clones do.
            diagnostics=clone_diagnostics(assignment.codes, assignment.sizes),
        )
        defined = clonotype_nouns(metric)[0]
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
                f"clonotypes must be a clonotypes artifact, not {quoted_value(clonotypes.kind)}"
            )
        status = require_complete_artifact(self.zw, clonotypes)
        if set(status.parameters or {}) != _CLONOTYPE_PARAMETERS:
            raise ValueError(
                "clonotypes was saved by an earlier version of CyteArc, whose "
                "parameters this version names otherwise; define the clonotypes "
                "again with ds.repertoire.clonotypes, which saves them anew"
            )
        return artifact_group(self.zw, clonotypes)

    def _representative_junctions(
        self, clonotypes: ArtifactRef, codes: np.ndarray, *, ledger: MemoryLedger
    ) -> np.ndarray:
        """The junction amino acids that name some clonotypes in figures.

        A clonotype is named by its representative cell's ``VDJ_1`` chain, or
        its ``VJ_1`` chain without one; clonotypes defined by
        ``receptor_arms="VJ"`` by the ``VJ_1`` chain first, which their
        cells share. Of the receptors, only the representatives' slots and
        junction codes, the bounds of the junctions they name, a chunk at a
        time or whole where that holds less, and those junctions' bytes are
        read, each array a physical chunk at a time. Each read, and the
        decoding, is charged to the caller's ``ledger`` beside what it holds
        and what earlier reads keep, which the junctions returned leave
        charged.

        Args:
            clonotypes: A complete ``clonotypes`` artifact.
            codes: The positions of some of its clonotypes, such as 0 for
                ``clonotype_1``.
            ledger: The caller's memory ledger.

        Returns:
            Each clonotype's junction, None where the chain that names it
            has no junction amino-acid sequence, as an object array.

        Raises:
            MemoryBudgetError: If a read does not fit the memory budget.
        """
        group = self._clonotype_group(clonotypes)
        status = inspect_artifact(self.zw, clonotypes)
        receptor_group = artifact_group(self.zw, status.input_ref("receptors"))
        naming = ("VJ_1", "VDJ_1")
        if (status.parameters or {}).get("receptor_arms") != "VJ":
            naming = ("VDJ_1", "VJ_1")
        wanted, back = np.unique(np.asarray(codes, dtype=np.int64), return_inverse=True)
        cells = read_rows(
            ledger,
            as_zarr_array(group["representatives"], name="representatives"),
            wanted,
            item="their representatives",
        ).astype(np.int64)
        distinct_cells, cell_back = np.unique(cells, return_inverse=True)
        stored_slots = read_rows(
            ledger,
            as_zarr_array(receptor_group["chain_slots"], name="chain_slots"),
            distinct_cells,
            item="their representatives' chain slots",
        )
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
            junction_codes[present] = read_rows(
                ledger,
                as_zarr_array(
                    receptor_group["contig_junction_aa"], name="contig_junction_aa"
                ),
                distinct_contigs,
                item="their junctions' codes",
            ).astype(np.int64)[contig_back.reshape(-1)]
        junctions = np.full(len(wanted), None, dtype=object)
        named = junction_codes >= 0
        if named.any():
            # Only the bytes of the junctions needed are read and decoded.
            needed, inverse = np.unique(junction_codes[named], return_inverse=True)
            text = _decoded_values(ledger, receptor_group, "junction_aa", needed)
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
            cell and the name of its ``within`` group, and each ``within``
            group's labels in ``group_keys``.

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
        MemoryLedger(
            self.memoryBytes, f"Loading the clonotypes of {n_cells} cells"
        ).peak("the tables of the cells and the clonotypes", resident)
        representatives = read_array(group, "representatives").astype(np.int64)
        slots = read_array(receptor_group, "chain_slots")[representatives]
        # Only the representatives' contigs are decoded.
        contigs = slots.reshape(-1)
        local = np.where(contigs >= 0, np.arange(len(contigs)), -1)
        chains = {
            name: _stored_text_at(receptor_group, name, contigs)
            for name in REPRESENTATIVE_FIELDS
            if name in fields
        }
        within_codes = read_array(group, "clonotype_within").astype(np.int64)
        # Each within group's labels, row by row; without within columns,
        # every clonotype is in one group without labels.
        within_keys = (
            [
                dict(zip(within, (str(label) for label in row), strict=True))
                for row in read_array(group, "within_labels")
                .reshape(-1, len(within))
                .tolist()
            ]
            if within
            else [{}]
        )
        keys = group_key_table(within, within_keys)
        names = keys.index.to_numpy(dtype=object)
        # Clonotypes defined without within columns record no group, -1.
        clonotype_groups = (
            names[within_codes] if within else np.repeat(names, len(within_codes))
        )
        cells, table = clonotype_frames(
            # The estimate above counts the IDs' read, which no budget limits
            # here.
            artifact_cell_ids(
                MemoryLedger(None, f"Loading the clonotypes of {n_cells} cells"),
                self.zw,
                self.cells._get_array("ids"),
                plan_artifact_cells(self.zw, clonotypes),
            ),
            codes=read_array(group, "codes"),
            clone_size=read_array(group, "clone_size"),
            clonotype_ids=read_array(group, "clonotype_ids"),
            clonotype_sizes=read_array(group, "clonotype_sizes"),
            n_configurations=read_array(group, "n_configurations"),
            receptor_type=read_array(receptor_group, "values")[representatives],
            groups=clonotype_groups,
            slots=local.reshape(slots.shape),
            chains=chains,
        )
        return Clonotypes(
            cells=cells,
            clonotypes=table,
            group_keys=keys,
            receptors=receptors,
            sequence=str(parameters["sequence"]),
            receptor_arms=str(parameters["receptor_arms"]),
            dual_ir=str(parameters["dual_ir"]),
            same_v_gene=bool(parameters["same_v_gene"]),
            same_j_gene=bool(parameters["same_j_gene"]),
            within=within,
            within_receptor_type=bool(parameters["within_receptor_type"]),
            metric=str(parameters["metric"]),
            max_distance=int(parameters["max_distance"]),
        )

    def _slot_units(
        self,
        source: ArtifactRef,
        fields: tuple[str, ...],
        *,
        slots: Any,
        count: Any,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        cell_selection: ArtifactRef | None,
        what: str,
        chosen_by: str,
    ) -> tuple[
        pd.DataFrame,
        dict[str, np.ndarray],
        tuple[str, ...],
        zarr.Group,
        MemoryLedger,
        dict[str, int],
    ]:
        """Each counted chain of the chosen slots, with its fields' stored codes.

        A ``clonotypes`` source counts only cells with a clonotype, and with
        ``count="clonotypes"`` each clonotype once per group and value;
        clonotypes defined across receptor types count by each cell's
        clonotype within its type. The
        cells are grouped through :func:`cell_groups`. Reading the slots and
        the clonotypes, grouping the cells and holding the chains are each
        charged to a memory ledger beside what the earlier steps hold, and so
        is the counting that follows, before the chains are found.

        Returns:
            The groups' keys, one row per group, indexed by its name and
            with its labels in the key columns; for each chain, ``group``,
            ``cell``, the cell's
            row, the stored codes of ``locus`` and of each field, and
            ``clonotype``, the clonotype code, with a clonotypes source; the
            slots; the ``immune_receptors`` artifact's group; the ledger
            of what the step holds; and the grouping's diagnostics, whose
            ``cellsWithoutClone`` counts the cells kept without a chain in
            the slots, or without a clonotype.
        """
        if isinstance(slots, str):
            slots = (slots,)
        if not isinstance(slots, tuple | list) or not slots:
            raise TypeError("slots must be a slot name or a sequence of slot names")
        chosen = tuple(slots)
        if any(slot not in SLOTS for slot in chosen) or len(set(chosen)) != len(chosen):
            raise ValueError(f"slots must hold distinct slots from {', '.join(SLOTS)}")
        if count not in ("cells", "clonotypes"):
            raise ValueError("count must be 'cells' or 'clonotypes'")
        if not isinstance(source, ArtifactRef):
            raise TypeError("source must be an ArtifactRef")
        codes_name = "codes"
        if source.kind == ClonotypeArguments.artifact_kind:
            opened = self._clonotype_group(source)
            clonotype_group: zarr.Group | None = opened
            status = inspect_artifact(self.zw, source)
            receptors = status.input_ref("receptors")
            # Clonotypes across receptor types count each cell's clonotype
            # within its type, as every clone summary measures them.
            if measured_within_types(source, status, opened):
                codes_name = CODES_WITHIN_TYPE
        elif source.kind == ImmuneReceptorImportArguments.artifact_kind:
            if count == "clonotypes":
                raise ValueError("count='clonotypes' needs a clonotypes artifact")
            clonotype_group = None
            receptors = source
        else:
            raise ValueError(
                "source must be an immune_receptors or clonotypes artifact, not "
                f"{quoted_value(source.kind)}"
            )
        group, stored = self._receptor_group(receptors)
        if "locus" not in stored:
            raise ValueError("The receptors store no locus")
        for name in fields:
            if name not in stored:
                raise ValueError(
                    f"{chosen_by} reads each chain's {name}, which these receptors "
                    f"do not store; they store {', '.join(stored)}"
                )
        stored_slots = as_zarr_array(group["chain_slots"], name="chain_slots")
        n_cells = int(stored_slots.shape[0])
        kept = len(chosen) * stored_slots.dtype.itemsize
        # The selections whose cells the grouping reads are checked once,
        # outside the budget, before anything is charged.
        receptor_cells = inspect_artifact(self.zw, receptors).input_ref(
            "cell_selection"
        )
        validate_cell_selection(self.zw, receptor_cells)
        if cell_selection is not None:
            validate_cell_selection(self.zw, cell_selection)
        # Reading the slots and the codes of every contig is sized by every
        # cell and contig of the receptors; the chains and their counting by
        # the chains of the cells and slots counted.
        ledger = MemoryLedger(
            self.memoryBytes, f"{what} of {n_cells} cells", codec_threads=self.nthreads
        )
        # The slots are read whole, beside the chosen ones, whether each cell
        # holds a chain in them, and the cells with one.
        all_slots = read_whole(
            ledger, group, "chain_slots", item="each cell's chain slots"
        )
        ledger.peak("the chosen slots", n_cells * (kept + len(chosen) + 1))
        slot_contigs = all_slots[:, [SLOTS.index(slot) for slot in chosen]]
        rows = (slot_contigs >= 0).any(axis=1)
        slots_bytes = all_slots.nbytes
        del all_slots
        ledger.release(slots_bytes)
        ledger.hold("the chosen slots", slot_contigs.nbytes + rows.nbytes)
        codes = None
        if clonotype_group is not None:
            # The clonotypes' codes, with their int64 copy and the mask of the
            # cells with a clonotype.
            stored_codes = read_whole(
                ledger, clonotype_group, codes_name, item="each cell's clonotype"
            )
            ledger.peak("each cell's clonotype as int64", 9 * n_cells)
            codes = stored_codes.astype(np.int64)
            rows &= codes >= 0
            # The copy is held before the stored codes are let go.
            ledger.hold("each cell's clonotype as int64", codes.nbytes)
            codes_bytes = stored_codes.nbytes
            del stored_codes
            ledger.release(codes_bytes)
        grouped = cell_groups(
            self.zw,
            self.cells,
            receptor_cells,
            rows,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            ledger=ledger,
        )
        # Each group's row of the keys, beside each cell's group and the
        # groups' labels, which the grouping holds, and the cells with a
        # chain until they are let go.
        ledger.hold(
            "the table of the groups' labels",
            len(grouped.keys)
            * (
                _KEY_BYTES_PER_GROUP
                + _KEY_BYTES_PER_GROUP_COLUMN * len(grouped.key_columns)
            ),
        )
        del rows
        ledger.release(n_cells)
        keys = group_key_table(grouped.key_columns, grouped.keys)
        # Each chain of a grouped cell in a chosen slot is counted, and the
        # stored codes of each contig's text field are read one field at a
        # time.
        ledger.peak("the chains of the grouped cells", (2 * len(chosen) + 1) * n_cells)
        counted = (grouped.codes >= 0)[:, None] & (slot_contigs >= 0)
        n_chains = int(np.count_nonzero(counted))
        n_fields = 1 + len(fields) + (codes is not None)
        ledger.remedy = (
            "count fewer chains, of a smaller cell_selection or of fewer slots"
        )
        ledger.hold(
            f"the {n_chains} chains counted",
            n_chains * (_CHAIN_BYTES + _CHAIN_BYTES_PER_FIELD * n_fields),
        )
        ledger.peak("the chains counted", len(counted.reshape(-1)))
        cells, positions = np.nonzero(counted)
        del counted
        contigs = slot_contigs[cells, positions]
        del positions
        units = {"group": grouped.codes[cells], "cell": cells}
        for name in ("locus", *fields):
            units[name] = _slot_codes(ledger, group, name, contigs)
        if codes is not None:
            units["clonotype"] = codes[cells]
        return keys, units, chosen, group, ledger, grouped.diagnostics

    def _value_names(
        self, ledger: MemoryLedger, group: zarr.Group, name: str, codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode the distinct values of a stored text field among some codes.

        Only the values that the codes name are decoded, through
        :func:`_decoded_values`, which charges each read to ``ledger``.

        Returns:
            Each code's position among the distinct values, -1 for a missing
            value, and those values, in the order of their text.
        """
        offsets = as_zarr_array(group[f"contig_{name}_offsets"], name=name)
        n_values = int(offsets.shape[0]) - 1
        ledger.peak(f"ranking the {name} values", _RANK_BYTES_PER_VALUE * n_values)
        local, needed = _value_ranks(codes, n_values)
        if not len(needed):
            return local, np.empty(0, dtype=object)
        return local, _decoded_values(ledger, group, name, needed)

    def _value_measures(
        self, ledger: MemoryLedger, group: zarr.Group, name: str, codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measure the distinct values of a stored text field among some codes.

        Only the values that the codes name are measured, through
        :func:`_value_lengths`, which charges each read to ``ledger``.

        Returns:
            Each code's length in characters, -1 for a missing value, and
            every length up to the longest, the names of those lengths.
        """
        offsets = as_zarr_array(group[f"contig_{name}_offsets"], name=name)
        n_values = int(offsets.shape[0]) - 1
        ledger.peak(f"ranking the {name} values", _RANK_BYTES_PER_VALUE * n_values)
        local, needed = _value_ranks(codes, n_values)
        lengths = _value_lengths(ledger, group, name, needed)
        measured = np.full(len(local), -1, dtype=np.int64)
        known = local >= 0
        measured[known] = lengths[local[known]]
        return measured, np.arange(int(lengths.max(initial=-1)) + 1, dtype=np.int64)

    def _check_count_budget(
        self,
        ledger: MemoryLedger,
        keys: pd.DataFrame,
        n_chains: int,
        widths: list[int],
        once: bool,
    ) -> None:
        """Charge the counting and its table to ``ledger`` before they run.

        The table holds a row for each distinct group, locus and values that
        a chain holds, at most one per chain and per combination of codes, of
        the group's name, the locus, the values, the count and the fraction.
        Counting each clonotype, or each cell, once also holds each chain's
        row with its clonotype or cell among the distinct rows.
        """
        n_rows = min(n_chains, math.prod([len(keys), *widths]))
        n_columns = 1 + len(widths) + 2
        needed = n_rows * (_ROW_BYTES + _ROW_BYTES_PER_COLUMN * n_columns)
        if once:
            needed += n_chains * (
                _ONCE_BYTES_PER_CHAIN + _ONCE_BYTES_PER_COLUMN * (len(widths) + 2)
            )
        ledger.peak("counting the chains and their table", needed)

    @validation_scoped
    def _repertoire_gene_usage(
        self,
        source: ArtifactRef,
        *,
        genes: str | Sequence[str] = "v_call",
        level: str = "gene",
        slots: str | Sequence[str] = ("VJ_1", "VDJ_1"),
        count: str = "cells",
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
    ) -> GeneUsage:
        """Count the genes of the chains of groups of cells.

        Each cell counts by its chains in ``slots``, by default its primary
        chain of each arm, ``VJ_1`` and ``VDJ_1``, as scirpy reads a slot's
        gene calls; each chain counts within its locus, so TRA and TRB genes
        get their own shares. ``genes`` names one gene field, such as
        ``"v_call"``, or several, such as ``("v_call", "j_call")`` for the
        V-J pairs of each chain. ``level`` names calls as called, with their
        allele (``"allele"``), as genes (``"gene"``, the default) or as
        families (``"family"``), and a call naming several genes keeps its
        first. A chain without a call is not counted.

        ``source`` is an ``immune_receptors`` artifact, whose cells with a
        chain count, or a ``clonotypes`` artifact, whose cells with a
        clonotype count; ``count="clonotypes"`` then counts each clonotype
        once per group and gene instead of each cell, which keeps expanded
        clones from dominating the shares, and clonotypes defined across
        receptor types by each cell's clonotype within its type, as every
        clone summary measures them. The groups are the combinations of
        the labels of ``group_by`` columns and of a ``groups`` label artifact,
        among the cells of ``cell_selection``; without either, every cell is
        one group. Nothing is saved.

        Args:
            source: An ``immune_receptors`` or ``clonotypes`` artifact.
            genes: One gene field or several, from ``v_call``, ``d_call``,
                ``j_call`` and ``c_call``.
            level: ``"allele"``, ``"gene"`` or ``"family"``.
            slots: The chain slots to count.
            count: ``"cells"`` or ``"clonotypes"``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell.

        Returns:
            One row per group, locus and gene, or gene combination, and the
            groups' labels in ``group_keys``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``source`` is not an ``immune_receptors`` or
                ``clonotypes`` artifact, an option is unknown, a gene field is
                not stored, ``count="clonotypes"`` is asked of receptors, a
                column of the groups is missing or named twice, two key columns
                share a name, or two groups get one name.
            KeyError: If the store holds no such ``source`` artifact, or not
                the receptors of the clonotypes.
            RuntimeError: If the ``source`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before
                it finished leaves them.
            MemoryError: If reading the chain slots, grouping the cells,
                holding the chains and counting them, or decoding the gene
                calls does not fit the memory budget.
        """
        fields = (genes,) if isinstance(genes, str) else genes
        if not isinstance(fields, tuple | list) or not fields:
            raise TypeError("genes must be a gene field or a sequence of gene fields")
        fields = tuple(fields)
        if any(name not in GENE_FIELDS for name in fields) or len(set(fields)) != len(
            fields
        ):
            raise ValueError(
                f"genes must hold distinct fields from {', '.join(GENE_FIELDS)}"
            )
        gene_names(np.asarray([], dtype=object), level)
        keys, units, chosen, group, ledger, grouping = self._slot_units(
            source,
            fields,
            slots=slots,
            count=count,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            what="Counting the genes of the chains",
            chosen_by="Counting gene usage",
        )
        # Each distinct call is decoded and named once.
        named = {
            name: gene_codes(
                *self._value_names(ledger, group, name, units.pop(name)), level
            )
            for name in fields
        }
        # The chains without a counted call, a flag per chain and its
        # comparison, which count_chains leaves out.
        ledger.peak("finding the chains without a call", 2 * len(units["cell"]))
        without_value = np.zeros(len(units["cell"]), dtype=bool)
        for gene_codes_of_chains, _names in named.values():
            without_value |= np.asarray(gene_codes_of_chains) < 0
        n_without_value = int(np.count_nonzero(without_value))
        del without_value
        loci = self._value_names(ledger, group, "locus", units.pop("locus"))
        once_per_cell = count == "cells" and _shares_an_arm(chosen)
        self._check_count_budget(
            ledger,
            keys,
            len(units["cell"]),
            [len(loci[1]), *(len(names) for _codes, names in named.values())],
            once=count == "clonotypes" or once_per_cell,
        )
        table, n_cells = count_chains(
            keys.index,
            units["group"],
            units["cell"],
            loci,
            named,
            count,
            clonotypes=units.get("clonotype"),
            once_per_cell=once_per_cell,
        )
        return GeneUsage(
            table=table,
            group_keys=keys,
            source=source,
            genes=fields,
            level=level,
            slots=chosen,
            count=count,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            n_cells=n_cells,
            diagnostics={**grouping, "chainsWithoutValue": n_without_value},
        )

    @validation_scoped
    def _repertoire_spectratype(
        self,
        source: ArtifactRef,
        *,
        sequence: str = "aa",
        slots: str | Sequence[str] = ("VJ_1", "VDJ_1"),
        count: str = "cells",
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        cell_selection: ArtifactRef | None = None,
    ) -> Spectratype:
        """Count the junction lengths of the chains of groups of cells.

        The spectratype of a repertoire is the distribution of its junction
        lengths, in residues (``sequence="aa"``) or nucleotides (``"nt"``),
        as scirpy's ``spectratype`` counts them; a narrow peak points to an
        expanded clone, and a repertoire of naive cells has a smooth,
        roughly Gaussian one. Junctions include their conserved cysteine and
        phenylalanine or tryptophan, as Cell Ranger's CDR3 does. Cells,
        slots, counts and groups are those of
        :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.gene_usage`.
        Nothing is saved.

        Args:
            source: An ``immune_receptors`` or ``clonotypes`` artifact.
            sequence: ``"aa"`` or ``"nt"``.
            slots: The chain slots to count.
            count: ``"cells"`` or ``"clonotypes"``.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            cell_selection: Only these cells, or None for every cell.

        Returns:
            One row per group, locus and length, and the groups' labels in
            ``group_keys``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``source`` is not an ``immune_receptors`` or
                ``clonotypes`` artifact, an option is unknown, the junctions
                are not stored, ``count="clonotypes"`` is asked of receptors,
                a column of the groups is missing or named twice, two key
                columns share a name, or two groups get one name.
            KeyError: If the store holds no such ``source`` artifact, or not
                the receptors of the clonotypes.
            RuntimeError: If the ``source`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before
                it finished leaves them.
            MemoryError: If reading the chain slots, grouping the cells,
                holding the chains and counting them, or measuring the
                junctions does not fit the memory budget.
        """
        if sequence not in SEQUENCES:
            raise ValueError(f"sequence must be one of {', '.join(SEQUENCES)}")
        field_name = "junction_aa" if sequence == "aa" else "junction"
        keys, units, chosen, group, ledger, grouping = self._slot_units(
            source,
            (field_name,),
            slots=slots,
            count=count,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            what="Counting the junction lengths of the chains",
            chosen_by=f"sequence={quoted_value(sequence)}",
        )
        # Each distinct junction is measured once, and a chain without one is
        # not counted.
        measured, names = self._value_measures(
            ledger, group, field_name, units.pop(field_name)
        )
        # A chain without a junction, which count_chains leaves out, is
        # counted from a comparison of each chain's length.
        ledger.peak("finding the chains without a junction", len(measured))
        n_without_value = int(np.count_nonzero(measured < 0))
        loci = self._value_names(ledger, group, "locus", units.pop("locus"))
        once_per_cell = count == "cells" and _shares_an_arm(chosen)
        self._check_count_budget(
            ledger,
            keys,
            len(units["cell"]),
            [len(loci[1]), len(names)],
            once=count == "clonotypes" or once_per_cell,
        )
        # Rows follow the groups' natural order, then the locus and length.
        table, n_cells = count_chains(
            keys.index,
            units["group"],
            units["cell"],
            loci,
            {"length": (measured, names)},
            count,
            clonotypes=units.get("clonotype"),
            by_count=False,
            once_per_cell=once_per_cell,
        )
        table["length"] = table["length"].astype(np.int64)
        return Spectratype(
            table=table,
            group_keys=keys,
            source=source,
            sequence=sequence,
            slots=chosen,
            count=count,
            group_by=group_columns(group_by),
            groups=groups,
            cell_selection=cell_selection,
            n_cells=n_cells,
            diagnostics={**grouping, "chainsWithoutValue": n_without_value},
        )

    @validation_scoped
    def _repertoire_export_airr(
        self,
        source: ArtifactRef,
        path: str | os.PathLike[str],
        *,
        fields: Sequence[str] | None = None,
        chains_only: bool = False,
    ) -> ExportSummary:
        """Write imported contigs as an AIRR Rearrangement file.

        The AIRR Community's Rearrangement TSV is what Immcantation
        (Change-O, SHazaM, dowser), dandelion and scirpy read, so exported
        contigs carry a CyteArc analysis to their clonal lineage, somatic
        hypermutation or other analyses. Each row is one contig: its
        ``cell_id``, the cell's ID in this store, the AIRR fields that the
        receptors store, and with a ``clonotypes`` artifact as ``source``,
        ``clone_id``, the cell's clonotype label, empty for a cell without
        one, replacing any imported ``clone_id``. The schema's required
        fields come first, empty when the receptors do not store them;
        logical fields are written ``T`` and ``F``, and other values as they
        are, unquoted, as the AIRR file format asks. The format forbids tabs
        and newlines in values, so a value that holds a tab, a newline or a
        carriage return is refused before the file is opened; it advises
        against ``@``, ``#``, ``"`` and ``'``, so the export warns with the
        number of values that hold them and their fields, and apart, in every
        field, with the values that start with a double quote, which readers
        that take quotes read as an opening quote. A ``sequence_id``
        that repeats, such as ``<barcode>_contig_1`` of the TCR and the BCR
        files of one cell, gets ``_2``, ``_3`` and so on from its second
        occurrence, in the order of the contigs, since the schema requires
        unique identifiers. A path ending in ``.gz`` is gzip-compressed at
        level 6.

        The contigs are read and written in blocks of about 32 MiB, at most a
        sixteenth of the memory budget, as many contigs as fit when each
        field holds its longest value. Beside one block, the export holds the
        cells' IDs and clonotypes, the distinct values of the text fields and
        the repeated IDs' new names, and until it has named the repeated IDs,
        a hash of each contig's ID. Every stored array is read a chunk at a
        time, and what each step reads and holds is checked against the
        memory budget before the step runs, all before the file is written;
        each check counts the stored chunk read last, which Zarr may still
        hold a moment after its read returns.

        Args:
            source: An ``immune_receptors`` or ``clonotypes`` artifact.
            path: The file to write.
            fields: The stored fields to write, or None for every stored
                field; ``sequence_id`` and ``cell_id``, and with clonotypes
                ``clone_id``, are always written.
            chains_only: Write only the contigs that count as chains.

        Returns:
            The file written, its ``rows``, the contigs written, and its
            ``diagnostics``: ``contigs``, ``renamedContigIds``, the repeated
            contig IDs written with a suffix, ``leadingQuoteValues`` and
            ``discouragedValues``, the values that start with a double quote
            and those that hold a character that the format advises against,
            and ``leadingQuoteFields`` and ``discouragedFields``, their
            fields. Each of the three counts of values and IDs above 0 gives a
            warning.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``source`` is not an ``immune_receptors`` or
                ``clonotypes`` artifact, a field is not stored, or a value to
                be written holds a tab, a newline or a carriage return.
            KeyError: If the store holds no such ``source`` artifact, or not
                the receptors of the clonotypes.
            RuntimeError: If the ``source`` artifact or the receptors of the
                clonotypes are incomplete, as an import that stopped before
                it finished leaves them.
            MemoryError: If a step of the export would hold more than the
                memory budget.
        """
        chains_only = bool_argument(chains_only, "chains_only")
        from ...writers.airr import (
            AIRR_EXPORT_DIAGNOSTICS,
            AIRR_EXPORT_RULES,
            AIRR_REQUIRED_FIELDS,
            AirrRearrangementWriter,
            suffix_repeated_ids,
        )

        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path")
        if not isinstance(source, ArtifactRef):
            raise TypeError("source must be an ArtifactRef")
        clonotypes = source.kind == ClonotypeArguments.artifact_kind
        clonotype_group: zarr.Group | None = None
        if clonotypes:
            clonotype_group = self._clonotype_group(source)
            receptors = inspect_artifact(self.zw, source).input_ref("receptors")
        elif source.kind == ImmuneReceptorImportArguments.artifact_kind:
            receptors = source
        else:
            raise ValueError(
                "source must be an immune_receptors or clonotypes artifact, not "
                f"{quoted_value(source.kind)}"
            )
        if fields is not None:
            if isinstance(fields, str) or not isinstance(fields, Sequence):
                raise TypeError("fields must be a sequence of field names or None")
            # Every row needs its contig ID, and every file its cells' IDs and,
            # with clonotypes, their labels, which no stored field holds.
            always = {"cell_id", "clone_id"} if clonotypes else {"cell_id"}
            fields = [
                "sequence_id",
                *[
                    name
                    for name in fields
                    if name != "sequence_id" and name not in always
                ],
            ]
        group, stored = self._receptor_group(receptors)
        if fields is None:
            chosen = stored
        else:
            unknown = [name for name in fields if name not in stored]
            if unknown:
                raise ValueError(
                    "The receptors store no field "
                    + ", ".join(quoted_value(str(name)) for name in unknown)
                    + "; they store "
                    + ", ".join(stored)
                )
            chosen = tuple(name for name in stored if name in set(fields))
        columns = ["cell_id", *chosen]
        if clonotypes and "clone_id" not in columns:
            columns.append("clone_id")
        leading = [
            name
            for name in ("cell_id", "clone_id", "locus", "c_call", "umi_count")
            if name in columns
        ]
        order = [*leading, *[name for name in columns if name not in leading]]
        n_cells = int(as_zarr_array(group["values"], name="values").shape[0])
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Exporting the {_n_contigs(group)} contigs of {n_cells} cells with "
            f"{len(chosen)} fields",
            remedy="pass fields= with fewer fields",
            codec_threads=self.nthreads,
        )
        ledger.hold(
            "the export's objects and the file's buffer",
            _EXPORT_OBJECT_BYTES + _EXPORT_OBJECT_BYTES_PER_FIELD * len(chosen),
        )
        # A clonotypes artifact's labels replace an imported clone_id.
        contigs = _ExportContigs(
            group,
            tuple(name for name in chosen if not (clonotypes and name == "clone_id")),
            chains_only=chains_only,
            ledger=ledger,
        )
        block_bytes = max(1, min(_EXPORT_BLOCK_BYTES, self.memoryBytes // 16))
        # Every stored array is read a chunk at a time, as the blocks count.
        with zarr_io_concurrency(1), zarr_decoding_threads(ledger.codec_threads):
            contigs.hold_text()
            clones: EncodedText | None = None
            clone_width = 0
            if clonotype_group is not None:
                # Each cell's clonotype is its code among the clonotypes'
                # labels, which follow the receptors' cells.
                codes = read_whole(
                    ledger, clonotype_group, "codes", item="each cell's clonotype"
                )
                labels, clone_width = _packed_text(
                    ledger,
                    as_zarr_array(
                        clonotype_group["clonotype_ids"], name="clonotype_ids"
                    ),
                    None,
                    block_bytes,
                    item="the clonotypes' labels",
                )
                clones = EncodedText(codes, labels.data, labels.offsets)
            cells, cell_width = self._export_cell_ids(
                ledger, receptors, n_cells, block_bytes
            )
            contigs.measure(block_bytes, cells=cells, clones=clones)
            hashes, id_rows = contigs.scan(block_bytes)
            # Sorting the hashes and finding those that repeat.
            naming = ledger.hold(
                "finding the contig IDs that repeat",
                _NAMING_BYTES_PER_ROW * len(hashes),
            )
            per_id = _ID_BYTES + _ID_BYTES_PER_BYTE * contigs.width("sequence_id")
            # The pass that reads the repeated IDs holds a block of IDs, their
            # bytes and their offsets as read and searched.
            passing = contigs.id_pass_bytes(id_rows) + (
                contigs.width("sequence_id") + 16
            ) * min(id_rows, contigs.n_contigs)
            asked = 0

            def ids_at(rows: np.ndarray) -> np.ndarray:
                # The IDs read earlier stay held while later ones are read.
                nonlocal asked
                asked += len(rows)
                ledger.peak(
                    f"naming {asked} repeated contig IDs", passing + per_id * asked
                )
                return contigs.ids_at(rows, id_rows)

            renamed = suffix_repeated_ids(hashes, ids_at)
            del hashes
            ledger.release(naming + contigs.hash_bytes)
            ledger.hold("the repeated IDs' new names", per_id * len(renamed[0]))
            block_rows = contigs.plan_blocks(
                n_columns=len(AIRR_REQUIRED_FIELDS)
                + sum(name not in AIRR_REQUIRED_FIELDS for name in order),
                cell_width=cell_width,
                clone_width=clone_width if clones is not None else None,
                block_bytes=block_bytes,
                compressed=os.fspath(path).endswith(".gz"),
            )
            with AirrRearrangementWriter(path, order, whole=contigs.whole) as writer:
                for block in contigs.blocks(block_rows, cells, clones, renamed):
                    writer.write(block)
                    # The block is let go before the next is read.
                    del block
        written = writer.n_rows
        logger.info(f"Wrote {written} contigs to {os.fspath(path)}")
        return export_summary(
            "export_airr",
            path,
            written,
            {**writer.diagnostics, "renamedContigIds": len(renamed[0])},
            AIRR_EXPORT_DIAGNOSTICS,
            AIRR_EXPORT_RULES,
        )

    def _export_cell_ids(
        self,
        ledger: MemoryLedger,
        receptors: ArtifactRef,
        n_cells: int,
        slice_bytes: int,
    ) -> tuple[EncodedText, int]:
        """The IDs of the receptors' cells as packed text, for an AIRR export.

        The positions of the receptors' cell selection are read, then the IDs
        a slice at a time, as :func:`_packed_text` reads them; each step is
        charged to ``ledger`` first, and the packed IDs stay charged.

        Returns:
            The IDs, whose codes are unused, and the bytes of the longest.
        """
        selection = inspect_artifact(self.zw, receptors).input_ref("cell_selection")
        # The receptors hold the selection's cells, so its identity check runs
        # within the read that the ledger charges.
        cell_index = read_cell_selection(
            ledger,
            self.zw,
            plan_cell_selection_read(self.zw, selection, cells=n_cells),
            item="the positions of the receptors' cells",
        )
        positions = int(cell_index.nbytes)
        packed = _packed_text(
            ledger,
            self.cells._get_array("ids"),
            cell_index,
            slice_bytes,
            item="the cells' IDs",
        )
        del cell_index
        ledger.release(positions)
        return packed


# An AIRR export renders its contigs in blocks of about this many bytes, at
# most a sixteenth of the memory budget, a whole number of blocks per stored
# chunk.
_EXPORT_BLOCK_BYTES = 32 * 2**20
# The rendered bytes of an integer, whose missing values are negative, of a
# logical value, and of a real number as repr writes it.
_INTEGER_TEXT_BYTES = 19
_FLAG_TEXT_BYTES = 1
_REAL_TEXT_BYTES = 24
# Bytes per contig that a block holds for each field, beside the text that it
# reads and renders: a text field's codes as read and as int64, a ragged
# field's offsets as read, searched and renumbered, an integer's codes and
# mask with its digits' offsets, a logical value's codes and flags, and a
# number's offsets.
_EXPORT_FIELD_BYTES = {TEXT: 12, RAGGED: 24, INTEGER: 17, FLAG: 11, NUMBER: 16}
# Bytes per contig that rendering one field holds while it runs, for a number
# beside four per byte of its widest text: its values, flags and text as
# Python objects, and their joined bytes.
_EXPORT_RENDER_BYTES = {TEXT: 0, RAGGED: 0, INTEGER: 9, FLAG: 10, NUMBER: 136}
# Bytes per contig of a block's rows: their cells, positions, widths and
# cursors, and one field's lengths while they are summed; and of each cell's
# clonotype, its code as gathered and as int64.
_EXPORT_ROW_BYTES = 96
_EXPORT_CLONE_BYTES = 12
# The Python objects of the stored arrays that an export opens, their
# metadata, codecs and reads, and of the file and a block's columns while it
# is written: 106 KB with two fields and 247 KB with 22, as traced on Python
# 3.13. The file's buffer is counted as one of io's default buffers: Python
# 3.13 buffers the file system's block, 4 KiB on common file systems, and
# Python 3.14 one default buffer, which it raised from 8 KiB to 128 KiB.
_EXPORT_OBJECT_BYTES = 120 * 2**10 + io.DEFAULT_BUFFER_SIZE
_EXPORT_OBJECT_BYTES_PER_FIELD = 8 * 2**10
# Writing a compressed file holds zlib's deflate state at level 6, about
# 270 KB as traced, and gzip's buffer of four of io's default buffers, 32 KiB
# before Python 3.14 and 512 KiB since.
_GZIP_BYTES = 352 * 2**10 + 4 * io.DEFAULT_BUFFER_SIZE
# Bytes per contig that measuring a span of contigs holds: a ragged field's
# offsets, running ends and lengths, or a number field's values and flags.
_MEASURE_BYTES_PER_ROW = 64
# Hashing a block's IDs holds about this many bytes per contig, beside this
# many per byte of the longest ID: the IDs as read, as Python strings and as
# their UTF-8 bytes for hashing, and their offsets and hashes.
_SCAN_BYTES_PER_ROW = 300
_SCAN_BYTES_PER_ID_BYTE = 6
# Finding the hashes that repeat holds this many bytes per contig beside the
# hashes: their sorted copy, and the positions and matches of a search.
_NAMING_BYTES_PER_ROW = 40
# Naming a repeated ID holds about this many bytes, beside this many per byte
# of the longest ID: its text, code, occurrence and new name.
_ID_BYTES = 200
_ID_BYTES_PER_BYTE = 4


def _stored_chunk_bytes(array: zarr.Array) -> int:
    """The bytes of one stored chunk of an array as stored and decoded, 0 without chunks."""
    geometry = array_geometry(array)
    return 0 if geometry is None else geometry.storedChunkBytes()


def _n_contigs(group: zarr.Group) -> int:
    return int(as_zarr_array(group["contig_cells"], name="contig_cells").shape[0])


def _even_rows(chunk: int, wanted: int) -> int:
    """Rows of blocks that divide a stored chunk evenly, at most ``wanted``."""
    wanted = max(1, min(chunk, wanted))
    return -(-chunk // -(-chunk // wanted))


def _chunk_reads(ledger: MemoryLedger, chunk: int) -> int:
    """What reads of stored chunks of ``chunk`` bytes, one after another, hold beside what the ledger counts.

    A read holds the chunk that it decodes while the chunk read before it may
    still be held, so reads hold two chunks at once: the one that the
    ledger's last read left behind, or one of their own where that is larger,
    and the one that they decode.
    """
    return chunk + max(0, chunk - ledger.left)


def _packed_text(
    ledger: MemoryLedger,
    array: zarr.Array,
    rows: np.ndarray | None,
    slice_bytes: int,
    *,
    item: str,
) -> tuple[EncodedText, int]:
    """Read rows of a stored text array as packed UTF-8 bytes, a slice at a time.

    Each slice is read through :func:`~cytearc.metadata.rows.read_rows`, then
    packed, which holds its joined text, the text's UTF-8 bytes and their
    copy, up to four bytes per character each, and its lengths and offsets.
    The packed slices are joined once all are read. Each step is charged to
    ``ledger`` first, and the packed text stays charged.

    Args:
        ledger: What the caller holds, and the budget.
        array: A stored text array.
        rows: The rows to read, in this order, or None for every row.
        slice_bytes: About the most bytes that reading and packing a slice
            holds; a slice holds at least one row.
        item: What the rows are, as refusals name them.

    Returns:
        The text, whose codes are unused, and the bytes of its longest value.
    """
    n_rows = int(array.shape[0]) if rows is None else len(rows)
    itemsize = int(np.dtype(array.dtype).itemsize)
    packing = 3 * itemsize + 24
    step = max(1, slice_bytes // (itemsize + ROW_INDEX_BYTES + packing))
    ledger.hold(f"the offsets of {item}", 8 * (n_rows + 1))
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    parts = []
    longest = 0
    for start in range(0, n_rows, step):
        stop = min(start + step, n_rows)
        chosen = np.arange(start, stop) if rows is None else rows[start:stop]
        values = read_rows(ledger, array, chosen, item=item)
        ledger.peak(f"packing {item}", (stop - start) * packing)
        data, bounds = pack_text(values)
        offsets[start + 1 : stop + 1] = bounds[1:] + offsets[start]
        longest = max(longest, int(np.diff(bounds).max()))
        del bounds
        parts.append(data)
        # The packed slice is charged before the stored rows are let go.
        ledger.hold(item, int(data.nbytes))
        values_bytes = values.nbytes
        del values, chosen
        ledger.release(values_bytes)
    # The packed slices, and the bytes that they are joined into.
    ledger.peak(f"joining {item}", int(offsets[-1]))
    joined = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)
    del parts
    return EncodedText(np.empty(0, dtype=np.int32), joined, offsets), longest


class _Window:
    """Read a stored array in ascending slices, one stored chunk at a time."""

    def __init__(self, array: zarr.Array) -> None:
        self._array = array
        self._chunk = max(1, int(array.chunks[0]))
        self._start = 0
        self._empty = np.asarray(array[0:0])
        self._values = self._empty

    def read(self, start: int, stop: int) -> np.ndarray:
        if start < self._start or stop > self._start + len(self._values):
            first = start - start % self._chunk
            last = min(-(-stop // self._chunk) * self._chunk, self._array.shape[0])
            # The chunks read before are let go before the next are read.
            self._values = self._empty
            self._values = np.asarray(self._array[first:last])
            self._start = first
        return self._values[start - self._start : stop - self._start]


class _RaggedWindow:
    """Read a stored ragged field in consecutive blocks of contigs."""

    def __init__(self, group: zarr.Group, name: str) -> None:
        self._data = as_zarr_array(group[f"contig_{name}"], name=name)
        self._ends = _Window(as_zarr_array(group[f"contig_{name}_offsets"], name=name))
        self._end = 0

    def _bounds(self, start: int, stop: int) -> np.ndarray:
        ends = self._ends.read(start + 1, stop + 1)
        bounds = np.empty(len(ends) + 1, dtype=np.int64)
        bounds[0] = self._end
        # A missing value's end is -1: it ends where the value before it does.
        np.maximum.accumulate(np.maximum(ends, self._end), out=bounds[1:])
        return bounds

    def read(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the block after the last one read, from contig 0 on.

        Returns:
            The bytes of the block's values, and the offsets of each value in
            them, the last its end; a missing value is empty.
        """
        bounds = self._bounds(start, stop)
        data = np.asarray(self._data[self._end : int(bounds[-1])])
        bounds -= self._end
        self._end += int(bounds[-1])
        return data, bounds

    def skip(self, start: int, stop: int) -> None:
        """Pass over the block after the last one read, reading only its ends."""
        self._end = int(self._bounds(start, stop)[-1])


def _kept(values: np.ndarray, kept: np.ndarray | None) -> np.ndarray:
    """Return a block's values of the rows of the file, all without ``kept``."""
    return values if kept is None else values[kept]


class _ExportContigs:
    """The contigs of an ``immune_receptors`` artifact, read for an AIRR export.

    Each pass reads blocks of contigs in order from the first; with
    ``chains_only``, a block's rows are its contigs that count as chains.
    Each step charges what it reads and holds to ``ledger`` first, and what
    it keeps stays charged there.
    """

    def __init__(
        self,
        group: zarr.Group,
        fields: tuple[str, ...],
        *,
        chains_only: bool,
        ledger: MemoryLedger,
    ) -> None:
        self._group = group
        self._fields = fields
        self._chains_only = chains_only
        self.ledger = ledger
        cells = self._array("contig_cells")
        self.n_contigs = int(cells.shape[0])
        self._chunk = max(1, int(cells.chunks[0]))
        self._texts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._widths: dict[str, int] = {}
        #: The number fields whose rows hold only whole numbers, which are
        #: written as integers.
        self.whole: set[str] = set()

    @property
    def hash_bytes(self) -> int:
        """The bytes of the hashes of the contigs' IDs."""
        return 8 * self.n_contigs

    def _array(self, name: str) -> zarr.Array:
        return as_zarr_array(self._group[name], name=name)

    def width(self, name: str) -> int:
        """The rendered bytes of a field's widest value."""
        return self._widths[name]

    def hold_text(self) -> None:
        """Read the distinct values of the text fields, which every block uses."""
        for name in self._fields:
            if FIELDS[name] != TEXT:
                continue
            data = read_whole(
                self.ledger,
                self._group,
                f"contig_{name}_values",
                item=f"the distinct {name} values",
            )
            offsets = read_whole(
                self.ledger,
                self._group,
                f"contig_{name}_offsets",
                item=f"the bounds of the distinct {name} values",
            )
            # Each value's length, while the longest is found.
            self.ledger.peak(f"measuring the {name} values", int(offsets.nbytes))
            self._widths[name] = int(np.diff(offsets).max()) if len(offsets) > 1 else 0
            self._texts[name] = (data, offsets)

    def measure(
        self, span_bytes: int, *, cells: EncodedText, clones: EncodedText | None
    ) -> None:
        """Find the widest value of each field and the whole number fields.

        The ragged fields' offsets and bytes and the number fields' values are
        read a span of contigs at a time, at most a stored chunk of contigs,
        after one check of what a span holds; a span's bytes are read a stored
        chunk at a time. Every value that the file would hold is checked for
        a tab, a newline or a carriage return, which the AIRR format forbids:
        the ragged fields' bytes as they are read, and the distinct values of
        the text fields, the cells' IDs and the clonotypes' labels once each;
        only for a distinct value that holds one are the codes or the cells of
        the contigs read, to find the contigs that would write it. With
        ``chains_only``, only the chains' values count. Call ``hold_text``
        first.

        Args:
            span_bytes: About the most bytes that measuring a span holds.
            cells: Each cell's ID, as the blocks write it.
            clones: Each cell's clonotype label, or None to write none.

        Raises:
            ValueError: If a value that the file would hold has a tab, a
                newline or a carriage return. The error names each such field,
                the number of its values that hold one and the ``sequence_id``
                of the first contig whose value does.
        """
        from ...writers.airr import _NO_CODES, _forbidden_message, _mark_coded_rows

        ragged = [name for name in self._fields if FIELDS[name] == RAGGED]
        numbers = [name for name in self._fields if FIELDS[name] == NUMBER]
        for name in self._fields:
            if FIELDS[name] == INTEGER:
                self._widths[name] = _INTEGER_TEXT_BYTES
            elif FIELDS[name] == FLAG:
                self._widths[name] = _FLAG_TEXT_BYTES
        coded = self._forbidden_codes(cells, clones)
        read = [
            *(f"contig_{name}_offsets" for name in ragged),
            *(f"contig_{name}" for name in ragged),
            *(f"contig_{name}" for name in numbers),
            *(["contig_chain"] if self._chains_only else []),
            *(array for array, _marked, _through in coded.values()),
        ]
        span = max(1, min(self._chunk, span_bytes // _MEASURE_BYTES_PER_ROW))
        chunk = self._read_bytes(read)
        self.ledger.peak(
            "measuring the widest value of each field",
            _chunk_reads(self.ledger, chunk)
            + self._piece_bytes(ragged)
            + _MEASURE_BYTES_PER_ROW * min(span, self.n_contigs),
        )
        ends = dict.fromkeys(ragged, 0)
        longest = dict.fromkeys(ragged, 0)
        whole = dict.fromkeys(numbers, True)
        largest = dict.fromkeys(numbers, 0.0)
        # The values of each field that hold a forbidden character, and the
        # first contig whose value does.
        counts: dict[str, int] = {}
        firsts: dict[str, int] = {}

        def measure_spans() -> None:
            """Measure each span of contigs, whose arrays are let go on return."""
            for first in range(0, self.n_contigs, span):
                measure_span(first)

        def count(
            name: str, first: int, marks: np.ndarray, kept: np.ndarray | None
        ) -> None:
            chosen = _kept(marks, kept)
            n_marked = int(np.count_nonzero(chosen))
            if n_marked:
                counts[name] = counts.get(name, 0) + n_marked
                row = int(np.argmax(chosen))
                firsts.setdefault(
                    name, first + (row if kept is None else int(kept[row]))
                )

        def measure_span(first: int) -> None:
            last = min(first + span, self.n_contigs)
            kept = (
                np.flatnonzero(np.asarray(self._array("contig_chain")[first:last]))
                if self._chains_only
                else None
            )
            marks = np.zeros(last - first, dtype=np.bool_)
            for name in ragged:
                stored = np.asarray(
                    self._array(f"contig_{name}_offsets")[first + 1 : last + 1]
                )
                bounds = np.empty(last - first + 1, dtype=np.int64)
                bounds[0] = ends[name]
                # A missing value's end is -1: it ends where the one before does.
                np.maximum.accumulate(np.maximum(stored, ends[name]), out=bounds[1:])
                del stored
                ends[name] = int(bounds[-1])
                lengths = _kept(np.diff(bounds), kept)
                if len(lengths):
                    longest[name] = max(longest[name], int(lengths.max()))
                del lengths
                marks[:] = False
                self._mark_ragged(name, bounds, marks)
                count(name, first, marks, kept)
            for name, (array, marked, through) in coded.items():
                values = np.asarray(self._array(array)[first:last])
                marks[:] = False
                _mark_coded_rows(
                    values,
                    marked,
                    _NO_CODES if through is None else through,
                    through is not None,
                    marks,
                )
                del values
                count(name, first, marks, kept)
            for name in numbers:
                if not whole[name]:
                    continue
                values = _kept(
                    np.asarray(self._array(f"contig_{name}")[first:last]), kept
                )
                present = values[~np.isnan(values)]
                whole[name] = bool(
                    np.isfinite(present).all() and (present == np.round(present)).all()
                )
                if whole[name] and len(present):
                    largest[name] = max(largest[name], float(np.abs(present).max()))

        measure_spans()
        if self.n_contigs:
            self.ledger.leave(self._held_bytes(read))
        self._widths.update(longest)
        for name in numbers:
            # A whole number is written as an integer, with its sign.
            self._widths[name] = (
                len(str(int(largest[name]))) + 1 if whole[name] else _REAL_TEXT_BYTES
            )
        self.whole = {name for name, flag in whole.items() if flag}
        if counts:
            found = sorted(counts, key=firsts.__getitem__)
            raise ValueError(
                _forbidden_message(
                    [
                        (name, counts[name], self._contig_id(firsts[name]))
                        for name in found
                    ],
                    "the contigs",
                )
            )
        # The codes of values that no contig of the file holds are let go.
        released = sum(8 * len(marked) for _, marked, _ in coded.values())
        coded.clear()
        self.ledger.release(released)

    def _forbidden_codes(
        self, cells: EncodedText, clones: EncodedText | None
    ) -> dict[str, tuple[str, np.ndarray, np.ndarray | None]]:
        """Find the distinct values that hold a tab, a newline or a carriage return.

        The text fields' values, the cells' IDs and the clonotypes' labels are
        each checked once; the codes of the values that hold one are checked
        first and stay counted in ``reads``.

        Returns:
            For each field with such a value, the stored array that gives each
            contig's code or cell, the codes of those values, ascending, and
            for clonotypes, each cell's clonotype code, through which a
            contig's cell names its label.
        """
        from ...writers.airr import _count_forbidden_values, _forbidden_values

        sources: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None]] = [
            (name, f"contig_{name}", *self._texts[name], None)
            for name in self._fields
            if FIELDS[name] == TEXT
        ]
        sources.append(("cell_id", "contig_cells", cells.data, cells.offsets, None))
        if clones is not None:
            sources.append(
                (
                    "clone_id",
                    "contig_cells",
                    clones.data,
                    clones.offsets,
                    np.asarray(clones.codes),
                )
            )
        coded = {}
        for name, array, data, offsets, through in sources:
            n_values = _count_forbidden_values(data, offsets)
            if n_values:
                self.ledger.hold(
                    f"the codes of the {name} values that the format forbids",
                    8 * n_values,
                )
                coded[name] = (
                    array,
                    _forbidden_values(data, offsets, n_values),
                    through,
                )
        return coded

    def _piece_bytes(self, ragged: list[str]) -> int:
        """The most bytes of a ragged field that reading a stored chunk gives."""
        held = 0
        for name in ragged:
            data = self._array(f"contig_{name}")
            rows = min(max(1, int(data.chunks[0])), int(data.shape[0]))
            held = max(held, rows * data.dtype.itemsize)
        return held

    def _mark_ragged(self, name: str, bounds: np.ndarray, marks: np.ndarray) -> None:
        """Mark the rows whose value of a ragged field holds a forbidden character.

        The bytes ``bounds[0]:bounds[-1]`` are read a stored chunk at a time.
        """
        from ...writers.airr import _mark_forbidden_rows

        data = self._array(f"contig_{name}")
        step = max(1, int(data.chunks[0]))
        start, end = int(bounds[0]), int(bounds[-1])
        while start < end:
            stop = min(end, (start // step + 1) * step)
            piece = np.asarray(data[start:stop])
            _mark_forbidden_rows(piece, start, bounds, marks)
            # A piece is let go before the next is read.
            del piece
            start = stop

    def _contig_id(self, position: int) -> str:
        """Read the ``sequence_id`` of one contig, checked first."""
        names = ["contig_sequence_id_offsets", "contig_sequence_id"]
        self.ledger.peak(
            "the sequence_id of a contig whose value the format forbids",
            _chunk_reads(self.ledger, self._read_bytes(names))
            + 16
            + self.width("sequence_id"),
        )
        start, end = np.asarray(self._array(names[0])[position : position + 2]).tolist()
        return bytes(np.asarray(self._array(names[1])[start:end])).decode("utf-8")

    def _window_bytes(self, names: list[tuple[str, int]], block_rows: int) -> int:
        """The most bytes that windows on stored arrays hold for blocks of contigs.

        Args:
            names: Each array, and the rows by which a block's rows are after
                its contigs' positions: 1 for the ends of ragged values.
            block_rows: The contigs of a block.
        """
        held = 0
        for name, shift in names:
            array = self._array(name)
            chunk = max(1, int(array.chunks[0]))
            if shift == 0 and chunk % self._chunk == 0:
                # Blocks stay within a stored chunk of contigs, so within one
                # of the array's.
                touched = 1
            else:
                touched = 1 + -(-max(block_rows - 1, 1) // chunk)
            held += min(touched * chunk, int(array.shape[0])) * array.dtype.itemsize
        return held

    def _read_bytes(self, names: list[str]) -> int:
        """What reading one stored chunk of the arrays holds, the most of any."""
        return max(
            (_stored_chunk_bytes(self._array(name)) for name in names), default=0
        )

    def _held_bytes(self, names: list[str]) -> int:
        """What a pass over the stored chunks of the arrays may leave behind.

        Each of the ledger's codec threads can still hold a chunk of one of
        them, up to the chunks of the arrays, each counted at the largest.
        """
        geometries = [
            geometry
            for name in names
            if (geometry := array_geometry(self._array(name))) is not None
        ]
        threads = self.ledger.holding_threads(
            sum(
                -(-geometry.shape[0] // geometry.axisChunk(0)) * geometry.bandChunks()
                for geometry in geometries
            )
        )
        return max(
            (geometry.heldReadBytes(threads) for geometry in geometries), default=0
        )

    def _id_arrays(self) -> list[tuple[str, int]]:
        windows = [("contig_sequence_id_offsets", 1)]
        if self._chains_only:
            windows.append(("contig_chain", 0))
        return windows

    def id_pass_bytes(self, block_rows: int) -> int:
        """What a pass over the contigs' IDs holds beside their own blocks."""
        windows = self._id_arrays()
        return self._window_bytes(windows, block_rows) + _chunk_reads(
            self.ledger,
            self._read_bytes([name for name, _ in windows] + ["contig_sequence_id"]),
        )

    def scan(self, block_bytes: int) -> tuple[np.ndarray, int]:
        """Hash the ID of each row of the file, checked first.

        Returns:
            The ``sequence_id_hashes`` of the rows, whose ``hash_bytes`` stay
            counted, and the contigs of a block of IDs.
        """
        from ...writers.airr import sequence_id_hashes

        per_row = _SCAN_BYTES_PER_ROW + _SCAN_BYTES_PER_ID_BYTE * self.width(
            "sequence_id"
        )
        block_rows = _even_rows(self._chunk, block_bytes // per_row)
        self.ledger.hold("the hashes of the contigs' IDs", self.hash_bytes)
        self.ledger.peak(
            "hashing a block of the contigs' IDs",
            per_row * min(block_rows, self.n_contigs) + self.id_pass_bytes(block_rows),
        )
        hashes = np.empty(self.n_contigs, dtype=np.uint64)
        ids = _RaggedWindow(self._group, "sequence_id")
        n_rows = 0
        for start, stop, kept in self._spans(block_rows):
            block = sequence_id_hashes(
                _kept(decode_ragged(*ids.read(start, stop)), kept)
            )
            hashes[n_rows : n_rows + len(block)] = block
            n_rows += len(block)
        if self.n_contigs:
            self.ledger.leave(
                self._held_bytes(
                    [name for name, _ in self._id_arrays()] + ["contig_sequence_id"]
                )
            )
        return hashes[:n_rows], block_rows

    def plan_blocks(
        self,
        *,
        n_columns: int,
        cell_width: int,
        clone_width: int | None,
        block_bytes: int,
        compressed: bool,
    ) -> int:
        """Choose the contigs of a block, and check what writing one holds.

        A block holds as many contigs as fit in ``block_bytes`` when each
        field holds its widest value: each row as rendered, the text that it
        reads and renders beside its row, and the bytes of each field and of
        the rendering of one field. The windows of stored chunks, two chunks'
        reads and, for a compressed file, its compressor come beside it.

        Args:
            n_columns: The columns of the file.
            cell_width: The bytes of the longest cell ID.
            clone_width: The bytes of the longest clonotype label, or None
                when no clonotype is written.
            block_bytes: The most bytes that a block's contigs should hold.
            compressed: Whether the file is gzip-compressed.

        Returns:
            The contigs of a block.
        """
        kinds = [FIELDS[name] for name in self._fields]
        rendered = (
            n_columns
            + cell_width
            + (clone_width or 0)
            + sum(self._widths[name] for name in self._fields)
        )
        beside = sum(
            self._widths[name]
            for name in self._fields
            if FIELDS[name] in (RAGGED, INTEGER, NUMBER)
        )
        working = (
            _EXPORT_ROW_BYTES
            + (_EXPORT_CLONE_BYTES if clone_width is not None else 0)
            + sum(_EXPORT_FIELD_BYTES[kind] for kind in kinds)
        )
        rendering = max(
            (
                _EXPORT_RENDER_BYTES[FIELDS[name]]
                + (4 * self._widths[name] if FIELDS[name] == NUMBER else 0)
                for name in self._fields
            ),
            default=0,
        )
        per_row = rendered + beside + working + rendering
        block_rows = _even_rows(self._chunk, block_bytes // per_row)
        windows = [("contig_cells", 0)]
        if self._chains_only:
            windows.append(("contig_chain", 0))
        for name, kind in zip(self._fields, kinds, strict=True):
            windows.append(
                (f"contig_{name}_offsets", 1)
                if kind == RAGGED
                else (f"contig_{name}", 0)
            )
        self.ledger.peak(
            "writing a block of contigs",
            per_row * min(block_rows, self.n_contigs)
            + (_GZIP_BYTES if compressed else 0)
            + self._window_bytes(windows, block_rows)
            + _chunk_reads(
                self.ledger,
                self._read_bytes(
                    [name for name, _ in windows]
                    + [
                        f"contig_{name}"
                        for name, kind in zip(self._fields, kinds)
                        if kind == RAGGED
                    ]
                ),
            ),
        )
        return block_rows

    def _spans(self, block_rows: int) -> Iterator[tuple[int, int, np.ndarray | None]]:
        """Yield each block's contigs and the rows of the file among them."""
        chain = _Window(self._array("contig_chain")) if self._chains_only else None
        for first in range(0, self.n_contigs, self._chunk):
            last = min(first + self._chunk, self.n_contigs)
            for start in range(first, last, block_rows):
                stop = min(start + block_rows, last)
                yield (
                    start,
                    stop,
                    None if chain is None else np.flatnonzero(chain.read(start, stop)),
                )

    def ids_at(self, rows: np.ndarray, block_rows: int) -> np.ndarray:
        """Return the IDs of rows, given their positions in ascending order."""
        rows = np.asarray(rows, dtype=np.int64)
        found = np.empty(len(rows), dtype=object)
        ids = _RaggedWindow(self._group, "sequence_id")
        done = before = 0
        for start, stop, kept in self._spans(block_rows):
            if done == len(rows):
                break
            n_rows = stop - start if kept is None else len(kept)
            upto = int(np.searchsorted(rows, before + n_rows))
            if upto > done:
                data, bounds = ids.read(start, stop)
                local = rows[done:upto] - before
                found[done:upto] = take_text(
                    data, bounds, local if kept is None else kept[local]
                )
                done = upto
            else:
                ids.skip(start, stop)
            before += n_rows
        return found

    def blocks(
        self,
        block_rows: int,
        cells: EncodedText,
        clones: EncodedText | None,
        renamed: tuple[np.ndarray, np.ndarray],
    ) -> Iterator[dict[str, Any]]:
        """Yield the columns of each block's rows, for the AIRR writer.

        Args:
            block_rows: The contigs of a block.
            cells: Each cell's ID; its codes are unused.
            clones: Each cell's clonotype label, or None to write none.
            renamed: The rows whose ID is written with a suffix, ascending,
                and their new IDs.
        """
        windows = _BlockWindows(
            positions=_Window(self._array("contig_cells")),
            codes={
                name: _Window(self._array(f"contig_{name}"))
                for name in self._fields
                if FIELDS[name] == TEXT
            },
            ragged={
                name: _RaggedWindow(self._group, name)
                for name in self._fields
                if FIELDS[name] == RAGGED
            },
            values={
                name: _Window(self._array(f"contig_{name}"))
                for name in self._fields
                if FIELDS[name] not in (TEXT, RAGGED)
            },
        )
        before = 0
        for start, stop, kept in self._spans(block_rows):
            block = self._block(
                windows, start, stop, kept, before, cells, clones, renamed
            )
            yield block
            # The block is let go before the next is read.
            del block
            before += stop - start if kept is None else len(kept)

    def _block(
        self,
        windows: "_BlockWindows",
        start: int,
        stop: int,
        kept: np.ndarray | None,
        before: int,
        cells: EncodedText,
        clones: EncodedText | None,
        renamed: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, Any]:
        """The columns of one block's rows, which follow ``before`` rows."""
        rows = np.arange(stop - start) if kept is None else kept
        cell = _kept(windows.positions.read(start, stop), kept)
        block: dict[str, Any] = {
            "cell_id": EncodedText(cell, cells.data, cells.offsets)
        }
        if clones is not None:
            block["clone_id"] = EncodedText(
                np.asarray(clones.codes)[cell], clones.data, clones.offsets
            )
        for name, window in windows.codes.items():
            data, offsets = self._texts[name]
            block[name] = EncodedText(
                _kept(window.read(start, stop), kept), data, offsets
            )
        renamed_rows, renamed_ids = renamed
        for name, reader in windows.ragged.items():
            data, bounds = reader.read(start, stop)
            codes = rows
            if name == "sequence_id":
                first, last = np.searchsorted(
                    renamed_rows, [before, before + len(rows)]
                )
                if last > first:
                    # A renamed row takes a value after the block's.
                    extra = EncodedText.from_categories(
                        np.empty(0, np.int32), renamed_ids[first:last]
                    )
                    data = np.concatenate([data, extra.data])
                    bounds = np.concatenate([bounds, extra.offsets[1:] + bounds[-1]])
                    codes = rows.copy()
                    codes[renamed_rows[first:last] - before] = (
                        stop - start + np.arange(last - first)
                    )
            block[name] = EncodedText(codes, data, bounds)
        for name, window in windows.values.items():
            stored = _kept(window.read(start, stop), kept)
            if FIELDS[name] == INTEGER:
                block[name] = pd.arrays.IntegerArray(stored, stored < 0)
            elif FIELDS[name] == FLAG:
                block[name] = pd.arrays.BooleanArray(stored == 1, stored < 0)
            else:
                block[name] = stored
        return block


@dataclass(frozen=True, slots=True)
class _BlockWindows:
    """The windows that an AIRR export's blocks read their contigs through."""

    positions: _Window
    codes: dict[str, _Window]
    ragged: dict[str, _RaggedWindow]
    values: dict[str, _Window]
