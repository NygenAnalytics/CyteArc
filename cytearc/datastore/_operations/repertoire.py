import hashlib
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    ClonalExpansionArguments,
    ClonotypeArguments,
    ImmuneReceptorImportArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    read_metadata_missing_rows,
    read_metadata_rows,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import (
    category_codes,
    resolve_grouping,
    sample_codes,
)
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
    RECEPTOR_ARMS,
    REPRESENTATIVE_FIELDS,
    SEQUENCES,
    Clonotypes,
    assign_clonotype_clusters,
    assign_clonotypes,
    clonotype_frames,
    clonotype_labels,
    representative_cells,
)
from ...repertoire.contigs import (
    FIELDS,
    RAGGED,
    TEXT,
    ContigEncoder,
    ContigTable,
    EncodedText,
    chunk_rows_for,
    decode_ragged,
    decode_text,
    take_text,
)
from ...repertoire.distances import METRICS
from ...repertoire.diversity import (
    INDEX_COLUMNS,
    ORDERS,
    CloneSizes,
    ClonalDiversity,
    Rarefaction,
    default_sizes,
    diversity_tables,
    rarefaction_curves,
)
from ...repertoire.expansion import (
    DEFAULT_BREAKPOINTS,
    DEFAULT_RANK_SPLITS,
    ClonalExpansion,
    checked_bounds,
    expansion_categories,
    expansion_tables,
    frequency_labels,
    group_clone_sizes,
)
from ...repertoire.groups import group_names
from ...repertoire.overlap import (
    ClonotypeOverlap,
    ClonotypeTracking,
    group_frame,
    overlap_extent,
    overlap_pairs,
    pair_counts,
    top_clonotypes,
    tracking_cells,
    tracking_reach,
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
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arrays import sorted_unique, unique_rows
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _RepertoireOperationsBase
else:
    _RepertoireOperationsBase = object

# Barcodes of the contig files that are not selected cells, named in the
# artifact's attributes and in the warning.
_UNMATCHED_EXAMPLES = 5
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
# At its peak, while it orders the contigs, ranks the chains and writes the
# artifact, an import holds about four times the bytes of its encoded contigs,
# as measured on 200,000 and 1,000,000 cells.
_PEAK_PER_ENCODED_BYTE = 4
# The barcode lookup holds each selected cell's ID as text with a hash slot.
_LOOKUP_BYTES_PER_CELL = 100
# Defining clonotypes holds, at its peak, about 900 bytes per cell: the chains
# of every slot, their distinct configurations and keys, and the labels, as
# measured on 1,000,000 cells.
_CLONOTYPE_BYTES_PER_CELL = 1_000
# The columns of the tables of clonotypes, which a within column may not share.
_CLONOTYPE_COLUMNS = frozenset(
    {"clonotype", "clone_size", "n_cells", "fraction", "n_configurations"}
    | {"receptor_type"}
)
# Bytes that a decoded contig field takes in a table, per contig, besides the
# text of ragged fields.
_TABLE_BYTES_PER_FIELD = 64
# The bytes of each chunk of a contig array. Packed text and per-contig codes
# are read and written whole, which chunks of megabytes make fast; an export
# holds one chunk of each array it reads.
_CONTIG_CHUNK_BYTES = 2 * 2**20
# Measuring clonal expansion holds, at its peak, about 270 bytes per cell: the
# clonotype and group codes, the clone sizes and frequencies, and the labels
# as written, as measured on 1,000,000 cells.
_EXPANSION_BYTES_PER_CELL = 300
# The columns of the tables of clonal expansion, which a within column may not
# share; plots add a "group" column.
_EXPANSION_COLUMNS = frozenset(
    {"group", "clonotype", "category", "group_clone_size", "group_clone_frequency"}
    | {"cells", "clonotypes", "fraction", "window"}
)
# The tables of loaded clonotypes and clonal expansion hold about 850 and 450
# bytes per cell at their peak, with the cell IDs and labels as text, as
# measured on 1,000,000 cells.
_CLONOTYPE_TABLE_BYTES_PER_CELL = 1_000
_EXPANSION_TABLE_BYTES_PER_CELL = 500
# The column of each group's receptor type in the tables of clonal expansion.
_RECEPTOR_TYPE_COLUMN = "receptor_type"
# Grouping cells holds, at its peak, about these bytes per cell and per column
# of labels: the cells' positions, codes and groups, and each column's labels
# while they are coded; 84 bytes per cell without labels and 206 with four
# columns, as measured on 1,000,000 cells.
_GROUPING_BYTES_PER_CELL = 100
_GROUPING_BYTES_PER_COLUMN = 70
# The columns of each grouped result besides the groups' labels, which a
# column of labels may not share; "group" names each group.
_DIVERSITY_COLUMNS = frozenset(
    {"group", *INDEX_COLUMNS, "order", "observed", "estimated", "at_size"}
    | {"coverage_at_size", "method_at_size", "at_coverage", "size_at_coverage"}
    | {"method_at_coverage"}
)
_RAREFACTION_COLUMNS = frozenset(
    {"group", "order", "size", "method", "diversity", "coverage"}
)
_OVERLAP_COLUMNS = frozenset({"group", "cells", "clonotypes"})
_TRACK_COLUMNS = frozenset({"group", "clonotype", "cells", "fraction"})
_USAGE_COLUMNS = frozenset(
    {"group", "locus", *GENE_FIELDS, "cells", "clonotypes", "fraction"}
)
_SPECTRATYPE_COLUMNS = frozenset(
    {"group", "locus", "length", "cells", "clonotypes", "fraction"}
)
# Comparing groups holds, at its peak, about 220 bytes per pair of groups of
# one stratum, the pairs' indices and their table, 16 bytes per product of
# two groups of the largest stratum, and 40 bytes per count of a clonotype in
# a group, as measured on 1,000 and 2,000 groups.
_OVERLAP_BYTES_PER_PAIR = 250
_OVERLAP_BYTES_PER_PRODUCT = 24
_OVERLAP_BYTES_PER_COUNT = 40
# Following clonotypes holds about 110 bytes per row of the table with three
# key columns, as measured on 1.3 million rows, and a byte per chosen
# clonotype and group.
_TRACK_BYTES_PER_ROW = 150


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ContigSource:
    """One contig file of an import, in the canonical order of sources."""

    path: str | os.PathLike[str]
    format: str
    sha256: str
    barcode_prefix: str | None
    n_columns: int

    def record(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "sha256": self.sha256,
            "barcode_prefix": self.barcode_prefix,
        }


def _contig_sources(paths: Any, barcode_prefix: Any) -> tuple[_ContigSource, ...]:
    """Check the files and prefixes of an import and order them canonically.

    Sources are ordered by the SHA-256 of their bytes, then by prefix, so the
    same files listed in another order give the same artifact.
    """
    from ...readers.airr import airr_columns, contig_file_format
    from ...readers.cellranger import vdj_contig_columns

    if isinstance(paths, str | os.PathLike):
        listed: list[Any] = [paths]
    elif isinstance(paths, Sequence) and not isinstance(paths, bytes):
        listed = list(paths)
    else:
        raise TypeError("paths must be a path or a sequence of paths")
    if not listed:
        raise ValueError("paths must name at least one contig file")
    for path in listed:
        if not isinstance(path, str | os.PathLike):
            raise TypeError("paths must be a path or a sequence of paths")
    if barcode_prefix is None or isinstance(barcode_prefix, str):
        prefixes: list[Any] = [barcode_prefix] * len(listed)
    elif isinstance(barcode_prefix, Sequence) and not isinstance(barcode_prefix, bytes):
        prefixes = list(barcode_prefix)
        if len(prefixes) != len(listed):
            raise ValueError(
                f"barcode_prefix lists {len(prefixes)} prefixes for {len(listed)} "
                "contig files; pass one prefix per file, or one for every file"
            )
    else:
        raise TypeError("barcode_prefix must be text, a sequence of text, or None")
    for prefix in prefixes:
        if prefix is not None and not isinstance(prefix, str):
            raise TypeError("barcode_prefix must be text, a sequence of text, or None")
    sources = []
    for path, prefix in zip(listed, prefixes, strict=True):
        file_format = contig_file_format(path)
        header = (
            vdj_contig_columns(path)
            if file_format == "cellranger"
            else airr_columns(path)
        )
        sources.append(
            _ContigSource(
                path=path,
                format=file_format,
                sha256=_file_sha256(path),
                barcode_prefix=prefix or None,
                n_columns=len(header),
            )
        )
    keys = [(source.sha256, source.barcode_prefix or "") for source in sources]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "A contig file is listed twice with the same barcode prefix; list "
            "each file once, or give each listing its own barcode_prefix"
        )
    order = sorted(range(len(sources)), key=keys.__getitem__)
    return tuple(sources[position] for position in order)


def _iter_source(
    source: _ContigSource, chunk_rows: int
) -> Iterator[tuple[int, pd.DataFrame]]:
    if source.format == "cellranger":
        from ...readers.cellranger import iter_vdj_contigs

        return iter_vdj_contigs(source.path, chunk_rows=chunk_rows)
    from ...readers.airr import iter_airr_rearrangements

    return iter_airr_rearrangements(source.path, chunk_rows=chunk_rows)


def _contig_arrays(table: ContigTable) -> dict[str, np.ndarray]:
    """Return the stored arrays of encoded contigs, keyed by array name.

    A text field ``f`` is stored as ``contig_f``, each contig's code, with the
    UTF-8 bytes of its distinct values in ``contig_f_values`` and their
    offsets in ``contig_f_offsets``. A sequence field is stored as
    ``contig_f``, the bytes of each contig's sequence, with
    ``contig_f_offsets``. Other fields are stored as ``contig_f``.
    """
    arrays: dict[str, np.ndarray] = {
        "contig_cells": np.asarray(table.cells, dtype=np.int32),
        "contig_sources": np.asarray(table.sources, dtype=np.int32),
    }
    # contig_cells, contig_sources and contig_chain are not fields.
    for name in table.fields:
        kind = FIELDS[name]
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


def _sorted_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted distinct labels and each label's code among them.

    The codes equal those of ``np.unique(values, return_inverse=True)``, by
    hashing rather than by sorting every label.
    """
    codes, distinct = pd.factorize(np.asarray(values), sort=True)
    return np.asarray(distinct), np.asarray(codes, dtype=np.int64)


def _labels_of(
    keys: list[dict[str, str]], column: str, codes: np.ndarray
) -> np.ndarray:
    """Each code's label in a column of the keys, None for -1."""
    labels = np.asarray([key.get(column) for key in keys], dtype=object)
    taken = np.full(len(codes), None, dtype=object)
    present = codes >= 0
    taken[present] = labels[codes[present]]
    return taken


def _contig_chunk_lengths(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    """Chunk each contig array in about 8 MiB, for whole-array reads and writes."""
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
        # The packed bytes are read once; only the needed values become text.
        text = take_text(
            _read(group, f"contig_{name}_values"),
            _read(group, f"contig_{name}_offsets"),
            needed,
        )
        values[codes >= 0] = text[inverse.reshape(-1)]
    return values


def _stored_fields(group: zarr.Group) -> tuple[str, ...]:
    fields = group.attrs["fields"]
    if not isinstance(fields, list) or not all(name in FIELDS for name in fields):
        raise ValueError("The immune_receptors artifact has no valid list of fields")
    return tuple(str(name) for name in fields)


def _within_columns(
    within: Any, name: str = "within", reserved: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    """Return the cell metadata columns of an argument as a tuple of names.

    A column may not share the name of a column of the result, in
    ``reserved``, which it would overwrite.
    """
    if within is None:
        return ()
    if isinstance(within, str):
        names: list[Any] = [within]
    elif isinstance(within, Sequence) and not isinstance(within, bytes):
        names = list(within)
    else:
        raise TypeError(f"{name} must be a column name, a sequence of names, or None")
    if not all(isinstance(column, str) and column for column in names):
        raise TypeError(f"{name} must be a column name, a sequence of names, or None")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} names a column twice")
    clashes = [column for column in names if column in reserved]
    if clashes:
        raise ValueError(
            f"{name} names {clashes[0]!r}, which is also a column of the result; "
            "copy the cell metadata column under another name"
        )
    return tuple(names)


def _slot_codes(group: zarr.Group, name: str, slots: np.ndarray) -> np.ndarray:
    """Return a text field's code in each cell's slots, -1 for an empty slot."""
    codes = _read(group, f"contig_{name}").astype(np.int64)
    taken = np.full(slots.shape, -1, dtype=np.int64)
    filled = slots >= 0
    taken[filled] = codes[slots[filled]]
    return taken


class _RepertoireOperationsMixin(_RepertoireOperationsBase):
    @validation_scoped
    def _repertoire_import_receptors(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        cell_selection: ArtifactRef,
        barcode_prefix: str | Sequence[str | None] | None = None,
        productive_only: bool = True,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import the immune receptor contigs of a set of cells.

        A V(D)J library, such as 10x Genomics Chromium Single Cell Immune
        Profiling, assembles each cell's T or B cell receptor transcripts into
        contigs. The import reads Cell Ranger's contig annotations,
        ``filtered_contig_annotations.csv`` or ``all_contig_annotations.csv``,
        or AIRR Rearrangement files, which Cell Ranger
        (``airr_rearrangement.tsv``), TRUST4, BD Rhapsody, IgBLAST with
        Change-O, MiXCR and dandelion write; each file's format is read from
        its header. It joins each contig to a selected cell by barcode, keeps
        every contig of the selected cells, and saves them as an
        ``immune_receptors`` artifact. Pass the TCR and BCR files of one
        library together, so a cell with both kinds of chain is found.

        Each cell's chains are ranked as scirpy ranks them, as the repertoire
        API reference describes: a contig counts as a chain when it
        has a locus of a receptor chain and a junction amino-acid sequence,
        is productive unless ``productive_only`` is False, and is not
        flagged by Cell Ranger as outside a cell or of low confidence. The
        first two chains of each arm fill the cell's ``VJ_1``, ``VJ_2``,
        ``VDJ_1`` and ``VDJ_2`` slots, and the chains give the cell's receptor
        type, subtype and chain pairing. A selected cell without a chain is
        ``"no IR"``.

        A contig's cell ID is its file's ``barcode_prefix`` followed by its
        barcode, matched exactly. A merged store prefixes each sample's
        barcodes with its name and two underscores, so pass each sample's
        files with that sample's prefix. Contigs whose barcode matches no
        selected cell are counted, saved in the artifact's attributes, and
        named in a warning. A file is identified by the SHA-256 of its bytes,
        so a moved file reuses the saved receptors and an edited file gives
        new ones, and the order of the files does not matter.

        The artifact's canonical ``values`` hold each cell's receptor type,
        and its other per-cell arrays are ``receptor_subtype``,
        ``chain_pairing``, ``multichain``, ``n_vj`` and ``n_vdj`` (its chains
        of each arm) and ``n_contigs``. ``select_cells`` reads them, so
        ``ds.select_cells(receptors, include=["TCR"])`` keeps the T cells
        with a receptor.

        Args:
            paths: One contig file, or several, plain or gzip-compressed.
            cell_selection: The cells that the contigs belong to.
            barcode_prefix: Text that precedes each barcode in the cell IDs,
                such as ``"sample1__"`` in a merged store: one for every
                file, or one per file. None adds nothing.
            productive_only: Count only productive contigs as chains.
            invalidate_cache: Import again instead of reusing saved receptors.

        Returns:
            A complete datastore-scoped ``immune_receptors`` artifact. Read it
            with
            :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.load_receptors`
            or plot it with ``ds.plots.receptor_qc``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a file is neither format, is malformed, repeats a
                contig ID, or is listed twice with the same prefix, or the
                selection holds no cell.
            PermissionError: If the store is read-only and the receptors are
                not saved yet.
            MemoryError: If the import would not fit the memory budget: it
                holds about four times the bytes of its encoded contigs at its
                peak.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(productive_only, bool):
            raise TypeError("productive_only must be a boolean")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        sources = _contig_sources(paths, barcode_prefix)
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
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
            required_attributes=(
                "fields",
                "n_file_contigs",
                "n_unmatched_contigs",
            ),
            reuse_validator=has_contigs,
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)

        # The lookup holds each selected cell's ID once, with a hash slot.
        lookup = pd.Index(
            np.asarray(self.cells.fetch_all("ids"))[cell_index].astype(object)
        )
        lookup_bytes = _LOOKUP_BYTES_PER_CELL * n_cells
        encoder = ContigEncoder()
        n_file_contigs = 0
        n_unmatched = 0
        # Unmatched barcodes are counted by their 64-bit hashes, which an
        # all_contig_annotations.csv of many barcodes keeps small.
        unmatched_hashes = np.empty(0, dtype=np.uint64)
        examples: list[str] = []
        for position, source in enumerate(sources):
            prefix = source.barcode_prefix or ""
            chunk_rows = chunk_rows_for(self.memoryBytes, columns=source.n_columns)
            for _first_line, chunk in _iter_source(source, chunk_rows):
                barcodes = chunk["cell_id"].to_numpy(dtype=object).astype(str)
                # The cell ID that a contig names: its file's prefix and its
                # barcode, so equal barcodes of two samples stay apart.
                identifiers = np.char.add(prefix, barcodes)
                rows = lookup.get_indexer(identifiers)
                matched = rows >= 0
                n_file_contigs += len(chunk)
                if not matched.all():
                    missing = pd.unique(identifiers[~matched])
                    n_unmatched += int(np.count_nonzero(~matched))
                    hashes = np.asarray(
                        pd.util.hash_array(missing.astype(object)), dtype=np.uint64
                    )
                    unmatched_hashes = sorted_unique(
                        np.concatenate([unmatched_hashes, hashes])
                    )
                    for barcode in missing.tolist():
                        if len(examples) == _UNMATCHED_EXAMPLES:
                            break
                        if barcode not in examples:
                            examples.append(str(barcode))
                encoder.add(
                    chunk.loc[matched].drop(columns="cell_id"),
                    rows[matched],
                    position,
                )
                peak = _PEAK_PER_ENCODED_BYTE * encoder.nbytes + lookup_bytes
                if peak > self.memoryBytes:
                    raise MemoryError(
                        f"Importing the contigs of {encoder.n_contigs} matched rows "
                        f"holds about {peak} bytes at its peak, more than the "
                        f"memory budget of {self.memoryBytes} bytes. Raise "
                        "mem_budget."
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
                f"{os.fspath(sources[position].path)!r}; each contig of a file "
                "needs its own ID"
            )
        table = encoder.finish()
        locus = table.text["locus"]
        index = index_chains(
            table, n_cells, locus=locus, productive_only=productive_only
        )
        quality = receptor_quality(index, locus)
        n_contigs = np.bincount(table.cells, minlength=n_cells).astype(np.int32)
        contig_arrays = {"contig_chain": index.eligible, **_contig_arrays(table)}
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
            attributes={
                "fields": list(table.fields),
                "n_file_contigs": int(n_file_contigs),
                "n_unmatched_contigs": int(n_unmatched),
                "n_unmatched_barcodes": len(unmatched_hashes),
                "unmatched_barcode_examples": examples,
            },
        )
        if n_unmatched:
            logger.warning(
                f"{n_unmatched} of {n_file_contigs} contigs, of "
                f"{len(unmatched_hashes)} barcodes, match no selected cell, for "
                "example " + ", ".join(repr(value) for value in examples) + ". "
                "Check barcode_prefix and cell_selection."
            )
        counts = pd.Series(quality.receptor_type).value_counts()
        logger.info(
            f"Imported {table.n_contigs} contigs of {n_cells} cells: "
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
            + _TABLE_BYTES_PER_FIELD * n_cells * (slot_columns + len(_CELL_ARRAYS))
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

    def _within_samples(
        self,
        cell_selection: ArtifactRef,
        columns: tuple[str, ...],
        rows: np.ndarray,
        *,
        reason: str,
    ) -> tuple[np.ndarray, list[dict[str, str]], str | None]:
        """Group the selected cells at ``rows`` by their labels in ``columns``.

        Returns each of those cells' sample code, each sample's labels, and the
        fingerprint of the cells' samples, as ``sample_codes`` does; without
        columns, one sample without labels and no fingerprint.
        """
        n_rows = int(np.count_nonzero(rows))
        if not columns:
            return sample_codes(columns, [], n_rows, reason=reason)
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)[rows]
        labels = [
            apply_missing_mask(
                np.asarray(read_metadata_rows(self.cells, column, cell_index)),
                read_metadata_missing_rows(self.cells, column, cell_index),
                labels=True,
            )
            for column in columns
        ]
        return sample_codes(columns, labels, n_rows, reason=reason)

    def _artifact_cell_ids(self, artifact: ArtifactRef) -> np.ndarray:
        """The IDs of a cell-aligned artifact's cells, in the order of its arrays."""
        selection = inspect_artifact(self.zw, artifact).input_ref("cell_selection")
        cell_index = read_stored_selection_indices(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        return np.asarray(read_metadata_rows_chunkwise(self.cells, "ids", cell_index))

    def _check_table_budget(self, n_cells: int, per_cell: int, name: str) -> None:
        """Refuse to load tables of ``n_cells`` cells beyond the memory budget."""
        resident = per_cell * int(n_cells)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Loading the {name} of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
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
        default for these metrics, since it decides the clusters.

        Clonotypes are numbered by size, so ``clonotype_1`` is the largest;
        ties follow the clonotypes' sorted chains, so the numbers do not
        depend on the order of the cells. The artifact's canonical ``values``
        hold each cell's clonotype label, empty for a cell without one, which
        groupings treat as missing; ``clone_size`` holds the cells of each
        cell's clonotype.

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
            ValueError: If an option is unknown, a distance metric has no
                cutoff, the receptors store no junction of the compared kind,
                or a ``within`` column has no label for a cell with a
                receptor.
            PermissionError: If the store is read-only and the clonotypes are
                not saved yet.
            MemoryError: If the definition, about 1,000 bytes per cell at its
                peak, does not fit the memory budget.
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
            if not isinstance(flag, bool):
                raise TypeError(f"{name} must be a boolean")
        columns = _within_columns(within, reserved=_CLONOTYPE_COLUMNS)
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}")
        if metric == "identity":
            if cutoff not in (None, 0) or isinstance(cutoff, bool):
                raise ValueError("metric='identity' takes no cutoff")
            limit = 0
        else:
            if cutoff is None:
                raise ValueError(
                    f"metric={metric!r} needs a cutoff, such as 1 or 2 "
                    "substitutions or edits, or 15 percent"
                )
            if isinstance(cutoff, bool) or not isinstance(cutoff, int | np.integer):
                raise TypeError("cutoff must be a non-negative integer")
            if cutoff < 0 or (metric == "normalized_hamming" and cutoff > 100):
                raise ValueError(
                    "cutoff must be a non-negative integer, at most 100 percent"
                )
            limit = int(cutoff)
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
        receptor_type = _read(group, "values").astype(str)
        n_cells = len(receptor_type)
        with_receptor = receptor_type != NO_RECEPTOR
        if within_receptor_type:
            _types, type_codes = _sorted_codes(receptor_type)
        else:
            type_codes = np.zeros(n_cells, dtype=np.int64)
        within_codes = np.zeros(n_cells, dtype=np.int64)
        codes, within_labels, fingerprint = self._within_samples(
            cell_selection,
            columns,
            with_receptor,
            reason="Every cell with a receptor needs a label in the within columns.",
        )
        within_codes[with_receptor] = codes
        n_within = max(1, int(within_codes.max()) + 1)
        groups = np.where(
            with_receptor, type_codes.astype(np.int64) * n_within + within_codes, -1
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
        resident = _CLONOTYPE_BYTES_PER_CELL * n_cells
        if metric != "identity":
            # A cluster search holds one pass of keys and one chunk of pairs.
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
            assignment = assign_clonotype_clusters(
                compared,
                _stored_text(group, field_name).categories,
                groups=groups,
                metric=metric,
                cutoff=limit,
                v_genes=v_genes,
                j_genes=j_genes,
                receptor_arms=receptor_arms,
                dual_ir=dual_ir,
                max_bytes=self.memoryBytes - resident,
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
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": values,
                "codes": assignment.codes.astype(np.int32),
                "clone_size": sizes,
                "clonotype_ids": labels,
                "clonotype_sizes": assignment.sizes,
                "representatives": representatives,
                "n_configurations": n_configurations,
                "clonotype_within": clonotype_within,
            },
            attributes={
                "n_clonotypes": assignment.n_clonotypes,
                "within_labels": within_labels,
            },
        )
        logger.info(
            f"Defined {assignment.n_clonotypes} clonotypes of {int(has.sum())} of "
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
            MemoryError: If the tables, about 1,000 bytes per cell, do not fit
                the memory budget.
        """
        group = self._clonotype_group(clonotypes)
        self._check_table_budget(
            as_zarr_array(group["codes"], name="codes").shape[0],
            _CLONOTYPE_TABLE_BYTES_PER_CELL,
            "clonotypes",
        )
        status = inspect_artifact(self.zw, clonotypes)
        parameters = dict(status.parameters or {})
        receptors = status.input_ref("receptors")
        receptor_group = artifact_group(self.zw, receptors)
        fields = _stored_fields(receptor_group)
        cell_ids = self._artifact_cell_ids(clonotypes)
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
        within = tuple(parameters.get("within") or ())
        within_labels = cast(
            list[dict[str, str]], group.attrs.get("within_labels") or [{}]
        )
        within_codes = _read(group, "clonotype_within").astype(np.int64)
        cells, table = clonotype_frames(
            cell_ids,
            codes=_read(group, "codes"),
            clone_size=_read(group, "clone_size"),
            clonotype_ids=_read(group, "clonotype_ids").astype(str),
            clonotype_sizes=_read(group, "clonotype_sizes"),
            n_configurations=_read(group, "n_configurations"),
            receptor_type=_read(receptor_group, "values").astype(str)[representatives],
            within={
                column: _labels_of(within_labels, column, within_codes)
                for column in within
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

    @validation_scoped
    def _repertoire_clonal_expansion(
        self,
        clonotypes: ArtifactRef,
        *,
        within: str | Sequence[str] | None = None,
        within_receptor_type: bool = True,
        breakpoints: Sequence[int] | None = None,
        frequency_breakpoints: Sequence[float] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Place each cell in an expansion category by its clone size in its group.

        A clonotype that holds more than one cell has expanded. Its size is
        counted within groups of cells: the combinations of the labels of the
        ``within`` cell metadata columns, such as the sample, and, with
        ``within_receptor_type``, of the receptor type, so T cells and B cells
        are counted apart. Without ``within``, a clonotype's size counts its
        cells in every sample, as scirpy's ``clonal_expansion`` does without
        ``expanded_in``.

        ``breakpoints`` sorts the cells by their clonotype's cells in the
        group, as scirpy's ``breakpoints`` do: the default ``(1, 2, 5, 20,
        100)`` makes the categories ``"1 cell"``, ``"2 cells"``,
        ``"3 to 5 cells"``, ``"6 to 20 cells"``, ``"21 to 100 cells"`` and
        ``"more than 100 cells"``.
        ``frequency_breakpoints`` sorts them instead by their clonotype's
        frequency, its share of the group's cells with a clonotype, in bins
        ``(lower, upper]`` that end at 1, as scRepertoire's
        ``clonalHomeostasis`` does: ``(0.0001, 0.001, 0.01, 0.1, 1)`` makes its
        Rare, Small, Medium, Large and Hyperexpanded categories, named
        ``"0 to 0.0001"`` to ``"0.1 to 1"``. A frequency depends on the
        group's number of cells, so compare groups of similar sizes, or bounds
        in cells.

        The artifact's canonical ``values`` hold each cell's category, empty
        for a cell without a clonotype, so
        ``ds.select_cells(expansion, include=["more than 100 cells"])`` keeps
        the cells of the largest clonotypes. ``group_clone_size`` and
        ``group_clone_frequency`` are its other per-cell arrays.

        Args:
            clonotypes: A ``clonotypes`` artifact.
            within: Cell metadata columns whose labels define the groups, such
                as the sample; every cell with a clonotype needs a label.
            within_receptor_type: Count each receptor type apart.
            breakpoints: Strictly increasing bounds in cells, or None.
            frequency_breakpoints: Strictly increasing frequency bounds, the
                last 1, or None. Pass at most one of the two bounds; without
                either, ``breakpoints`` are ``(1, 2, 5, 20, 100)``.
            invalidate_cache: Measure again instead of reusing a saved result.

        Returns:
            A complete datastore-scoped ``clonal_expansion`` artifact aligned
            to the clonotypes' cells. Read it with
            :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.load_clonal_expansion`
            or plot it with ``ds.plots.clonal_expansion``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If both bounds are given, the bounds are invalid,
                ``within`` names ``receptor_type`` while
                ``within_receptor_type`` adds it, or a ``within`` column has no
                label for a cell with a clonotype.
            PermissionError: If the store is read-only and the result is not
                saved yet.
            MemoryError: If the measurement, about 300 bytes per cell, does
                not fit the memory budget.
        """
        group = self._clonotype_group(clonotypes)
        for name, flag in (
            ("within_receptor_type", within_receptor_type),
            ("invalidate_cache", invalidate_cache),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"{name} must be a boolean")
        columns = _within_columns(within, reserved=_EXPANSION_COLUMNS)
        if within_receptor_type and _RECEPTOR_TYPE_COLUMN in columns:
            raise ValueError(
                f"within names {_RECEPTOR_TYPE_COLUMN!r}, which within_receptor_type "
                "already adds; pass within_receptor_type=False to use the column"
            )
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
        status = inspect_artifact(self.zw, clonotypes)
        cell_selection = status.input_ref("cell_selection")
        codes = _read(group, "codes").astype(np.int64)
        n_cells = len(codes)
        has = codes >= 0
        within_codes, within_labels, fingerprint = self._within_samples(
            cell_selection,
            columns,
            has,
            reason="Every cell with a clonotype needs a label in the within columns.",
        )
        arguments = ClonalExpansionArguments(
            clonotypes=clonotypes,
            within_fingerprint=fingerprint,
            within=columns,
            within_receptor_type=within_receptor_type,
            breakpoints=counts,
            frequency_breakpoints=frequencies,
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
                "group_clone_size": ((n_cells,), "i"),
                "group_clone_frequency": ((n_cells,), "f"),
                "groups": ((n_cells,), "i"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("labels", "group_keys"),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        resident = _EXPANSION_BYTES_PER_CELL * n_cells
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Measuring the clonal expansion of {n_cells} cells holds about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        if within_receptor_type:
            receptors = status.input_ref("receptors")
            receptor_type = _read(artifact_group(self.zw, receptors), "values")
            types, type_codes = _sorted_codes(receptor_type.astype(str)[has])
        else:
            types = np.asarray([], dtype=str)
            type_codes = np.zeros(len(within_codes), dtype=np.int64)
        combinations, group_codes = unique_rows(
            np.column_stack([within_codes, type_codes]).astype(np.int64)
        )
        group_keys = [
            {
                **within_labels[sample],
                **(
                    {_RECEPTOR_TYPE_COLUMN: str(types[kind])}
                    if within_receptor_type
                    else {}
                ),
            }
            for sample, kind in combinations.tolist()
        ]
        # Tables and plots name each group by its labels.
        group_names(group_keys)
        groups = np.full(n_cells, -1, dtype=np.int64)
        groups[has] = group_codes
        sizes, clone_frequencies = group_clone_sizes(codes, groups)
        categories, labels = expansion_categories(
            sizes,
            clone_frequencies,
            breakpoints=counts,
            frequency_breakpoints=frequencies,
        )
        names = np.asarray(labels, dtype=object)
        values = np.full(n_cells, "", dtype=object)
        values[has] = names[categories[has]]
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": values,
                "group_clone_size": sizes,
                "group_clone_frequency": clone_frequencies,
                "groups": groups.astype(np.int32),
            },
            attributes={"labels": list(labels), "group_keys": group_keys},
        )
        logger.info(
            f"Measured the clonal expansion of {int(has.sum())} cells with a "
            f"clonotype in {len(group_keys)} groups"
        )
        return planned.ref

    @validation_scoped
    def _repertoire_load_clonal_expansion(
        self,
        expansion: ArtifactRef,
        *,
        splits: Sequence[int] = DEFAULT_RANK_SPLITS,
    ) -> ClonalExpansion:
        """Read saved clonal expansion as tables of cells, categories and ranks.

        Besides each cell's category and each group's cells and clonotypes per
        category, the loader ranks each group's clonotypes by their cells and
        counts the cells that the clonotypes in each window of ranks hold, as
        scRepertoire's ``clonalProportion`` does: by default the 10 largest,
        11 to 100, 101 to 1000, 1001 to 10000, 10001 to 30000, 30001 to
        100000, and the rest.

        Args:
            expansion: A ``clonal_expansion`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.clonal_expansion`.
            splits: The last rank of each window but the last, strictly
                increasing.

        Returns:
            The clonal expansion, with one row per cell, one per group and
            category, and one per group and rank window.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``expansion`` is not a ``clonal_expansion``
                artifact, or ``splits`` are invalid.
            MemoryError: If the tables, about 500 bytes per cell, do not fit
                the memory budget.
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
        self._check_table_budget(
            as_zarr_array(group["groups"], name="groups").shape[0],
            _EXPANSION_TABLE_BYTES_PER_CELL,
            "clonal expansion",
        )
        status = inspect_artifact(self.zw, expansion)
        parameters = dict(status.parameters or {})
        clonotypes = status.input_ref("clonotypes")
        within = tuple(str(column) for column in parameters["within"])
        within_receptor_type = bool(parameters["within_receptor_type"])
        labels = tuple(str(name) for name in cast(list[str], group.attrs["labels"]))
        group_keys = cast(list[dict[str, str]], group.attrs["group_keys"])
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
        clonotype_group = self._clonotype_group(clonotypes)
        cells, table, ranks = expansion_tables(
            self._artifact_cell_ids(expansion),
            categories=categories,
            labels=labels,
            clonotypes=_read(clonotype_group, "codes"),
            clonotype_ids=_read(clonotype_group, "clonotype_ids").astype(str),
            groups=_read(group, "groups"),
            group_keys=group_keys,
            key_columns=(
                *within,
                *((_RECEPTOR_TYPE_COLUMN,) if within_receptor_type else ()),
            ),
            sizes=sizes,
            frequencies=clone_frequencies,
            splits=bounds,
        )
        return ClonalExpansion(
            cells=cells,
            categories=table,
            rank_windows=ranks,
            clonotypes=clonotypes,
            labels=labels,
            breakpoints=None if counts is None else tuple(int(v) for v in counts),
            frequency_breakpoints=(
                None if frequencies is None else tuple(float(v) for v in frequencies)
            ),
            splits=bounds,
            within=within,
            within_receptor_type=within_receptor_type,
        )

    def _clone_groups(
        self,
        clonotypes: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        within_receptor_type: bool,
        cell_selection: ArtifactRef | None,
        reserved: frozenset[str] = frozenset(),
    ) -> tuple[pd.DataFrame, list[CloneSizes]]:
        """Group the cells with a clonotype and count each group's clone sizes.

        Returns:
            One row per group, in the natural order of the labels, with its
            name in ``group`` and its labels in the key columns, and each
            group's clone sizes.
        """
        keys, pair_groups, _codes, pair_cells = self._clone_counts(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            reserved=reserved,
        )
        bounds = np.searchsorted(pair_groups, np.arange(len(keys) + 1))
        clone_sizes = [
            CloneSizes.from_sizes(pair_cells[bounds[index] : bounds[index + 1]])
            for index in range(len(keys))
        ]
        return keys, clone_sizes

    def _clone_counts(
        self,
        clonotypes: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        within_receptor_type: bool,
        cell_selection: ArtifactRef | None,
        reserved: frozenset[str] = frozenset(),
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Group the cells with a clonotype and count each clonotype's cells.

        The groups are those of :meth:`_cell_groups`, among the cells with a
        clonotype.

        Returns:
            One row per group, in the natural order of the labels, with its
            name in ``group`` and its labels in the key columns, and for each
            distinct pair of group and clonotype, sorted by group and
            clonotype code, its group, its clonotype code and its cells.
        """
        clonotype_group = self._clonotype_group(clonotypes)
        codes = _read(clonotype_group, "codes").astype(np.int64)
        status = inspect_artifact(self.zw, clonotypes)
        key_columns, group_keys, cell_groups = self._cell_groups(
            status.input_ref("cell_selection"),
            status.input_ref("receptors"),
            codes >= 0,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            reserved=reserved,
        )
        grouped = cell_groups >= 0
        pair_groups, pair_codes, pair_cells = pair_counts(
            cell_groups[grouped], codes[grouped]
        )
        return group_frame(key_columns, group_keys), pair_groups, pair_codes, pair_cells

    def _cell_groups(
        self,
        artifact_cells: ArtifactRef,
        receptors: ArtifactRef,
        rows: np.ndarray,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        within_receptor_type: bool,
        cell_selection: ArtifactRef | None,
        reserved: frozenset[str] = frozenset(),
    ) -> tuple[tuple[str, ...], list[dict[str, str]], np.ndarray]:
        """Group the cells of a cell-aligned artifact by their labels.

        The groups are the combinations of the labels of the ``group_by``
        columns, of the ``groups`` artifact and, with
        ``within_receptor_type``, of the receptor type of ``receptors``. A
        cell outside ``rows`` or ``cell_selection``, or without a label, is
        in no group.

        Args:
            artifact_cells: The cell selection of the artifact's cells.
            receptors: The ``immune_receptors`` artifact of those cells.
            rows: The cells that may join a group.
            group_by: Cell metadata columns.
            groups: A cell-label artifact.
            within_receptor_type: Make the receptor type part of each group.
            cell_selection: Only these cells, or None.

        Returns:
            The key columns, each group's labels in them, in the natural order
            of the labels, and each cell's group, -1 for none.
        """
        if not isinstance(within_receptor_type, bool):
            raise TypeError("within_receptor_type must be a boolean")
        columns = _within_columns(group_by, "group_by", reserved)
        if groups is not None and not isinstance(groups, ArtifactRef):
            raise TypeError("groups must be an ArtifactRef or None")
        key_columns = (
            *columns,
            *((groups.kind,) if groups is not None else ()),
            *((_RECEPTOR_TYPE_COLUMN,) if within_receptor_type else ()),
        )
        if len(set(key_columns)) != len(key_columns):
            raise ValueError(
                f"The groups' columns {list(key_columns)} name one column twice; "
                "rename the column or pass within_receptor_type=False"
            )
        n_cells = len(rows)
        n_columns = len(key_columns)
        resident = n_cells * (
            _GROUPING_BYTES_PER_CELL + _GROUPING_BYTES_PER_COLUMN * n_columns
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Grouping {n_cells} cells by {n_columns} columns holds about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        has = np.asarray(rows, dtype=bool).copy()
        cell_index = read_stored_selection_indices(
            self.zw,
            artifact_cells,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if cell_selection is not None:
            if not isinstance(cell_selection, ArtifactRef):
                raise TypeError("cell_selection must be an ArtifactRef or None")
            chosen = read_stored_selection_indices(
                self.zw,
                cell_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            has &= np.isin(cell_index, chosen)
        cell_index = cell_index[has]
        # Each column's labels are coded once, in the natural order of their
        # texts, and only the distinct labels are checked and named.
        coded: list[tuple[np.ndarray, list[str]]] = [
            category_codes(
                read_metadata_rows(self.cells, column, cell_index),
                read_metadata_missing_rows(self.cells, column, cell_index),
            )
            for column in columns
        ]
        if groups is not None:
            resolved = resolve_grouping(self.zw, self.cells, groups)
            codes, names = category_codes(resolved.labels, resolved.missing_mask)
            at = pd.Index(np.asarray(resolved.cell_idx, dtype=np.int64)).get_indexer(
                cell_index
            )
            aligned = np.full(len(cell_index), -1, dtype=np.int64)
            aligned[at >= 0] = codes[at[at >= 0]]
            coded.append((aligned, names))
        if within_receptor_type:
            receptor_type = _read(artifact_group(self.zw, receptors), "values")
            coded.append(category_codes(receptor_type[has]))
        cell_groups = np.full(n_cells, -1, dtype=np.int64)
        if not coded:
            cell_groups[has] = 0
            return key_columns, [{}], cell_groups
        stacked = np.column_stack([codes for codes, _names in coded])
        keep = (stacked >= 0).all(axis=1)
        combinations, group_codes = unique_rows(stacked[keep])
        group_keys = [
            {
                column: names[code]
                for column, (_codes, names), code in zip(
                    key_columns, coded, row, strict=True
                )
            }
            for row in combinations.tolist()
        ]
        selected = np.flatnonzero(has)
        cell_groups[selected[keep]] = group_codes
        # Groups are named by their labels joined by commas, and need one
        # name each.
        group_names(group_keys)
        return key_columns, group_keys, cell_groups

    @validation_scoped
    def _repertoire_diversity(
        self,
        clonotypes: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        within_receptor_type: bool = True,
        cell_selection: ArtifactRef | None = None,
        size: int | None = None,
        coverage: float | None = None,
    ) -> ClonalDiversity:
        """Measure the diversity of the clonotypes of groups of cells.

        The groups are the combinations of the labels of the ``group_by``
        cell metadata columns, such as the sample, of the ``groups`` label
        artifact, such as a clustering, and, with ``within_receptor_type``, of
        the receptor type, so T cells and B cells are measured apart. Cells
        without a clonotype or a label are left out. A clonotype's size in a
        group is its cells there.

        The indices are those of scRepertoire's ``clonalDiversity``,
        immunarch's ``repDiversity`` and vegan's ``estimateR``: Shannon
        entropy, Pielou's evenness, the Simpson indices, the Gini coefficient,
        D50, Chao1 in its classic and bias-corrected forms, and ACE. Each
        depends on how many cells a group holds, so the Hill numbers of
        orders 0, 1 and 2, the effective numbers of clonotypes, are also
        rarefied or extrapolated to a common ``size`` and to a common sample
        ``coverage``, as iNEXT's ``estimateD`` does (Chao et al. 2014): by
        default the smallest of the groups' doubled sizes, and the lowest
        coverage of the groups at their doubled sizes. Compare groups by
        these standardized numbers. Nothing is saved.

        Args:
            clonotypes: A ``clonotypes`` artifact.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            within_receptor_type: Measure each receptor type apart.
            cell_selection: Only these cells, such as the T cells that
                ``ds.select_cells(receptors, include=["TCR"])`` keeps, or None
                for every cell of the clonotypes.
            size: The common size in cells, or None for the default.
            coverage: The common coverage, between 0 and 1, or None for the
                default.

        Returns:
            The groups' indices and Hill numbers, observed, asymptotic and
            standardized.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If no cell has a clonotype and a group, a column of the
                groups is named twice, ``size`` is not positive, or a group
                cannot reach ``coverage``.
            MemoryError: If grouping the cells does not fit the memory budget.
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
        keys, clone_sizes = self._clone_groups(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            reserved=_DIVERSITY_COLUMNS,
        )
        if not clone_sizes or all(clones.n == 0 for clones in clone_sizes):
            raise ValueError("No cell has both a clonotype and a group")
        table, hill, common, level = diversity_tables(
            clone_sizes, keys, size=size, coverage=coverage
        )
        return ClonalDiversity(
            indices=table,
            hill=hill,
            clonotypes=clonotypes,
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            size=common,
            coverage=level,
        )

    @validation_scoped
    def _repertoire_rarefaction(
        self,
        clonotypes: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        within_receptor_type: bool = True,
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
        :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.diversity`.
        Nothing is saved.

        By default each group's curve has iNEXT's ``knots`` sizes: about half
        evenly from 1 to its size and half from there to ``endpoint``, by
        default twice its size. ``sizes`` gives every group the same sizes
        instead.

        Args:
            clonotypes: A ``clonotypes`` artifact.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            within_receptor_type: Measure each receptor type apart.
            cell_selection: Only these cells, or None for every cell of the
                clonotypes.
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
            ValueError: If an order, size, endpoint or knots is invalid, or no
                cell has a clonotype and a group.
            MemoryError: If grouping the cells does not fit the memory budget.
        """
        if isinstance(orders, str | bytes) or not isinstance(orders, Sequence):
            raise TypeError("orders must be a sequence of 0, 1 and 2")
        chosen = tuple(orders)
        if (
            not chosen
            or any(isinstance(order, bool) or order not in ORDERS for order in chosen)
            or len(set(chosen)) != len(chosen)
        ):
            raise ValueError("orders must hold distinct orders from 0, 1 and 2")
        if sizes is not None:
            if isinstance(sizes, str | bytes) or not isinstance(sizes, Sequence):
                raise TypeError("sizes must be a sequence of positive integers")
            if not sizes or any(
                isinstance(value, bool)
                or not isinstance(value, int | np.integer)
                or value < 1
                for value in sizes
            ):
                raise ValueError("sizes must hold positive integers")
        # The defaults' arguments are checked once, before any group is read.
        default_sizes(2, endpoint, knots)
        keys, clone_sizes = self._clone_groups(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            reserved=_RAREFACTION_COLUMNS,
        )
        if not clone_sizes or all(clones.n == 0 for clones in clone_sizes):
            raise ValueError("No cell has both a clonotype and a group")
        return Rarefaction(
            curves=rarefaction_curves(
                clone_sizes,
                keys,
                chosen,
                sizes=sizes,
                endpoint=endpoint,
                knots=knots,
            ),
            clonotypes=clonotypes,
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            orders=chosen,
        )

    def _group_table(
        self, keys: pd.DataFrame, pair_groups: np.ndarray, pair_cells: np.ndarray
    ) -> pd.DataFrame:
        """One row per group: its name, labels, cells and clonotypes."""
        n_groups = len(keys)
        return keys.assign(
            cells=np.bincount(
                pair_groups, weights=pair_cells, minlength=n_groups
            ).astype(np.int64),
            clonotypes=np.bincount(pair_groups, minlength=n_groups),
        )

    @validation_scoped
    def _repertoire_overlap(
        self,
        clonotypes: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        within_receptor_type: bool = True,
        cell_selection: ArtifactRef | None = None,
    ) -> ClonotypeOverlap:
        """Compare groups of cells by the clonotypes they share.

        The groups are the combinations of the labels of the ``group_by``
        cell metadata columns, such as the sample or the tissue, and of the
        ``groups`` label artifact, such as a clustering, among the cells of
        ``cell_selection``. With ``within_receptor_type`` the receptor type
        is part of each group and only groups of one receptor type are
        compared, since a T cell and a B cell never share a clonotype. Cells
        without a clonotype or a label are left out. Nothing is saved.

        Each pair of groups gets presence indices, which count clonotypes,
        and abundance indices, which weigh them by their cells, as
        scRepertoire's ``clonalOverlap`` and immunarch's ``repOverlap`` report
        them: ``shared`` (the clonotypes of both), ``jaccard``, ``sorensen``,
        ``overlap`` (shared over the smaller group's clonotypes), ``cosine``
        (Ochiai's, of presence), ``morisita_horn`` and ``abundance_cosine``,
        and the share of each group's cells whose clonotype the other holds.
        Presence indices grow with the cells of both groups, since a larger
        sample finds more of the rare clonotypes; the Morisita-Horn index
        depends little on them and is near 1 when the same expanded clones
        dominate both groups.

        Args:
            clonotypes: A ``clonotypes`` artifact.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            within_receptor_type: Make the receptor type part of each group
                and compare only groups of one receptor type.
            cell_selection: Only these cells, or None for every cell of the
                clonotypes.

        Returns:
            The groups' cells and clonotypes and the indices of every pair.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If fewer than two groups hold a cell with a clonotype,
                or a column of the groups is named twice.
            MemoryError: If grouping the cells, or the pairs of groups, do not
                fit the memory budget.
        """
        keys, pair_groups, pair_codes, pair_cells = self._clone_counts(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            reserved=_OVERLAP_COLUMNS,
        )
        if len(pair_cells) == 0 or len(keys) < 2:
            raise ValueError(
                "Overlap needs at least two groups of cells with a clonotype"
            )
        strata = (
            keys[_RECEPTOR_TYPE_COLUMN].to_numpy(dtype=object)
            if within_receptor_type
            else None
        )
        n_pairs, largest = overlap_extent(len(keys), strata)
        resident = (
            _OVERLAP_BYTES_PER_PAIR * n_pairs
            + _OVERLAP_BYTES_PER_PRODUCT * largest**2
            + _OVERLAP_BYTES_PER_COUNT * len(pair_cells)
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Comparing {len(keys)} groups gives {n_pairs} pairs, which hold "
                f"about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Compare fewer groups, with fewer "
                "group_by columns or a cell_selection, or raise mem_budget."
            )
        sizes = self._group_table(keys, pair_groups, pair_cells)
        pairs = overlap_pairs(
            pair_groups, pair_codes, pair_cells, len(keys), strata=strata
        )
        names = keys["group"].to_numpy(dtype=object)
        pairs.insert(0, "group_1", names[pairs.pop("first").to_numpy()])
        pairs.insert(1, "group_2", names[pairs.pop("second").to_numpy()])
        return ClonotypeOverlap(
            group_sizes=sizes,
            pairs=pairs,
            clonotypes=clonotypes,
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
        )

    @validation_scoped
    def _repertoire_track(
        self,
        clonotypes: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None = None,
        groups: ArtifactRef | None = None,
        within_receptor_type: bool = True,
        cell_selection: ArtifactRef | None = None,
        top: int = 10,
        labels: Sequence[str] | None = None,
    ) -> ClonotypeTracking:
        """Follow chosen clonotypes across groups of cells.

        By default the clonotypes are the ``top`` with the most cells in each
        group, together, as scRepertoire's ``clonalCompare`` chooses them;
        ``labels`` names them instead, such as ``["clonotype_1"]``. Each
        chosen clonotype gets its cells and share of cells in every group of
        its receptor type, zeros included, so a clonotype that expands
        between time points or spreads between tissues shows it. The groups
        are those of
        :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.overlap`.
        Nothing is saved.

        Args:
            clonotypes: A ``clonotypes`` artifact.
            group_by: Cell metadata columns whose labels define the groups.
            groups: A cell-label artifact whose labels define the groups.
            within_receptor_type: Make the receptor type part of each group.
            cell_selection: Only these cells, or None for every cell of the
                clonotypes.
            top: The clonotypes with the most cells to take from each group.
            labels: Clonotype labels to follow instead, or None.

        Returns:
            One row per chosen clonotype and group, and the junction that
            names each chosen clonotype.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``top`` is not positive, a label names no
                clonotype, or no cell has a clonotype and a group.
            MemoryError: If grouping the cells, or the table of the chosen
                clonotypes' groups, does not fit the memory budget.
        """
        if isinstance(top, bool) or not isinstance(top, int | np.integer):
            raise TypeError("top must be a positive integer")
        if top < 1:
            raise ValueError("top must be a positive integer")
        if labels is not None and (
            isinstance(labels, str | bytes)
            or not isinstance(labels, Sequence)
            or not all(isinstance(label, str) for label in labels)
        ):
            raise TypeError("labels must be a sequence of clonotype labels or None")
        keys, pair_groups, pair_codes, pair_cells = self._clone_counts(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            reserved=_TRACK_COLUMNS,
        )
        if len(pair_cells) == 0:
            raise ValueError("No cell has both a clonotype and a group")
        identifiers = _read(self._clonotype_group(clonotypes), "clonotype_ids").astype(
            str
        )
        if labels is None:
            chosen = top_clonotypes(
                pair_groups, pair_codes, pair_cells, len(keys), int(top)
            )
        else:
            positions = pd.Index(identifiers).get_indexer(list(labels))
            unknown = [label for label, at in zip(labels, positions) if at < 0]
            if unknown:
                raise ValueError(f"No clonotype is labelled {unknown[:5]}")
            chosen = sorted_unique(positions.astype(np.int64))
        sizes = self._group_table(keys, pair_groups, pair_cells)
        stratum_names, strata = np.unique(
            keys[_RECEPTOR_TYPE_COLUMN].to_numpy(dtype=object)
            if within_receptor_type
            else np.zeros(len(keys), dtype=object),
            return_inverse=True,
        )
        receptor_types, junctions = self._representative_chains(clonotypes, chosen)
        # A labelled clonotype without cells in the groups is followed through
        # the groups of its own receptor type.
        if within_receptor_type:
            lookup = {name: code for code, name in enumerate(stratum_names.tolist())}
            fallback = np.asarray(
                [lookup.get(name, -1) for name in receptor_types.tolist()],
                dtype=np.int64,
            )
        else:
            fallback = np.zeros(len(chosen), dtype=np.int64)
        reach = tracking_reach(pair_groups, pair_codes, chosen, strata, fallback)
        n_rows = int((reach * np.bincount(strata, minlength=reach.shape[1])).sum())
        resident = _TRACK_BYTES_PER_ROW * n_rows + reach.shape[0] * len(keys)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Following {len(chosen)} clonotypes through {len(keys)} groups "
                f"gives {n_rows} rows, which hold about {resident} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Follow "
                "fewer clonotypes, with a smaller top or labels, or fewer "
                "groups, or raise mem_budget."
            )
        followed, rows, counts = tracking_cells(
            pair_groups, pair_codes, pair_cells, chosen, strata, reach
        )
        if len(rows):
            table = keys.take(rows).reset_index(drop=True)
            names = np.asarray(identifiers[chosen].tolist(), dtype=object)
            table.insert(0, "clonotype", names[followed])
            table["cells"] = counts
            table["fraction"] = counts / sizes["cells"].to_numpy()[rows]
        else:
            table = pd.DataFrame(
                columns=["clonotype", *keys.columns, "cells", "fraction"]
            )
        return ClonotypeTracking(
            cells=table,
            clonotypes=clonotypes,
            tracked=tuple(identifiers[chosen].tolist()),
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            junctions=junctions,
        )

    def _representative_chains(
        self, clonotypes: ArtifactRef, codes: np.ndarray
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Read the receptor type and naming junction of some clonotypes.

        Both are those of each clonotype's representative cell, and the
        junction is the residues of its ``VDJ_1`` chain, or of its ``VJ_1``
        chain without a VDJ chain. Only the representatives' rows are read.

        Returns:
            The receptor type of each clonotype, and its junction.
        """
        group = self._clonotype_group(clonotypes)
        cells = np.asarray(
            as_zarr_array(
                group["representatives"], name="representatives"
            ).get_coordinate_selection(np.asarray(codes, dtype=np.int64)),
            dtype=np.int64,
        )
        receptors = inspect_artifact(self.zw, clonotypes).input_ref("receptors")
        receptor_group = artifact_group(self.zw, receptors)
        values = as_zarr_array(receptor_group["values"], name="values")
        receptor_types = np.asarray(values.get_coordinate_selection(cells)).astype(str)
        slots = np.asarray(
            as_zarr_array(
                receptor_group["chain_slots"], name="chain_slots"
            ).get_orthogonal_selection(
                (
                    cells,
                    np.asarray(
                        [SLOTS.index("VDJ_1"), SLOTS.index("VJ_1")], dtype=np.int64
                    ),
                )
            ),
            dtype=np.int64,
        ).reshape(len(cells), 2)
        contigs = np.where(slots[:, 0] >= 0, slots[:, 0], slots[:, 1])
        junctions = _stored_text_at(receptor_group, "junction_aa", contigs)
        return receptor_types, tuple(junctions.tolist())

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
        reserved: frozenset[str] = frozenset(),
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], tuple[str, ...], zarr.Group]:
        """Each counted chain of the chosen slots, with its fields' stored codes.

        A ``clonotypes`` source counts only cells with a clonotype, and with
        ``count="clonotypes"`` each clonotype once per group and value.

        Returns:
            One row per group, with its name in ``group`` and its labels in
            the key columns; for each chain, ``group``, ``cell``, the cell's
            row, the text codes of ``locus`` and of each field, and
            ``clonotype``, the clonotype code, with a clonotypes source; the
            slots; and the ``immune_receptors`` artifact's group.
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
        if source.kind == ClonotypeArguments.artifact_kind:
            codes = _read(self._clonotype_group(source), "codes").astype(np.int64)
            receptors = inspect_artifact(self.zw, source).input_ref("receptors")
        elif source.kind == ImmuneReceptorImportArguments.artifact_kind:
            if count == "clonotypes":
                raise ValueError("count='clonotypes' needs a clonotypes artifact")
            codes = None
            receptors = source
        else:
            raise ValueError(
                "source must be an immune_receptors or clonotypes artifact, not "
                f"{source.kind!r}"
            )
        group, stored = self._receptor_group(receptors)
        for name in fields:
            if name not in stored:
                raise ValueError(f"The receptors store no {name}")
        slot_contigs = _read(group, "chain_slots")[
            :, [SLOTS.index(slot) for slot in chosen]
        ]
        rows = (slot_contigs >= 0).any(axis=1)
        if codes is not None:
            rows &= codes >= 0
        key_columns, group_keys, cell_groups = self._cell_groups(
            inspect_artifact(self.zw, receptors).input_ref("cell_selection"),
            receptors,
            rows,
            group_by=group_by,
            groups=groups,
            within_receptor_type=False,
            cell_selection=cell_selection,
            reserved=reserved,
        )
        cells, positions = np.nonzero((cell_groups >= 0)[:, None] & (slot_contigs >= 0))
        contigs = slot_contigs[cells, positions]
        units = {
            "group": cell_groups[cells],
            "cell": cells.astype(np.int64),
            **{name: _slot_codes(group, name, contigs) for name in ("locus", *fields)},
        }
        if codes is not None:
            units["clonotype"] = codes[cells]
        return group_frame(key_columns, group_keys), units, chosen, group

    @staticmethod
    def _distinct_text(
        group: zarr.Group, name: str, codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode the distinct values of a stored text field among some codes.

        Returns:
            Each code's position among the distinct values, -1 for a missing
            value, and those values, in the order of their text.
        """
        known = codes >= 0
        needed = sorted_unique(codes[known])
        values = np.empty(0, dtype=object)
        if len(needed):
            values = take_text(
                _read(group, f"contig_{name}_values"),
                _read(group, f"contig_{name}_offsets"),
                needed,
            )
        local = np.full(len(codes), -1, dtype=np.int64)
        local[known] = np.searchsorted(needed, codes[known])
        return local, values

    @staticmethod
    def _text_lengths(
        group: zarr.Group, name: str, codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measure the distinct values of a stored text field among some codes.

        Returns:
            Each code's length in characters, -1 for a missing value, and
            every length up to the longest, the names of those codes.
        """
        known = codes >= 0
        needed = sorted_unique(codes[known])
        lengths = text_lengths(
            _read(group, f"contig_{name}_values"),
            _read(group, f"contig_{name}_offsets"),
            needed,
        )
        measured = np.full(len(codes), -1, dtype=np.int64)
        measured[known] = lengths[np.searchsorted(needed, codes[known])]
        return measured, np.arange(int(lengths.max(initial=-1)) + 1, dtype=np.int64)

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
        clones from dominating the shares. The groups are the combinations of
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
            One row per group, locus and gene, or gene combination.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an option is unknown, a gene field is not stored,
                or ``count="clonotypes"`` is asked of receptors.
            MemoryError: If grouping the cells does not fit the memory budget.
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
        keys, units, chosen, group = self._slot_units(
            source,
            fields,
            slots=slots,
            count=count,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=_USAGE_COLUMNS,
        )
        # Each distinct call is decoded and named once.
        named = {
            name: gene_codes(*self._distinct_text(group, name, units[name]), level)
            for name in fields
        }
        table, n_cells = count_chains(
            keys,
            units["group"],
            units["cell"],
            self._distinct_text(group, "locus", units["locus"]),
            named,
            count,
            clonotypes=units.get("clonotype"),
        )
        return GeneUsage(
            table=table,
            source=source,
            genes=fields,
            level=level,
            slots=chosen,
            count=count,
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            cell_selection=cell_selection,
            n_cells=n_cells,
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
            One row per group, locus and length.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an option is unknown, the junctions are not
                stored, or ``count="clonotypes"`` is asked of receptors.
            MemoryError: If grouping the cells does not fit the memory budget.
        """
        if sequence not in SEQUENCES:
            raise ValueError(f"sequence must be one of {', '.join(SEQUENCES)}")
        field_name = "junction_aa" if sequence == "aa" else "junction"
        keys, units, chosen, group = self._slot_units(
            source,
            (field_name,),
            slots=slots,
            count=count,
            group_by=group_by,
            groups=groups,
            cell_selection=cell_selection,
            reserved=_SPECTRATYPE_COLUMNS,
        )
        # Rows follow the groups' natural order, then the locus and length.
        table, n_cells = count_chains(
            keys,
            units["group"],
            units["cell"],
            self._distinct_text(group, "locus", units["locus"]),
            {"length": self._text_lengths(group, field_name, units[field_name])},
            count,
            clonotypes=units.get("clonotype"),
            by_count=False,
        )
        table["length"] = table["length"].astype(np.int64)
        return Spectratype(
            table=table,
            source=source,
            sequence=sequence,
            slots=chosen,
            count=count,
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            cell_selection=cell_selection,
            n_cells=n_cells,
        )

    @validation_scoped
    def _repertoire_export_airr(
        self,
        source: ArtifactRef,
        path: str | os.PathLike[str],
        *,
        fields: Sequence[str] | None = None,
        chains_only: bool = False,
    ) -> int:
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
        logical fields are written ``T`` and ``F``. A ``sequence_id`` that
        repeats, such as ``<barcode>_contig_1`` of the TCR and the BCR files
        of one cell, gets ``_2``, ``_3`` and so on from its second
        occurrence, in the order of the contigs, since the schema requires
        unique identifiers. A path ending in ``.gz`` is gzip-compressed at
        level 6.

        The contigs are read and written in blocks of about 32 MiB of
        rendered text, so the export holds one block, the cells' IDs and
        clonotypes, and the distinct values of the text fields at once.

        Args:
            source: An ``immune_receptors`` or ``clonotypes`` artifact.
            path: The file to write.
            fields: The stored fields to write, or None for every stored
                field; ``sequence_id`` is always written.
            chains_only: Write only the contigs that count as chains.

        Returns:
            The number of contigs written.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``source`` is another kind of artifact or a field
                is not stored.
            MemoryError: If the export's estimated peak exceeds the memory
                budget.
        """
        from ...writers.airr import (
            AIRR_REQUIRED_FIELDS,
            AirrRearrangementWriter,
            suffix_repeated_ids,
        )

        if not isinstance(chains_only, bool):
            raise TypeError("chains_only must be a boolean")
        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path")
        if not isinstance(source, ArtifactRef):
            raise TypeError("source must be an ArtifactRef")
        clonotypes = source.kind == ClonotypeArguments.artifact_kind
        if clonotypes:
            self._clonotype_group(source)
            receptors = inspect_artifact(self.zw, source).input_ref("receptors")
        elif source.kind == ImmuneReceptorImportArguments.artifact_kind:
            receptors = source
        else:
            raise ValueError(
                "source must be an immune_receptors or clonotypes artifact, not "
                f"{source.kind!r}"
            )
        if fields is not None:
            if isinstance(fields, str) or not isinstance(fields, Sequence):
                raise TypeError("fields must be a sequence of field names or None")
            # Every row needs its contig ID.
            fields = [
                "sequence_id",
                *[name for name in fields if name != "sequence_id"],
            ]
        group, stored = self._receptor_group(receptors)
        if fields is None:
            chosen = stored
        else:
            unknown = [name for name in fields if name not in stored]
            if unknown:
                raise ValueError(
                    "The receptors store no field "
                    + ", ".join(repr(str(name)) for name in unknown)
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
        # A clonotypes artifact's labels replace an imported clone_id.
        contigs = _ExportContigs(
            group,
            tuple(name for name in chosen if not (clonotypes and name == "clone_id")),
            chains_only=chains_only,
        )
        n_cells = int(as_zarr_array(group["values"], name="values").shape[0])
        plan = contigs.plan(
            n_cells=n_cells,
            n_columns=len(AIRR_REQUIRED_FIELDS)
            + sum(name not in AIRR_REQUIRED_FIELDS for name in order),
            clonotypes=clonotypes,
            budget=self.memoryBytes,
        )
        if plan.peak > self.memoryBytes:
            raise MemoryError(
                f"Exporting the {contigs.n_contigs} contigs of {n_cells} cells with "
                f"{len(chosen)} fields holds about {plan.peak} bytes at its peak, "
                f"more than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget, or pass fields= with fewer fields."
            )
        selection = inspect_artifact(self.zw, receptors).input_ref("cell_selection")
        clones = None
        if clonotypes:
            # The labels are read at the receptors' cells, which contig_cells
            # counts.
            labels = self._artifacts_load_values(source, cell_selection=selection)
            cell_ids, clones = labels.cell_ids, _clone_text(labels.values)
            del labels
        else:
            # The cells' IDs come with their smallest per-cell array.
            cell_ids = self._artifacts_load_values(
                receptors, value="n_contigs"
            ).cell_ids
        cells = EncodedText.from_categories(np.empty(0, np.int32), cell_ids)
        del cell_ids
        hashes, whole = contigs.scan(plan.block_rows)

        def ids_at(rows: np.ndarray) -> np.ndarray:
            held = plan.held + _ID_BYTES * len(rows)
            if held > self.memoryBytes:
                raise MemoryError(
                    f"Naming {len(rows)} repeated contig IDs holds about {held} "
                    f"bytes, more than the memory budget of {self.memoryBytes} "
                    "bytes. Raise mem_budget."
                )
            return contigs.ids_at(rows, plan.block_rows)

        renamed = suffix_repeated_ids(hashes, ids_at)
        del hashes
        if len(renamed[0]):
            logger.warning(
                f"{len(renamed[0])} contig IDs repeat; their later occurrences "
                "are written with a suffix such as _2"
            )
        with AirrRearrangementWriter(path, order, whole=whole) as writer:
            for block in contigs.blocks(plan.block_rows, cells, clones, renamed):
                writer.write(block)
        written = writer.n_rows
        logger.info(f"Wrote {written} contigs to {os.fspath(path)}")
        return written


# An AIRR export renders its contigs in blocks of about this many bytes of
# text, a whole number of blocks per stored chunk.
_EXPORT_BLOCK_BYTES = 32 * 2**20
# The rendered bytes of a cell ID, a clonotype label, and a value of each kind
# of field besides text.
_EXPORT_CELL_ID_BYTES = 32
_EXPORT_LABEL_BYTES = 16
_EXPORT_VALUE_BYTES = {"integer": 7, "flag": 2, "number": 20}
# The bytes that a block holds per contig and field while it is written,
# besides the text that it reads and writes: a text field's codes, and an
# integer, flag or number as it is rendered.
_EXPORT_FIELD_BYTES = {"text": 8, "integer": 25, "flag": 16, "number": 70}
# Resident memory grows by about 1.6 bytes per byte that a block holds, and by
# up to 64 MiB for Zarr's reads, as measured on 100,000 cells of AIRR files
# and 200,000 cells of Cell Ranger files.
_EXPORT_BLOCK_FACTOR = 1.7
_EXPORT_READ_BYTES = 64 * 2**20
# Reading and packing the cells' IDs holds up to about 600 bytes per cell at
# its peak, and reading their clonotype labels 250 more; packed, both hold 48.
_EXPORT_SETUP_BYTES_PER_CELL = 600
_EXPORT_LABEL_BYTES_PER_CELL = 250
_EXPORT_HELD_BYTES_PER_CELL = 48
# Hashing the contig IDs holds about this many bytes per contig, and naming a
# repeated ID about this many per ID read.
_HASH_BYTES = 40
_ID_BYTES = 200


@dataclass(frozen=True, slots=True)
class _ExportPlan:
    """The block size of an AIRR export and the bytes it holds."""

    block_rows: int
    held: int
    peak: int


class _Window:
    """Read a stored array in ascending slices, one stored chunk at a time."""

    def __init__(self, array: zarr.Array) -> None:
        self._array = array
        self._chunk = max(1, int(array.chunks[0]))
        self._start = 0
        self._values = np.asarray(array[0:0])

    def read(self, start: int, stop: int) -> np.ndarray:
        if start < self._start or stop > self._start + len(self._values):
            first = start - start % self._chunk
            last = min(-(-stop // self._chunk) * self._chunk, self._array.shape[0])
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


def _clone_text(labels: np.ndarray) -> EncodedText:
    """Encode each cell's clonotype label, -1 for a cell without one."""
    from ...repertoire.contigs import encode_text

    values = np.asarray(labels, dtype=object).copy()
    values[np.asarray(pd.isna(values), dtype=bool) | (values == "")] = None
    return encode_text(values)


class _ExportContigs:
    """The contigs of an ``immune_receptors`` artifact, read for an AIRR export.

    Each pass reads blocks of contigs in order from the first; with
    ``chains_only``, a block's rows are its contigs that count as chains.
    """

    def __init__(
        self, group: zarr.Group, fields: tuple[str, ...], *, chains_only: bool
    ) -> None:
        self._group = group
        self._fields = fields
        self._chains_only = chains_only
        cells = as_zarr_array(group["contig_cells"], name="contig_cells")
        self.n_contigs = int(cells.shape[0])
        self._chunk = max(1, int(cells.chunks[0]))

    def _array(self, name: str) -> zarr.Array:
        return as_zarr_array(self._group[name], name=name)

    def plan(
        self, *, n_cells: int, n_columns: int, clonotypes: bool, budget: int
    ) -> _ExportPlan:
        """Choose the contigs of a block and count the bytes that the export holds.

        Args:
            n_cells: The receptors' cells.
            n_columns: The columns of the file.
            clonotypes: Whether each cell's clonotype is written.
            budget: The memory budget; a block takes at most a sixteenth.
        """
        n_contigs = max(1, self.n_contigs)
        # Each contig's rendered row, with its tabs and newline, and the other
        # bytes that a block holds per contig: its cell's and clonotype's
        # codes, and its row's offsets.
        rendered = n_columns + _EXPORT_CELL_ID_BYTES
        working = 32
        if clonotypes:
            rendered += _EXPORT_LABEL_BYTES
            working += 8
        categories = 0
        # A stored chunk of contig_cells and contig_chain, and of each field.
        windows = 9 * self._chunk
        for name in self._fields:
            kind = FIELDS[name]
            if kind == TEXT:
                data = int(self._array(f"contig_{name}_values").shape[0])
                n_values = int(self._array(f"contig_{name}_offsets").shape[0]) - 1
                categories += data + 8 * (n_values + 1)
                rendered += data // max(1, n_values)
                working += _EXPORT_FIELD_BYTES[kind]
                windows += 4 * self._chunk
            elif kind == RAGGED:
                # The bytes read, which are also written.
                text = int(self._array(f"contig_{name}").shape[0]) // n_contigs
                rendered += text
                working += text
                windows += 8 * self._chunk
            else:
                rendered += _EXPORT_VALUE_BYTES[kind]
                working += _EXPORT_FIELD_BYTES[kind]
                windows += 8 * self._chunk
        block_bytes = max(1, min(_EXPORT_BLOCK_BYTES, budget // 16))
        wanted = max(1, min(self._chunk, block_bytes // rendered))
        # Blocks divide each stored chunk evenly, so a chunk is read once.
        block_rows = -(-self._chunk // -(-self._chunk // wanted))
        held = categories + _EXPORT_HELD_BYTES_PER_CELL * n_cells
        setup = n_cells * (
            _EXPORT_SETUP_BYTES_PER_CELL
            + (_EXPORT_LABEL_BYTES_PER_CELL if clonotypes else 0)
        )
        block = (
            windows
            + int(_EXPORT_BLOCK_FACTOR * (rendered + working) * block_rows)
            + _EXPORT_READ_BYTES
        )
        return _ExportPlan(
            block_rows=block_rows,
            held=held + _HASH_BYTES * self.n_contigs,
            peak=held + max(setup, _HASH_BYTES * self.n_contigs, block),
        )

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

    def scan(self, block_rows: int) -> tuple[np.ndarray, set[str]]:
        """Hash the ID of each row and find the number fields that are whole.

        Returns:
            The ``sequence_id_hashes`` of the rows, and the number fields
            whose rows hold only whole numbers, which are written as integers.
        """
        from ...repertoire.contigs import NUMBER
        from ...writers.airr import sequence_id_hashes

        ids = _RaggedWindow(self._group, "sequence_id")
        numbers = {
            name: _Window(self._array(f"contig_{name}"))
            for name in self._fields
            if FIELDS[name] == NUMBER
        }
        whole = dict.fromkeys(numbers, True)
        hashes = []
        for start, stop, kept in self._spans(block_rows):
            hashes.append(
                sequence_id_hashes(_kept(decode_ragged(*ids.read(start, stop)), kept))
            )
            for name, window in numbers.items():
                if whole[name]:
                    values = _kept(window.read(start, stop), kept)
                    present = values[~np.isnan(values)]
                    whole[name] = bool(
                        np.isfinite(present).all()
                        and (present == np.round(present)).all()
                    )
        return (
            np.concatenate([np.empty(0, dtype=np.uint64), *hashes]),
            {name for name, flag in whole.items() if flag},
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
        from ...repertoire.contigs import FLAG, INTEGER

        positions = _Window(self._array("contig_cells"))
        text = {
            name: (
                _Window(self._array(f"contig_{name}")),
                _read(self._group, f"contig_{name}_values"),
                _read(self._group, f"contig_{name}_offsets"),
            )
            for name in self._fields
            if FIELDS[name] == TEXT
        }
        ragged = {
            name: _RaggedWindow(self._group, name)
            for name in self._fields
            if FIELDS[name] == RAGGED
        }
        values = {
            name: _Window(self._array(f"contig_{name}"))
            for name in self._fields
            if FIELDS[name] not in (TEXT, RAGGED)
        }
        renamed_rows, renamed_ids = renamed
        before = 0
        for start, stop, kept in self._spans(block_rows):
            rows = np.arange(stop - start) if kept is None else kept
            cell = _kept(positions.read(start, stop), kept)
            block: dict[str, Any] = {
                "cell_id": EncodedText(cell, cells.data, cells.offsets)
            }
            if clones is not None:
                block["clone_id"] = EncodedText(
                    np.asarray(clones.codes)[cell], clones.data, clones.offsets
                )
            for name, (window, data, offsets) in text.items():
                block[name] = EncodedText(
                    _kept(window.read(start, stop), kept), data, offsets
                )
            for name, reader in ragged.items():
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
                        bounds = np.concatenate(
                            [bounds, extra.offsets[1:] + bounds[-1]]
                        )
                        codes = rows.copy()
                        codes[renamed_rows[first:last] - before] = (
                            stop - start + np.arange(last - first)
                        )
                block[name] = EncodedText(codes, data, bounds)
            for name, window in values.items():
                stored = _kept(window.read(start, stop), kept)
                if FIELDS[name] == INTEGER:
                    block[name] = pd.arrays.IntegerArray(stored, stored < 0)
                elif FIELDS[name] == FLAG:
                    block[name] = pd.arrays.BooleanArray(stored == 1, stored < 0)
                else:
                    block[name] = stored
            yield block
            before += len(rows)
