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
)
from ...metadata.selection import (
    resolve_grouping,
    sample_codes,
    valid_category_mask,
)
from ...repertoire.chains import (
    NO_RECEPTOR,
    RECEPTOR_TYPES,
    index_chains,
    receptor_quality,
)
from ...repertoire.clonotypes import (
    DUAL_IR,
    RECEPTOR_ARMS,
    REPRESENTATIVE_FIELDS,
    SEQUENCES,
    Clonotypes,
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
from ...repertoire.diversity import (
    INDEX_COLUMNS,
    ORDERS,
    CloneSizes,
    ClonalDiversity,
    Rarefaction,
    common_coverage,
    common_size,
    coverage_at,
    default_sizes,
    diversity_at,
    estimated_hill,
    hill_curve,
    indices,
    observed_hill,
    size_for_coverage,
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
from ...repertoire.overlap import (
    ClonotypeOverlap,
    ClonotypeTracking,
    group_names,
    overlap_pairs,
    top_clonotypes,
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
from ...utils.arrays import sort_categories
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
# Bytes that a decoded contig field takes in a table, per contig, besides the
# text of ragged fields.
_TABLE_BYTES_PER_FIELD = 64
# Measuring clonal expansion holds, at its peak, about 270 bytes per cell: the
# clonotype and group codes, the clone sizes and frequencies, and the labels
# as written, as measured on 1,000,000 cells.
_EXPANSION_BYTES_PER_CELL = 300
# The tables of loaded clonotypes and clonal expansion hold about 850 and 450
# bytes per cell at their peak, with the cell IDs and labels as text, as
# measured on 1,000,000 cells.
_CLONOTYPE_TABLE_BYTES_PER_CELL = 1_000
_EXPANSION_TABLE_BYTES_PER_CELL = 500
# The column of each group's receptor type in the tables of clonal expansion.
_RECEPTOR_TYPE_COLUMN = "receptor_type"
# Grouping cells for diversity holds about 200 bytes per cell: the clonotype
# and group codes and each grouping column's labels.
_DIVERSITY_BYTES_PER_CELL = 300


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
        "contig_cells": np.asarray(table.cells, dtype=np.int64),
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


def _within_columns(within: Any, name: str = "within") -> tuple[str, ...]:
    """Return the cell metadata columns of an argument as a tuple of names."""
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

        cell_ids = np.asarray(self.cells.fetch_all("ids"))[cell_index].astype(str)
        lookup = pd.Index(cell_ids)
        lookup_bytes = _LOOKUP_BYTES_PER_CELL * n_cells
        encoder = ContigEncoder()
        n_file_contigs = 0
        n_unmatched = 0
        unmatched_barcodes: set[str] = set()
        examples: list[str] = []
        for position, source in enumerate(sources):
            prefix = source.barcode_prefix or ""
            chunk_rows = chunk_rows_for(self.memoryBytes, columns=source.n_columns)
            for _first_line, chunk in _iter_source(source, chunk_rows):
                barcodes = chunk["cell_id"].to_numpy(dtype=object).astype(str)
                rows = lookup.get_indexer(np.char.add(prefix, barcodes))
                matched = rows >= 0
                n_file_contigs += len(chunk)
                if not matched.all():
                    missing = barcodes[~matched]
                    n_unmatched += len(missing)
                    for barcode in pd.unique(missing):
                        if barcode not in unmatched_barcodes:
                            unmatched_barcodes.add(barcode)
                            if len(examples) < _UNMATCHED_EXAMPLES:
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
                "contig_chain": index.eligible,
                **_contig_arrays(table),
            },
            attributes={
                "fields": list(table.fields),
                "n_file_contigs": int(n_file_contigs),
                "n_unmatched_contigs": int(n_unmatched),
                "n_unmatched_barcodes": len(unmatched_barcodes),
                "unmatched_barcode_examples": examples,
            },
        )
        if n_unmatched:
            logger.warning(
                f"{n_unmatched} of {n_file_contigs} contigs, of "
                f"{len(unmatched_barcodes)} barcodes, match no selected cell, for "
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
        if not isinstance(receptors, ArtifactRef):
            raise TypeError("receptors must be an ArtifactRef")
        if receptors.kind != ImmuneReceptorImportArguments.artifact_kind:
            raise ValueError(
                "receptors must be an immune_receptors artifact, not "
                f"{receptors.kind!r}"
            )
        status = require_complete_artifact(self.zw, receptors)
        group = artifact_group(self.zw, receptors)
        stored = _stored_fields(group)
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
        loaded = {
            name: self._artifacts_load_values(
                receptors, value=None if name == "values" else name
            )
            for name in _CELL_ARRAYS
        }
        cell_ids = loaded["values"].cell_ids
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
            receptor_type=np.asarray(loaded["values"].values),
            receptor_subtype=np.asarray(loaded["receptor_subtype"].values),
            chain_pairing=np.asarray(loaded["chain_pairing"].values),
            n_vj=np.asarray(loaded["n_vj"].values),
            n_vdj=np.asarray(loaded["n_vdj"].values),
            n_contigs=np.asarray(loaded["n_contigs"].values),
            slots=slots,
            columns=columns,
            fields=chosen,
        )
        return ImmuneReceptors(
            cells=cells,
            chains=chains,
            fields=stored,
            productive_only=bool((status.parameters or {})["productive_only"]),
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
            invalidate_cache: Define again instead of reusing saved clonotypes.

        Returns:
            A complete datastore-scoped ``clonotypes`` artifact aligned to the
            receptors' cells. Read it with
            :meth:`~cytearc.datastore.namespaces.RepertoireAccessor.load_clonotypes`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an option is unknown, the receptors store no
                junction of the compared kind, or a ``within`` column has no
                label for a cell with a receptor.
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
        columns = _within_columns(within)
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
            _types, type_codes = np.unique(receptor_type, return_inverse=True)
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
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Defining the clonotypes of {n_cells} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        slots = _read(group, "chain_slots")
        assignment = assign_clonotypes(
            _slot_codes(group, field_name, slots),
            groups=groups,
            v_genes=_slot_codes(group, "v_call", slots) if same_v_gene else None,
            j_genes=_slot_codes(group, "j_call", slots) if same_j_gene else None,
            receptor_arms=receptor_arms,
            dual_ir=dual_ir,
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
        labels = self._artifacts_load_values(clonotypes)
        sizes = self._artifacts_load_values(clonotypes, value="clone_size")
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
            labels.cell_ids,
            values=np.asarray(labels.values),
            clone_size=np.asarray(sizes.values),
            clonotype_ids=_read(group, "clonotype_ids").astype(str),
            clonotype_sizes=_read(group, "clonotype_sizes"),
            n_configurations=_read(group, "n_configurations"),
            receptor_type=_read(receptor_group, "values").astype(str)[representatives],
            within={
                column: np.asarray(
                    [
                        within_labels[code][column] if code >= 0 else None
                        for code in within_codes.tolist()
                    ],
                    dtype=object,
                )
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
        columns = _within_columns(within)
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
            types, type_codes = np.unique(
                receptor_type.astype(str)[has], return_inverse=True
            )
        else:
            types = np.asarray([], dtype=str)
            type_codes = np.zeros(len(within_codes), dtype=np.int64)
        combinations, group_codes = np.unique(
            np.column_stack([within_codes, type_codes]).astype(np.int64),
            axis=0,
            return_inverse=True,
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
        groups = np.full(n_cells, -1, dtype=np.int64)
        groups[has] = group_codes.reshape(-1)
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
        loaded = self._artifacts_load_values(expansion)
        categories = pd.Index(labels).get_indexer(np.asarray(loaded.values, dtype=str))
        clonotype_group = self._clonotype_group(clonotypes)
        cells, table, ranks = expansion_tables(
            loaded.cell_ids,
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
            sizes=_read(group, "group_clone_size"),
            frequencies=_read(group, "group_clone_frequency"),
            splits=bounds,
        )
        counts = parameters["breakpoints"]
        frequencies = parameters["frequency_breakpoints"]
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
    ) -> tuple[tuple[str, ...], list[dict[str, str]], list[CloneSizes]]:
        """Group the cells with a clonotype and count each group's clone sizes.

        Returns:
            The key columns, each group's labels in them, and each group's
            clone sizes, groups in the natural order of their labels.
        """
        key_columns, group_keys, pair_groups, _codes, pair_cells = self._clone_counts(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
        )
        bounds = np.searchsorted(pair_groups, np.arange(len(group_keys) + 1))
        clone_sizes = [
            CloneSizes.from_sizes(pair_cells[bounds[index] : bounds[index + 1]])
            for index in range(len(group_keys))
        ]
        return key_columns, group_keys, clone_sizes

    def _clone_counts(
        self,
        clonotypes: ArtifactRef,
        *,
        group_by: str | Sequence[str] | None,
        groups: ArtifactRef | None,
        within_receptor_type: bool,
        cell_selection: ArtifactRef | None,
    ) -> tuple[
        tuple[str, ...], list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray
    ]:
        """Group the cells with a clonotype and count each clonotype's cells.

        The groups are the combinations of the labels of the ``group_by``
        columns, of the ``groups`` artifact and, with
        ``within_receptor_type``, of the receptor type. A cell without a
        clonotype, outside ``cell_selection``, or without a label, is in no
        group.

        Returns:
            The key columns, each group's labels in them, in the natural order
            of the labels, and for each distinct pair of group and clonotype,
            sorted by group and clonotype code, its group, its clonotype code
            and its cells.
        """
        clonotype_group = self._clonotype_group(clonotypes)
        if not isinstance(within_receptor_type, bool):
            raise TypeError("within_receptor_type must be a boolean")
        columns = _within_columns(group_by, "group_by")
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
        codes = _read(clonotype_group, "codes").astype(np.int64)
        has = codes >= 0
        resident = _DIVERSITY_BYTES_PER_CELL * len(codes)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Grouping the clonotypes of {len(codes)} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget."
            )
        status = inspect_artifact(self.zw, clonotypes)
        cell_index = read_stored_selection_indices(
            self.zw,
            status.input_ref("cell_selection"),
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
        labels: list[np.ndarray] = []
        for column in columns:
            values = np.asarray(read_metadata_rows(self.cells, column, cell_index))
            missing = read_metadata_missing_rows(self.cells, column, cell_index)
            labels.append(_labels_or_none(values, missing))
        if groups is not None:
            resolved = resolve_grouping(self.zw, self.cells, groups)
            values = np.asarray(resolved.labels, dtype=object)
            valid = valid_category_mask(values, missing_mask=resolved.missing_mask)
            lookup = pd.Series(
                values[valid], index=np.asarray(resolved.cell_idx)[valid]
            )
            aligned = lookup.reindex(cell_index).to_numpy(dtype=object)
            labels.append(_labels_or_none(aligned, pd.isna(aligned)))
        if within_receptor_type:
            receptor_type = _read(
                artifact_group(self.zw, status.input_ref("receptors")), "values"
            ).astype(str)
            labels.append(receptor_type[has].astype(object))
        clone_codes = codes[has]
        if not labels:
            present, cells = np.unique(clone_codes, return_counts=True)
            return (
                key_columns,
                [{}],
                np.zeros(len(present), dtype=np.int64),
                present.astype(np.int64),
                cells.astype(np.int64),
            )
        label_codes = []
        names_by_column = []
        keep = np.ones(len(clone_codes), dtype=bool)
        for values in labels:
            present = np.asarray([value is not None for value in values], dtype=bool)
            names = sort_categories(pd.unique(values[present]))
            lookup = {name: position for position, name in enumerate(names)}
            column_codes = np.full(len(values), -1, dtype=np.int64)
            column_codes[present] = [lookup[value] for value in values[present]]
            label_codes.append(column_codes)
            names_by_column.append(names)
            keep &= present
        stacked = np.column_stack(label_codes)[keep]
        combinations, group_codes = np.unique(stacked, axis=0, return_inverse=True)
        group_codes = group_codes.reshape(-1)
        group_keys = [
            {
                column: str(names[code])
                for column, names, code in zip(
                    key_columns, names_by_column, row, strict=True
                )
            }
            for row in combinations.tolist()
        ]
        kept = clone_codes[keep]
        n_codes = int(kept.max()) + 1 if len(kept) else 1
        pairs, cells = np.unique(group_codes * n_codes + kept, return_counts=True)
        return (
            key_columns,
            group_keys,
            (pairs // n_codes).astype(np.int64),
            (pairs % n_codes).astype(np.int64),
            cells.astype(np.int64),
        )

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
        key_columns, group_keys, clone_sizes = self._clone_groups(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
        )
        if not clone_sizes or all(clones.n == 0 for clones in clone_sizes):
            raise ValueError("No cell has both a clonotype and a group")
        common = common_size(clone_sizes) if size is None else int(size)
        level = common_coverage(clone_sizes) if coverage is None else float(coverage)
        index_rows = []
        hill_rows = []
        for key, clones in zip(group_keys, clone_sizes, strict=True):
            index_rows.append({**key, **indices(clones)})
            needed = size_for_coverage(clones, level) if level < 1 else float("nan")
            for order in ORDERS:
                at_size, size_method = diversity_at(clones, common, order)
                at_coverage, coverage_method = diversity_at(clones, needed, order)
                hill_rows.append(
                    {
                        **key,
                        "order": order,
                        "observed": observed_hill(clones, order),
                        "estimated": estimated_hill(clones, order),
                        "at_size": at_size,
                        "coverage_at_size": coverage_at(clones, common),
                        "method_at_size": size_method,
                        "at_coverage": at_coverage,
                        "size_at_coverage": needed,
                        "method_at_coverage": coverage_method,
                    }
                )
        return ClonalDiversity(
            indices=pd.DataFrame(index_rows, columns=[*key_columns, *INDEX_COLUMNS]),
            hill=pd.DataFrame(hill_rows),
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
        fixed: np.ndarray | None = None
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
            fixed = np.unique(np.asarray(sizes, dtype=np.int64))
        # The defaults' arguments are checked once, before any group is read.
        default_sizes(2, endpoint, knots)
        key_columns, group_keys, clone_sizes = self._clone_groups(
            clonotypes,
            group_by=group_by,
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
        )
        if not clone_sizes or all(clones.n == 0 for clones in clone_sizes):
            raise ValueError("No cell has both a clonotype and a group")
        rows = []
        for key, clones in zip(group_keys, clone_sizes, strict=True):
            group_sizes = (
                default_sizes(clones.n, endpoint, knots) if fixed is None else fixed
            )
            covered = [coverage_at(clones, float(m)) for m in group_sizes]
            for order in chosen:
                values, methods = hill_curve(clones, group_sizes.tolist(), order)
                for m, sample_coverage, value, method in zip(
                    group_sizes, covered, values, methods, strict=True
                ):
                    rows.append(
                        {
                            **key,
                            "order": order,
                            "size": int(m),
                            "method": method,
                            "diversity": value,
                            "coverage": sample_coverage,
                        }
                    )
        return Rarefaction(
            curves=pd.DataFrame(
                rows,
                columns=[
                    *key_columns,
                    "order",
                    "size",
                    "method",
                    "diversity",
                    "coverage",
                ],
            ),
            clonotypes=clonotypes,
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
            orders=chosen,
        )

    def _group_table(
        self,
        key_columns: tuple[str, ...],
        group_keys: list[dict[str, str]],
        pair_groups: np.ndarray,
        pair_cells: np.ndarray,
    ) -> pd.DataFrame:
        """One row per group: its name, labels, cells and clonotypes."""
        n_groups = len(group_keys)
        return pd.DataFrame(
            {
                "group": group_names(group_keys),
                **{
                    column: [key[column] for key in group_keys]
                    for column in key_columns
                },
                "cells": np.bincount(
                    pair_groups, weights=pair_cells, minlength=n_groups
                ).astype(np.int64),
                "clonotypes": np.bincount(pair_groups, minlength=n_groups),
            }
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
            MemoryError: If grouping the cells does not fit the memory budget.
        """
        key_columns, group_keys, pair_groups, pair_codes, pair_cells = (
            self._clone_counts(
                clonotypes,
                group_by=group_by,
                groups=groups,
                within_receptor_type=within_receptor_type,
                cell_selection=cell_selection,
            )
        )
        if len(pair_cells) == 0 or len(group_keys) < 2:
            raise ValueError(
                "Overlap needs at least two groups of cells with a clonotype"
            )
        sizes = self._group_table(key_columns, group_keys, pair_groups, pair_cells)
        strata = (
            np.asarray([key[_RECEPTOR_TYPE_COLUMN] for key in group_keys])
            if within_receptor_type
            else None
        )
        pairs = overlap_pairs(
            pair_groups, pair_codes, pair_cells, len(group_keys), strata=strata
        )
        names = np.asarray(sizes["group"], dtype=object)
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
            One row per chosen clonotype and group.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``top`` is not positive, a label names no
                clonotype, or no cell has a clonotype and a group.
            MemoryError: If grouping the cells does not fit the memory budget.
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
        key_columns, group_keys, pair_groups, pair_codes, pair_cells = (
            self._clone_counts(
                clonotypes,
                group_by=group_by,
                groups=groups,
                within_receptor_type=within_receptor_type,
                cell_selection=cell_selection,
            )
        )
        if len(pair_cells) == 0:
            raise ValueError("No cell has both a clonotype and a group")
        identifiers = _read(self._clonotype_group(clonotypes), "clonotype_ids").astype(
            str
        )
        if labels is None:
            chosen = top_clonotypes(
                pair_groups, pair_codes, pair_cells, len(group_keys), int(top)
            )
        else:
            positions = pd.Index(identifiers).get_indexer(list(labels))
            unknown = [label for label, at in zip(labels, positions) if at < 0]
            if unknown:
                raise ValueError(f"No clonotype is labelled {unknown[:5]}")
            chosen = np.unique(positions)
        sizes = self._group_table(key_columns, group_keys, pair_groups, pair_cells)
        strata = (
            np.asarray([key[_RECEPTOR_TYPE_COLUMN] for key in group_keys])
            if within_receptor_type
            else np.zeros(len(group_keys), dtype=object)
        )
        observed = pd.DataFrame(
            {"code": pair_codes, "group": pair_groups, "cells": pair_cells}
        )
        observed = observed[observed["code"].isin(chosen)]
        rows = []
        for code in chosen.tolist():
            mine = observed[observed["code"] == code]
            cells_by_group = dict(zip(mine["group"], mine["cells"], strict=True))
            # The groups of every stratum that the clonotype has cells in.
            reach = set(strata[mine["group"].to_numpy()].tolist())
            for group in range(len(group_keys)):
                if strata[group] not in reach:
                    continue
                count = int(cells_by_group.get(group, 0))
                rows.append(
                    {
                        "clonotype": identifiers[code],
                        "group": sizes["group"].iat[group],
                        **group_keys[group],
                        "cells": count,
                        "fraction": count / int(sizes["cells"].iat[group]),
                    }
                )
        return ClonotypeTracking(
            cells=pd.DataFrame(
                rows,
                columns=["clonotype", "group", *key_columns, "cells", "fraction"],
            ),
            clonotypes=clonotypes,
            tracked=tuple(identifiers[chosen].tolist()),
            group_by=_within_columns(group_by, "group_by"),
            groups=groups,
            within_receptor_type=within_receptor_type,
            cell_selection=cell_selection,
        )


def _labels_or_none(values: np.ndarray, missing: Any) -> np.ndarray:
    """Return labels as objects, None where a label is missing or blank."""
    array = np.asarray(values, dtype=object).copy()
    valid = valid_category_mask(
        array, missing_mask=None if missing is None else np.asarray(missing, dtype=bool)
    )
    array[~valid] = None
    array[valid] = [str(value) for value in array[valid]]
    return array
