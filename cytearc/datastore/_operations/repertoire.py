import hashlib
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import ClonotypeArguments, ImmuneReceptorImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    read_metadata_missing_rows,
    read_metadata_rows,
)
from ...metadata.selection import sample_codes
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


def _within_columns(
    within: Any, reserved: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    """Return the cell metadata columns of ``within`` as a tuple of names.

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
        raise TypeError("within must be a column name, a sequence of names, or None")
    if not all(isinstance(name, str) and name for name in names):
        raise TypeError("within must be a column name, a sequence of names, or None")
    if len(set(names)) != len(names):
        raise ValueError("within names a column twice")
    clashes = [column for column in names if column in reserved]
    if clashes:
        raise ValueError(
            f"within names {clashes[0]!r}, which is also a column of the result; "
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
                # The cell ID that a contig names: its file's prefix and its
                # barcode, so equal barcodes of two samples stay apart.
                identifiers = np.char.add(prefix, barcodes)
                rows = lookup.get_indexer(identifiers)
                matched = rows >= 0
                n_file_contigs += len(chunk)
                if not matched.all():
                    missing = identifiers[~matched]
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
        columns = _within_columns(within, reserved=_CLONOTYPE_COLUMNS)
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
        fingerprint = None
        within_labels: list[dict[str, str]] = [{}]
        if columns:
            cell_index = read_stored_selection_indices(
                self.zw,
                cell_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)[with_receptor]
            column_labels = [
                read_metadata_rows(self.cells, column, cell_index) for column in columns
            ]
            missing = [
                read_metadata_missing_rows(self.cells, column, cell_index)
                for column in columns
            ]
            codes, within_labels, fingerprint = sample_codes(
                columns,
                [
                    apply_missing_mask(np.asarray(values), mask, labels=True)
                    for values, mask in zip(column_labels, missing, strict=True)
                ],
                len(cell_index),
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
        """
        group = self._clonotype_group(clonotypes)
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
