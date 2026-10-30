import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import VariantCountImportArguments
from ...metadata.artifacts import plan_cell_data_artifact
from ...metadata.rows import read_metadata_rows_chunkwise
from ...storage.arrays import create_metadata_column
from ...storage.artifact_writer import PlannedArtifact, artifact_transaction
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.layout import get_compressors
from ...storage.profiles import resolve_storage_profile
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.logging import logger
from ...variants.counts import (
    CALL_DTYPE,
    COUNT_FIELDS,
    COUNT_UNITS,
    DEPTH_DTYPE,
    MAX_COUNT,
    Contig,
    CountUnit,
    EntryBatch,
    SiteTable,
    VariantCounts,
    VariantSource,
    count_dtype,
    finish_band,
    merge_sites,
    variant_count_frames,
)
from ...variants.sorting import CellRowSorter
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    warn_unmatched_barcodes,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _VariantOperationsBase
else:
    _VariantOperationsBase = object

# Rows of each chunk of a stored long table: the row bands that a reader of a
# band of cells decompresses.
_ROW_BAND = 1 << 20
# Bytes held for each selected cell while counts are imported: whether a file
# names it, its depth sum and covered sites, and three offsets.
_IMPORT_BYTES_PER_CELL = 64
# Bytes of one parsed row at its peak: the parsed table, the batch's arrays and
# the row it becomes.
_PARSE_BYTES_PER_ROW = 320
# Each band row is held about this many times: the rows read, their sort, and
# the stored rows derived from them.
_BAND_COPIES = 6
# How the memory left beside the per-cell state is shared: the parsed chunks
# take a fifth; the sort buffer a tenth, since a spill holds about three
# buffers; and the bands, read beside the last buffer, two fifths.
_PARSE_SHARE = 5
_BUFFER_SHARE = 10
_BAND_SHARE = 2.5
# The rows parsed at once lie in [_MIN_CHUNK_ROWS, _MAX_CHUNK_ROWS].
_MIN_CHUNK_ROWS = 1_000
_MAX_CHUNK_ROWS = 2_000_000
# Bytes per row of the loaded tables: codes, values and pandas' overhead.
_LOADED_BYTES_PER_ROW = 120
# Bytes per cell of the loaded cells table and its index.
_LOADED_BYTES_PER_CELL = 200
# The per-cell arrays of a variant_counts artifact, its canonical one first.
_CELL_ARRAYS = ("mean_depth", "covered_sites", "listed")
_FILES_TEXT = ("the variant file", "the variant files")


def _read_source(
    path: str, format: str | None, swap_strands: bool, threads: int
) -> VariantSource:
    """Read the description of one file or folder of variant counts."""
    from ...readers import _variant_files as files

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    detected = format
    if detected is None:
        if os.path.isdir(path):
            detected = files.folder_format(path)
        elif files.is_hdf5(path):
            detected = files.hdf5_format(path)
        else:
            from ...readers.got import got_table_format

            detected = got_table_format(path)
    if detected not in ("mgatk", "maegatk") and swap_strands:
        raise ValueError(
            "swap_strands applies to mgatk files from call or bcall of mgatk 0.4.0 "
            f"to 0.5.8, not to {detected} files"
        )
    if detected in ("mgatk", "maegatk"):
        from ...readers.mgatk import mgatk_source

        return mgatk_source(
            path, format=format, swap_strands=swap_strands, threads=threads
        )
    if detected in ("cellsnp_lite", "mquad"):
        from ...readers.cellsnp import cellsnp_source

        return cellsnp_source(path, format=detected)
    if detected == "vartrix":
        from ...readers.vartrix import vartrix_source

        return vartrix_source(path)
    if detected == "giftwrap":
        from ...readers.giftwrap import giftwrap_source

        return giftwrap_source(path)
    if detected == "mito":
        from ...readers.mito_afm import mito_afm_source

        return mito_afm_source(path)
    from ...readers.got import got_source

    return got_source(path, format=detected)


@dataclass(frozen=True, slots=True)
class _Source:
    """A source file with how its barcodes name cells and its identity."""

    file: BarcodeFile
    reader: VariantSource
    digests: dict[str, str]

    def record(self) -> dict[str, Any]:
        return {
            "format": self.reader.format,
            "files": dict(sorted(self.digests.items())),
            **self.file.record(),
            **self.reader.notes,
        }


def _repeated_row(rows: np.ndarray, keys: Sequence[str]) -> int | None:
    """The first row whose keys equal the row before it, or None."""
    if len(rows) < 2:
        return None
    same = np.ones(len(rows) - 1, dtype=bool)
    for name in keys:
        same &= rows[name][1:] == rows[name][:-1]
    hits = np.flatnonzero(same)
    return int(hits[0]) + 1 if len(hits) else None


def _distinct_calls(rows: np.ndarray) -> tuple[np.ndarray, int | None, int]:
    """Keep the first of each run of calls of one cell, allele and source.

    Returns:
        The kept rows; for repeats that differ in their call, the row of the
        first such repeat, or None; and the number of rows dropped.
    """
    if len(rows) < 2:
        return rows, None, 0
    same = (
        (rows["cell"][1:] == rows["cell"][:-1])
        & (rows["key"][1:] == rows["key"][:-1])
        & (rows["source"][1:] == rows["source"][:-1])
    )
    differing = same & (rows["value"][1:] != rows["value"][:-1])
    if differing.any():
        return rows, int(np.flatnonzero(differing)[0]) + 1, 0
    kept = np.r_[True, ~same]
    return rows[kept], None, int(np.count_nonzero(same))


class _Vocabulary:
    """Text coded in order of first appearance, sorted when stored."""

    def __init__(self) -> None:
        self._codes: dict[str, int] = {}

    def codes(self, texts: np.ndarray) -> np.ndarray:
        distinct, inverse = np.unique(
            np.asarray(texts, dtype=object), return_inverse=True
        )
        lookup = np.asarray(
            [self._codes.setdefault(str(text), len(self._codes)) for text in distinct],
            dtype=np.int32,
        )
        return lookup[inverse]

    def sorted(self) -> tuple[list[str], np.ndarray]:
        """The texts in order, and each code's position among them."""
        texts = sorted(self._codes)
        remap = np.empty(len(self._codes), dtype=np.int32)
        for position, text in enumerate(texts):
            remap[self._codes[text]] = position
        return texts, remap


class _Appender:
    """Append rows to one-dimensional arrays of an artifact in row bands."""

    def __init__(self, group: zarr.Group, dtypes: Mapping[str, Any]) -> None:
        compressors = get_compressors(
            resolve_storage_profile(group.store),
            zarrFormat=int(group.metadata.zarr_format),
        )
        self.arrays = {
            name: group.create_array(
                name,
                shape=(0,),
                chunks=(_ROW_BAND,),
                dtype=dtype,
                compressors=compressors,
            )
            for name, dtype in dtypes.items()
        }
        self.n_rows = {name: 0 for name in dtypes}

    def append(self, name: str, values: np.ndarray) -> None:
        if not len(values):
            return
        array = self.arrays[name]
        start = self.n_rows[name]
        stop = start + len(values)
        array.resize((stop,))
        array[start:stop] = values
        self.n_rows[name] = stop


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    """Write a whole array: text as a metadata column, numbers chunked by rows."""
    if values.dtype.kind in "OUS":
        create_metadata_column(
            group,
            name,
            data=np.asarray(values, dtype=object).astype(str),
            overwrite=True,
            chunkSize=min(max(len(values), 1), 100_000),
        )
        return
    compressors = get_compressors(
        resolve_storage_profile(group.store),
        zarrFormat=int(group.metadata.zarr_format),
    )
    array = group.create_array(
        name,
        shape=values.shape,
        chunks=(min(max(len(values), 1), _ROW_BAND),),
        dtype=values.dtype,
        compressors=compressors,
    )
    if len(values):
        array[:] = values


def _read(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[:])


def _stored_sites(group: zarr.Group) -> SiteTable:
    records = cast(list[dict[str, Any]], group.attrs["contigs"])
    contigs = tuple(
        Contig(
            name=str(contig["name"]),
            length=None if contig["length"] is None else int(contig["length"]),
            sequence_sha256=contig["sequence_sha256"],
        )
        for contig in records
    )
    return SiteTable(
        contigs=contigs,
        site_contig=_read(group, "site_contigs"),
        site_position=_read(group, "site_positions"),
        site_name=_read(group, "site_names").astype(str).astype(object),
        site_channel=_read(group, "site_channels").astype(str).astype(object),
        allele_site=_read(group, "allele_sites"),
        allele=_read(group, "alleles").astype(str).astype(object),
        allele_role=_read(group, "allele_roles").astype(str).astype(object),
    )


def _site_positions(table: SiteTable, sites: Any) -> np.ndarray:
    """The sites that ``sites`` names, as ascending positions in the table."""
    if sites is None:
        return np.arange(table.n_sites, dtype=np.int64)
    if isinstance(sites, str | bytes | int) or not isinstance(sites, Sequence):
        raise TypeError(
            "sites must be a sequence of site labels, such as ['chrM:310'], or of "
            "positions, or None"
        )
    labels = {label: site for site, label in enumerate(table.labels().tolist())}
    chosen: list[int] = []
    unknown: list[Any] = []
    for item in sites:
        if isinstance(item, str):
            site = labels.get(item)
        elif isinstance(item, int | np.integer) and not isinstance(item, bool):
            coordinates = np.flatnonzero(table.site_contig >= 0)
            if len(table.contigs) != 1:
                raise ValueError(
                    "Sites are named by position only on an artifact of one contig; "
                    "name them by label, such as 'chrM:310'"
                )
            at = coordinates[table.site_position[coordinates] == int(item)]
            site = int(at[0]) if len(at) else None
        else:
            raise TypeError(
                f"sites must hold site labels or positions, not {type(item).__name__}"
            )
        if site is None:
            unknown.append(item)
        else:
            chosen.append(site)
    if unknown:
        raise ValueError(
            f"The artifact has no site {', '.join(repr(item) for item in unknown[:5])}"
        )
    return np.unique(np.asarray(chosen, dtype=np.int64))


class _VariantOperationsMixin(_VariantOperationsBase):
    def _variant_selection_indices(self, cell_selection: ArtifactRef) -> np.ndarray:
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

    @validation_scoped
    def _variants_import_counts(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        cell_selection: ArtifactRef,
        barcode_prefix: str | Sequence[str | None] | None = None,
        barcode_suffix: str | Sequence[str | None] | None = None,
        barcode_pattern: str | Sequence[str | None] | None = None,
        format: str | None = None,
        count_unit: str | None = None,
        swap_strands: bool = False,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import per-cell allele counts of mitochondrial or targeted variants.

        Mitochondrial lineage tracing (mtscATAC-seq, MAESTER) and targeted
        genotyping (GoT, GIFT) count, for each cell, the reads or UMIs of each
        allele at variant sites: the positions of the mitochondrial genome, or
        targets and probes without coordinates. The import reads those counts,
        joins each cell to a selected cell by barcode and saves a
        ``variant_counts`` artifact: the sites and their alleles, and for each
        cell its depth at each site, its count of each allele and the calls
        the source made itself. Depth 0 means the cell was not covered at the
        site, so an allele frequency there is undefined rather than 0.

        Each path is a folder or a file of one format, recognised from its
        contents: an mgatk folder (``<name>.A.txt.gz`` to ``.T.txt.gz``, the
        ``<contig>_refAllele.txt`` reference and the depth table), a maegatk
        folder, a cellsnp-lite or MQuad folder, a VarTrix folder of its
        coverage matrices with the VCF and barcodes, MiTo's allele frequency
        matrix, giftwrap's ``counts.N.h5`` or a folder of its per-UMI table,
        and the GoT tables of IronThrone, GoT-Multi, Gotcha and GEO. The
        variants API reference lists the files and columns of each format.
        Six-column mgatk files look like maegatk's, which counts strands the
        other way round, so they need ``format``.

        A cell's ID is its file's ``barcode_prefix``, the barcode and
        ``barcode_suffix``, matched exactly; ``barcode_pattern`` keeps the
        part of each barcode that its ``barcode`` group matches first.
        Barcodes that match no selected cell are counted, saved in the
        artifact's ``n_file_barcodes``, ``n_unmatched_barcodes`` and
        ``unmatched_barcode_examples`` attributes, and named in a warning,
        which a reused import gives again. A source is identified by the
        SHA-256 of each file it reads, so moved files reuse the saved counts
        and the order of the paths does not matter.

        Several paths must share a format, such as one mgatk folder per
        library with each library's prefix. Sites that they share are one
        site, so their counts must lie on the same reference: a contig with
        another length or sequence raises ``ValueError``, as does a cell that
        two files count at the same site.

        The canonical ``mean_depth`` holds each cell's depth summed over the
        sites and divided by their number, NaN for a selected cell that no
        file names; ``covered_sites`` counts its sites with depth above 0 and
        ``listed`` marks the cells a file names. ``ds.select_cells(counts,
        low=10)`` keeps cells with a mean depth of at least 10.

        Args:
            paths: One file or folder, or several of one format.
            cell_selection: The cells that the counts belong to.
            barcode_prefix: Text before each barcode in the cell IDs, such as
                ``"sample1__"`` in a merged store: one for every path, or one
                per path. None adds nothing.
            barcode_suffix: Text after each barcode in the cell IDs, such as
                ``"-1"``: one for every path, or one per path.
            barcode_pattern: A regular expression whose named group
                ``barcode`` is the part of each barcode that names the cell,
                such as ``r"^(?P<barcode>[ACGT]{16})"`` to trim a multiplexed
                Flex barcode; a barcode it does not match names no cell. One
                for every path, or one per path.
            format: One of ``"mgatk"``, ``"maegatk"``, ``"cellsnp_lite"``,
                ``"mquad"``, ``"vartrix"``, ``"mito"``, ``"giftwrap"``,
                ``"ironthrone"``, ``"got"``, ``"got_multi"`` and ``"gotcha"``,
                or None to recognise each path's format.
            count_unit: ``"reads"``, ``"umis"`` or ``"molecules"``: what
                cellsnp-lite, MQuad and VarTrix counted, which their files do
                not state, since it depends on how they were run. Other
                formats count in a unit of their own, which a given value must
                match.
            swap_strands: Swap the forward and reverse counts of mgatk files
                written by ``call`` or ``bcall`` of mgatk 0.4.0 to 0.5.8, which
                counted reverse-aligned reads as forward.
            invalidate_cache: Import again instead of reusing saved counts.

        Returns:
            A complete datastore-scoped ``variant_counts`` artifact. Read it
            with :meth:`~cytearc.datastore.namespaces.VariantsAccessor.load_counts`
            or plot it with ``ds.plots.variant_depth``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a format is unknown or its files are malformed,
                the paths hold several formats, a count unit is missing or
                differs from the format's, the files lie on different
                references, a cell is counted twice at a site, an option lists
                a value for another number of paths, or the selection holds no
                cell.
            FileNotFoundError: If a path or a file it needs is missing.
            PermissionError: If the store is read-only and the counts are not
                saved yet.
            MemoryError: If the import does not fit the memory budget.
        """
        from ...readers._variant_files import VARIANT_FORMATS, source_sha256

        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(swap_strands, bool):
            raise TypeError("swap_strands must be a boolean")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        if format is not None and format not in VARIANT_FORMATS:
            raise ValueError(
                f"format must be one of {', '.join(VARIANT_FORMATS)}, or None, not "
                f"{format!r}"
            )
        if count_unit is not None and count_unit not in COUNT_UNITS:
            raise ValueError(
                f"count_unit must be one of {', '.join(COUNT_UNITS)}, or None, not "
                f"{count_unit!r}"
            )
        readers: dict[str, tuple[VariantSource, str, dict[str, str]]] = {}

        def identify(path: str | os.PathLike[str]) -> str:
            key = os.fspath(path)
            if key not in readers:
                reader = _read_source(key, format, swap_strands, self.nthreads)
                digest, digests = source_sha256(reader.files)
                readers[key] = (reader, digest, digests)
            return readers[key][1]

        files = barcode_files(
            paths,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            what="variant file",
            identify=identify,
        )
        sources = [
            _Source(
                file=file,
                reader=readers[os.fspath(file.path)][0],
                digests=readers[os.fspath(file.path)][2],
            )
            for file in files
        ]
        formats = sorted({source.reader.format for source in sources})
        if len(formats) > 1:
            raise ValueError(
                f"The paths hold several formats, {', '.join(formats)}; import each "
                "format on its own"
            )
        first = sources[0].reader
        unit = self._variant_count_unit(first, count_unit)
        site_table, site_maps, allele_maps = merge_sites(
            [source.reader.sites() for source in sources],
            open_alleles=first.open_alleles,
        )
        cell_index = self._variant_selection_indices(cell_selection)
        n_cells = len(cell_index)
        arguments = VariantCountImportArguments(
            cell_selection=cell_selection,
            sources=tuple(source.record() for source in sources),
            count_unit=unit,
            swap_strands=swap_strands,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_variant_counts(arguments, cell_selection, n_cells)
        files_text = _FILES_TEXT[0] if len(sources) == 1 else _FILES_TEXT[1]
        if planned.reused:
            warn_unmatched_barcodes(
                artifact_group(self.zw, planned.ref).attrs, files_text
            )
            return planned.ref
        held = (LOOKUP_BYTES_PER_CELL + _IMPORT_BYTES_PER_CELL) * n_cells
        room = self.memoryBytes - held
        chunk_rows = int(
            min(room // (_PARSE_SHARE * _PARSE_BYTES_PER_ROW), _MAX_CHUNK_ROWS)
        )
        if chunk_rows < _MIN_CHUNK_ROWS:
            raise MemoryError(
                f"Importing variant counts of {n_cells} cells holds about {held} "
                f"bytes before it reads a row, which leaves too little of the memory "
                f"budget of {self.memoryBytes} bytes to read the files. Raise "
                "mem_budget, or pass a smaller cell_selection."
            )
        self._require_writable(arguments.operation)
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )
        join = CellBarcodeJoin(
            cell_ids,
            memory_bytes=self.memoryBytes,
            held_bytes=_IMPORT_BYTES_PER_CELL * n_cells,
        )
        counts_dtype = count_dtype(first.count_fields)
        sorter = CellRowSorter(
            {
                "counts": (counts_dtype, ("cell", "key")),
                "depth": (DEPTH_DTYPE, ("cell", "key")),
                "calls": (CALL_DTYPE, ("cell", "key", "source")),
            },
            n_cells=n_cells,
            buffer_bytes=room // _BUFFER_SHARE,
        )
        call_sources = _Vocabulary()
        call_values = _Vocabulary()
        listed = np.zeros(n_cells, dtype=bool)
        with sorter:
            n_rows = 0
            for source, site_map, allele_map in zip(
                sources, site_maps, allele_maps, strict=True
            ):
                # Matrix sources share one barcode array across their batches,
                # which is joined once.
                joined: tuple[np.ndarray | None, np.ndarray] = (None, np.empty(0))
                for batch in source.reader.batches(chunk_rows):
                    n_rows += batch.n_rows
                    if batch.barcodes is not joined[0]:
                        joined = (
                            batch.barcodes,
                            join.cells(source.file, batch.barcodes),
                        )
                    self._add_variant_batch(
                        sorter,
                        batch,
                        cells=joined[1],
                        site_map=site_map,
                        allele_map=allele_map,
                        counts_dtype=counts_dtype,
                        listed=listed,
                        call_sources=call_sources,
                        call_values=call_values,
                    )
            if not n_rows:
                raise ValueError(
                    f"No row of counts, depth, calls or cells is in {files_text}"
                )
            band_rows = max(
                1, int(room / _BAND_SHARE) // (_BAND_COPIES * counts_dtype.itemsize)
            )
            attributes = {
                **join.attributes(),
                "format": first.format,
                "count_unit": unit,
                "depth_definition": first.depth_definition,
                "stranded": "forward" in first.count_fields,
                "count_fields": list(first.count_fields),
                "contigs": [contig.record() for contig in site_table.contigs],
                "n_sorted_runs": 0,
            }
            self._write_variant_counts(
                planned,
                sorter,
                sites=site_table,
                reader=first,
                counts_dtype=counts_dtype,
                band_rows=int(band_rows),
                cell_ids=cell_ids,
                listed=listed,
                call_sources=call_sources,
                call_values=call_values,
                attributes=attributes,
            )
        warn_unmatched_barcodes(attributes, files_text)
        logger.info(
            f"Saved the variant counts of {int(listed.sum())} of {n_cells} cells at "
            f"{site_table.n_sites} sites"
        )
        return planned.ref

    @staticmethod
    def _variant_count_unit(reader: VariantSource, count_unit: str | None) -> CountUnit:
        if reader.count_unit is None:
            if count_unit is None:
                raise ValueError(
                    f"{reader.format} counts UMIs or reads depending on how it was "
                    "run, and its files do not say which; pass count_unit='umis' or "
                    "count_unit='reads'"
                )
            return cast(CountUnit, count_unit)
        if count_unit is not None and count_unit != reader.count_unit:
            raise ValueError(
                f"{reader.format} counts {reader.count_unit}, not {count_unit}; "
                "count_unit names the unit of formats whose files do not state it"
            )
        return cast(CountUnit, reader.count_unit)

    def _plan_variant_counts(
        self,
        arguments: VariantCountImportArguments,
        cell_selection: ArtifactRef,
        n_cells: int,
    ) -> PlannedArtifact:
        record = arguments.to_record()

        def has_tables(_ref: ArtifactRef, group: zarr.Group) -> bool:
            names = set(group.array_keys())
            required = {
                "depth_offsets",
                "count_offsets",
                "call_offsets",
                "depth_sites",
                "depth",
                "count_alleles",
                "counts",
                "call_alleles",
                "call_sources",
                "call_values",
                "site_contigs",
                "site_positions",
                "site_names",
                "site_channels",
                "site_depth",
                "site_cells",
                "allele_sites",
                "alleles",
                "allele_roles",
                "call_source_names",
                "call_value_names",
            }
            fields = group.attrs.get("count_fields")
            if not isinstance(fields, list) or not required <= names:
                return False
            if not {f"counts_{name}" for name in fields} <= names:
                return False
            return all(
                as_zarr_array(group[name], name=name).shape == (n_cells + 1,)
                for name in ("depth_offsets", "count_offsets", "call_offsets")
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
            arrays={
                "mean_depth": ((n_cells,), "f"),
                "covered_sites": ((n_cells,), "i"),
                "listed": ((n_cells,), "b"),
            },
            invalidate_cache=arguments.invalidate_cache,
            required_attributes=(
                "n_file_barcodes",
                "n_unmatched_barcodes",
                "unmatched_barcode_examples",
                "format",
                "count_unit",
                "depth_definition",
                "count_fields",
                "contigs",
            ),
            reuse_validator=has_tables,
        )

    @staticmethod
    def _add_variant_batch(
        sorter: CellRowSorter,
        batch: EntryBatch,
        *,
        cells: np.ndarray,
        site_map: np.ndarray,
        allele_map: np.ndarray,
        counts_dtype: np.dtype[Any],
        listed: np.ndarray,
        call_sources: "_Vocabulary",
        call_values: "_Vocabulary",
    ) -> None:
        """Join a batch's rows to cells and add those of selected cells."""
        row_cells = cells[batch.barcode_codes]
        kept = row_cells >= 0
        listed[row_cells[kept]] = True
        if batch.kind == "cells" or not kept.any():
            return
        if batch.kind == "calls":
            rows = np.empty(int(kept.sum()), dtype=CALL_DTYPE)
            rows["cell"] = row_cells[kept]
            rows["key"] = allele_map[batch.keys[kept]]
            rows["source"] = call_sources.codes(
                np.full(len(rows), batch.call_source, dtype=object)
            )
            rows["value"] = call_values.codes(batch.calls[kept])
            sorter.add("calls", rows)
            return
        values = np.asarray(batch.values, dtype=np.int64)[kept]
        if len(values) and (int(values.min()) < 0 or int(values.max()) > MAX_COUNT):
            raise ValueError(
                f"A {batch.kind} value lies outside 0 to {MAX_COUNT}, the range that "
                "counts are stored in"
            )
        if batch.kind == "depth":
            rows = np.empty(len(values), dtype=DEPTH_DTYPE)
            rows["cell"] = row_cells[kept]
            rows["key"] = site_map[batch.keys[kept]]
            rows["depth"] = values
            sorter.add("depth", rows)
            return
        rows = np.empty(len(values), dtype=counts_dtype)
        rows["cell"] = row_cells[kept]
        rows["key"] = allele_map[batch.keys[kept]]
        rows["count"] = values
        for name in counts_dtype.names or ():
            if name in ("cell", "key", "count"):
                continue
            field_values = np.asarray(batch.fields[name])[kept]
            if (
                COUNT_FIELDS[name].kind == "u"
                and len(field_values)
                and (np.nanmin(field_values) < 0 or np.nanmax(field_values) > MAX_COUNT)
            ):
                raise ValueError(
                    f"A value of {name} lies outside 0 to {MAX_COUNT}, the range "
                    "that counts are stored in"
                )
            rows[name] = field_values
        sorter.add("counts", rows)

    def _write_variant_counts(
        self,
        planned: PlannedArtifact,
        sorter: CellRowSorter,
        *,
        sites: SiteTable,
        reader: VariantSource,
        counts_dtype: np.dtype[Any],
        band_rows: int,
        cell_ids: np.ndarray,
        listed: np.ndarray,
        call_sources: _Vocabulary,
        call_values: _Vocabulary,
        attributes: dict[str, Any],
    ) -> None:
        n_cells = len(cell_ids)
        source_names, source_remap = call_sources.sorted()
        value_names, value_remap = call_values.sorted()
        depth_total = np.zeros(n_cells, dtype=np.float64)
        covered = np.zeros(n_cells, dtype=np.int64)
        site_depth = np.zeros(sites.n_sites, dtype=np.float64)
        site_cells = np.zeros(sites.n_sites, dtype=np.int64)
        offsets = {
            name: np.zeros(n_cells + 1, dtype=np.int64)
            for name in ("depth", "count", "call")
        }
        fields = [name for name in counts_dtype.names or () if name in COUNT_FIELDS]
        repeated_calls = 0

        def name_cell(cell: int) -> str:
            return str(cell_ids[cell])

        def allele_text(allele: int) -> str:
            site = int(sites.allele_site[allele])
            return f"{sites.allele[allele]!r} at the site {sites.label(site)}"

        with artifact_transaction(self.zw, planned) as group:
            appender = _Appender(
                group,
                {
                    "depth_sites": np.int32,
                    "depth": np.uint32,
                    "count_alleles": np.int32,
                    "counts": np.uint32,
                    **{f"counts_{name}": COUNT_FIELDS[name] for name in fields},
                    "call_alleles": np.int32,
                    "call_sources": np.int32,
                    "call_values": np.int32,
                },
            )
            for first, stop, rows in sorter.bands(band_rows):
                counts_rows = rows["counts"]
                repeated = _repeated_row(counts_rows, ("cell", "key"))
                if repeated is not None:
                    raise ValueError(
                        f"The cell {name_cell(int(counts_rows['cell'][repeated]))!r} "
                        "is counted twice for the allele "
                        f"{allele_text(int(counts_rows['key'][repeated]))}, by two "
                        "files or two rows of one file"
                    )
                depth_rows = rows["depth"]
                repeated = _repeated_row(depth_rows, ("cell", "key"))
                if repeated is not None:
                    raise ValueError(
                        f"The cell {name_cell(int(depth_rows['cell'][repeated]))!r} "
                        "has two depths at the site "
                        f"{sites.label(int(depth_rows['key'][repeated]))}, from two "
                        "files or two rows of one file"
                    )
                # A call that a source repeats is kept once; two different
                # calls of one source about one cell's allele are refused.
                call_rows, conflict, n_repeats = _distinct_calls(rows["calls"])
                if conflict is not None:
                    raise ValueError(
                        f"The cell {name_cell(int(call_rows['cell'][conflict]))!r} "
                        "has two different calls of "
                        f"{source_names[source_remap[call_rows['source'][conflict]]]} "
                        f"for the allele {allele_text(int(call_rows['key'][conflict]))}"
                    )
                repeated_calls += n_repeats
                depth_out, counts_out = finish_band(
                    counts_rows,
                    depth_rows,
                    sites=sites,
                    depth_roles=reader.depth_roles,
                    reference_from_depth=reader.reference_from_depth,
                    name_cell=name_cell,
                )
                width = stop - first
                band_cells = depth_out["cell"].astype(np.int64) - first
                depth_values = depth_out["depth"].astype(np.float64)
                depth_total[first:stop] += np.bincount(
                    band_cells, weights=depth_values, minlength=width
                )
                per_cell = np.bincount(band_cells, minlength=width)
                covered[first:stop] += per_cell
                site_depth += np.bincount(
                    depth_out["key"], weights=depth_values, minlength=sites.n_sites
                )
                site_cells += np.bincount(depth_out["key"], minlength=sites.n_sites)
                for name, cells_of_rows in (
                    ("depth", depth_out["cell"]),
                    ("count", counts_out["cell"]),
                    ("call", call_rows["cell"]),
                ):
                    band_counts = np.bincount(
                        cells_of_rows.astype(np.int64) - first, minlength=width
                    )
                    offsets[name][first + 1 : stop + 1] = offsets[name][
                        first
                    ] + np.cumsum(band_counts)
                appender.append("depth_sites", depth_out["key"])
                appender.append("depth", depth_out["depth"])
                appender.append("count_alleles", counts_out["key"])
                appender.append("counts", counts_out["count"])
                for name in fields:
                    appender.append(f"counts_{name}", counts_out[name])
                appender.append("call_alleles", call_rows["key"])
                appender.append("call_sources", source_remap[call_rows["source"]])
                appender.append("call_values", value_remap[call_rows["value"]])
            mean_depth = np.where(listed, depth_total / max(sites.n_sites, 1), np.nan)
            _write_array(group, "mean_depth", mean_depth)
            _write_array(group, "covered_sites", covered)
            _write_array(group, "listed", listed)
            _write_array(group, "depth_offsets", offsets["depth"])
            _write_array(group, "count_offsets", offsets["count"])
            _write_array(group, "call_offsets", offsets["call"])
            _write_array(group, "site_contigs", sites.site_contig)
            _write_array(group, "site_positions", sites.site_position)
            _write_array(group, "site_names", sites.site_name)
            _write_array(group, "site_channels", sites.site_channel)
            _write_array(group, "site_depth", site_depth.astype(np.int64))
            _write_array(group, "site_cells", site_cells)
            _write_array(group, "allele_sites", sites.allele_site)
            _write_array(group, "alleles", sites.allele)
            _write_array(group, "allele_roles", sites.allele_role)
            _write_array(
                group, "call_source_names", np.asarray(source_names, dtype=object)
            )
            _write_array(
                group, "call_value_names", np.asarray(value_names, dtype=object)
            )
            attributes["n_sorted_runs"] = sorter.n_runs
            attributes["n_repeated_calls"] = repeated_calls
            attributes["n_sites"] = sites.n_sites
            attributes["n_alleles"] = sites.n_alleles
            attributes["n_listed_cells"] = int(listed.sum())
            group.attrs.update(attributes)

    def _variant_group(self, counts: ArtifactRef) -> zarr.Group:
        """Return a complete ``variant_counts`` artifact's group."""
        if not isinstance(counts, ArtifactRef):
            raise TypeError("counts must be an ArtifactRef")
        if counts.kind != VariantCountImportArguments.artifact_kind:
            raise ValueError(
                f"counts must be a variant_counts artifact, not {counts.kind!r}"
            )
        require_complete_artifact(self.zw, counts)
        return artifact_group(self.zw, counts)

    def _variant_cell_index(self, counts: ArtifactRef) -> np.ndarray:
        """The store rows of a variant_counts artifact's cells, in its order."""
        selection = inspect_artifact(self.zw, counts).input_ref("cell_selection")
        return read_stored_selection_indices(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)

    def _variant_cell_positions(
        self, counts: ArtifactRef, cell_index: np.ndarray, cells: Any
    ) -> np.ndarray:
        """The artifact's cells that ``cells`` names, as ascending positions."""
        if cells is None:
            return np.arange(len(cell_index), dtype=np.int64)
        if isinstance(cells, ArtifactRef):
            chosen = read_stored_selection_indices(
                self.zw,
                cells,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)
            positions = np.searchsorted(cell_index, chosen)
            inside = positions < len(cell_index)
            inside[inside] = cell_index[positions[inside]] == chosen[inside]
            if not inside.all():
                raise ValueError(
                    f"cells selects {int((~inside).sum())} cells that the counts do "
                    "not hold; pass a selection within the counts' cell_selection"
                )
            return np.asarray(positions, dtype=np.int64)
        if isinstance(cells, str | bytes) or not isinstance(cells, Sequence):
            raise TypeError(
                "cells must be a cell selection ArtifactRef, a sequence of cell IDs, "
                "or None"
            )
        ids = pd.Index(
            np.asarray(
                read_metadata_rows_chunkwise(self.cells, "ids", cell_index),
                dtype=object,
            )
        )
        wanted = np.asarray(list(cells), dtype=object)
        positions = np.asarray(ids.get_indexer(wanted), dtype=np.int64)
        if np.any(positions < 0):
            missing = wanted[positions < 0][:5]
            raise ValueError(
                "The counts hold no cell "
                + ", ".join(repr(str(cell)) for cell in missing)
            )
        return np.unique(positions)

    def _variant_rows(
        self,
        group: zarr.Group,
        table: str,
        names: Sequence[str],
        positions: np.ndarray,
        max_rows: int,
    ) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        """Yield the rows of ``positions``' cells of a long table in pieces.

        Each piece holds each row's cell position and the arrays ``names``;
        a piece holds at most ``max_rows`` rows, or one cell's.
        """
        offsets = _read(group, f"{table}_offsets")
        if not len(positions):
            return
        # Runs of consecutive cells read as one range.
        breaks = np.flatnonzero(np.diff(positions) != 1) + 1
        for run in np.split(positions, breaks):
            start_cell = int(run[0])
            while start_cell <= int(run[-1]):
                start = int(offsets[start_cell])
                stop_cell = int(
                    np.searchsorted(offsets, start + max_rows, side="right") - 1
                )
                stop_cell = min(int(run[-1]) + 1, max(start_cell + 1, stop_cell))
                stop = int(offsets[stop_cell])
                arrays = {
                    name: np.asarray(as_zarr_array(group[name], name=name)[start:stop])
                    for name in names
                }
                cells_of_rows = np.repeat(
                    np.arange(start_cell, stop_cell, dtype=np.int64),
                    np.diff(offsets[start_cell : stop_cell + 1]),
                )
                yield cells_of_rows, arrays
                start_cell = stop_cell

    @validation_scoped
    def _variants_load_counts(
        self,
        counts: ArtifactRef,
        *,
        sites: Sequence[str | int] | None = None,
        cells: ArtifactRef | Sequence[str] | None = None,
    ) -> VariantCounts:
        """Read saved variant counts as tables of sites, alleles, cells and rows.

        The long tables hold one row per stored entry: a cell's depth at a
        site where it was covered, its count of an allele it showed, and the
        calls its source made. A cell and site without a depth row was not
        covered, so a frequency there is undefined; one with depth but
        without a count row of an allele showed none of it. ``sites`` and
        ``cells`` keep the tables to some sites and cells; ``cells=[]``
        reads only the sites and alleles, with each site's depth summed over
        all cells.

        Args:
            counts: A ``variant_counts`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.import_counts`.
            sites: Site labels as the ``sites`` table names them, such as
                ``"chrM:310"`` or ``"IDH2_Mut1"``, or positions of an
                artifact of one contig, such as ``310``; None for all sites.
            cells: A cell selection within the counts' cells, or cell IDs;
                None for all cells.

        Returns:
            A :class:`~cytearc.variants.VariantCounts` record.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``counts`` is not a complete ``variant_counts``
                artifact, or a site or cell is not in it.
            MemoryError: If the tables do not fit the memory budget; pass
                fewer sites or cells.
        """
        group = self._variant_group(counts)
        table = _stored_sites(group)
        kept_sites = _site_positions(table, sites)
        cell_index = self._variant_cell_index(counts)
        positions = self._variant_cell_positions(counts, cell_index, cells)
        resident = _LOADED_BYTES_PER_CELL * len(positions)
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Loading the counts of {len(positions)} cells holds about {resident} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Pass fewer cells."
            )
        rank = np.full(len(cell_index), -1, dtype=np.int64)
        rank[positions] = np.arange(len(positions))
        site_kept = np.zeros(table.n_sites, dtype=bool)
        site_kept[kept_sites] = True
        allele_kept = site_kept[table.allele_site]
        fields = [str(name) for name in cast(list[str], group.attrs["count_fields"])]
        max_rows = max(1, self.memoryBytes // (4 * _LOADED_BYTES_PER_ROW))
        held = resident

        def collect(
            name: str, arrays: Sequence[str], key: str, kept: np.ndarray
        ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
            nonlocal held
            cells_parts: list[np.ndarray] = []
            parts: dict[str, list[np.ndarray]] = {array: [] for array in arrays}
            # Without a site to keep, no row is read.
            chosen_positions = positions if kept.any() else positions[:0]
            for cells_of_rows, values in self._variant_rows(
                group, name, arrays, chosen_positions, max_rows
            ):
                chosen = kept[values[key]]
                held += _LOADED_BYTES_PER_ROW * int(chosen.sum())
                if held > self.memoryBytes:
                    raise MemoryError(
                        f"Loading the counts holds more than the memory budget of "
                        f"{self.memoryBytes} bytes. Pass fewer sites or cells."
                    )
                cells_parts.append(rank[cells_of_rows[chosen]])
                for array in arrays:
                    parts[array].append(values[array][chosen])
            return (
                np.concatenate(cells_parts) if cells_parts else np.empty(0, np.int64),
                {
                    array: np.concatenate(pieces)
                    if pieces
                    else np.empty(
                        0, dtype=as_zarr_array(group[array], name=array).dtype
                    )
                    for array, pieces in parts.items()
                },
            )

        depth_cells, depth = collect(
            "depth", ("depth_sites", "depth"), "depth_sites", site_kept
        )
        count_arrays = (
            "count_alleles",
            "counts",
            *(f"counts_{name}" for name in fields),
        )
        count_cells, count_values = collect(
            "count", count_arrays, "count_alleles", allele_kept
        )
        call_cells, call_values = collect(
            "call",
            ("call_alleles", "call_sources", "call_values"),
            "call_alleles",
            allele_kept,
        )
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index[positions]),
            dtype=object,
        )
        frames = variant_count_frames(
            sites=table,
            site_depth=_read(group, "site_depth"),
            site_cells=_read(group, "site_cells"),
            kept_sites=kept_sites,
            cell_ids=cell_ids,
            listed=_read(group, "listed")[positions],
            mean_depth=_read(group, "mean_depth")[positions],
            covered_sites=_read(group, "covered_sites")[positions],
            depth=(depth_cells, depth["depth_sites"], depth["depth"]),
            counts=(
                count_cells,
                count_values["count_alleles"],
                {
                    "count": count_values["counts"],
                    **{name: count_values[f"counts_{name}"] for name in fields},
                },
            ),
            calls=(
                call_cells,
                call_values["call_alleles"],
                call_values["call_sources"],
                call_values["call_values"],
            ),
            call_sources=_read(group, "call_source_names").astype(str).tolist(),
            call_values=_read(group, "call_value_names").astype(str).tolist(),
        )
        attributes = group.attrs
        return VariantCounts(
            **frames,
            format=str(attributes["format"]),
            count_unit=str(attributes["count_unit"]),
            depth_definition=str(attributes["depth_definition"]),
        )
