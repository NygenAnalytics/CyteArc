import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import zarr

from ...clones.assignments import CloneAssignment, ranked_clones
from ...metadata.arguments import (
    VariantCloneArguments,
    VariantCountImportArguments,
    VariantSelectionArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact
from ...metadata.rows import read_metadata_rows_chunkwise, text_read_bytes
from ...metadata.selection import cell_selection_indices
from ...storage.arrays import create_metadata_column
from ...storage.artifact_writer import (
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.layout import get_compressors
from ...storage.profiles import resolve_storage_profile
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
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
from ...variants.frequencies import (
    FrequencyClones,
    band_frequencies,
    clone_status_counts,
    frequency_clones,
)
from ...variants.masks import VARIANT_MASKS, masked_variants
from ...variants.sorting import CellRowSorter, sort_rows
from ...variants.statistics import (
    HETEROPLASMY_LEVELS,
    VariantSelection,
    VariantStatisticsAccumulator,
    informative_variants,
    resolve_thresholds,
    variant_selection_frame,
)
from .cell_barcodes import (
    LOOKUP_BYTES_PER_CELL,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    warn_unmatched_barcodes,
)
from .cell_groups import cell_groups
from .clone_artifacts import (
    check_clone_write_budget,
    log_clones,
    plan_clones,
    write_clones,
    zarr_write_buffer_bytes,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _VariantOperationsBase
else:
    _VariantOperationsBase = object

# The most rows of a chunk of a stored long table, which a reader of a band
# of cells decompresses whole; a shorter table is one chunk of its own size.
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
# The smallest budget that parses mgatk's base files on several threads:
# each thread's parser and allocator hold a few megabytes beyond its rows,
# which the shares above absorb only in a budget of this size.
_THREADED_PARSE_BYTES = 1 << 30
# Every stored long array holds 4-byte values. Appending a band rewrites the
# chunks it ends in, read, decoded and encoded again: 2 to 4 chunks' bytes
# at once, as measured on chunks of 1,048,576 rows.
_STORED_BYTES = 4
_REWRITTEN_CHUNKS = 4
# The rows parsed at once lie in [_MIN_CHUNK_ROWS, _MAX_CHUNK_ROWS].
_MIN_CHUNK_ROWS = 1_000
_MAX_CHUNK_ROWS = 2_000_000
# Bytes per row of the loaded tables: codes, values and pandas' overhead.
_LOADED_BYTES_PER_ROW = 120
# Bytes per cell of the loaded cells table and its index.
_LOADED_BYTES_PER_CELL = 200
# Bytes held for each cell of the counts, loaded or not: its store row and
# rank, and each table's offsets and each per-cell array, which are read
# whole one at a time; 32 bytes as traced on 500,000 cells.
_LOADED_BYTES_PER_COUNTS_CELL = 40
# The per-cell arrays of a variant_counts artifact, its canonical one first.
_CELL_ARRAYS = ("mean_depth", "covered_sites", "listed")
_FILES_TEXT = ("the variant file", "the variant files")
# Bytes per stored row of a band that statistics or allele frequencies read:
# the depth and count rows with their cells, and the keys, lookups and
# frequencies derived from them.
_STATISTIC_BYTES_PER_ROW = 200
# Bytes per allele of the per-variant sums that statistics keep.
_STATISTIC_BYTES_PER_VARIANT = 240
# Reading bands holds, for each cell of the counts, chosen or not, each
# table's offsets and their sum and whether the cell is chosen, and while the
# cells are chosen its store row and position and whether a file named it:
# about 40 bytes, as traced on 100,000 and 300,000 cells.
_BAND_BYTES_PER_COUNTS_CELL = 48
# Calling clones holds, for each cell of the counts, its store row, its row
# among the scored cells and its group, while the frequencies are found and
# the graph is partitioned.
_CLONE_BYTES_PER_COUNTS_CELL = 24
# Bytes per cell and informative variant of the allele frequency matrix.
_FREQUENCY_BYTES_PER_ENTRY = 8
# Bytes per cell of a band and informative variant while the band's
# frequencies are found: its depths at their sites, its counts, the depths
# per variant and the frequencies, as float64.
_BAND_FREQUENCY_BYTES = 32
# The Leiden seed of variant clones, CyteArc's default.
_CLONE_SEED = 4444
# Bytes held for each cell while called clones are written: its position,
# group, cluster and status of the call, and its code and clone size.
_CLONE_WRITE_BYTES_PER_CELL = 50
# The smallest Jaccard index of two neighbourhoods that Seurat's shared
# neighbour graph keeps.
_SHARED_NEIGHBOR_PRUNE = 1 / 15
# The attributes that saved variant clones need to be reused.
_CLONE_ATTRIBUTES = (
    "n_clones",
    "n_variants",
    "n_cells_scored",
    "n_cells_clustered",
    "n_cells_without_coverage",
    "n_cells_without_alternative",
    "n_cells_without_neighbors",
    "n_cells_without_group",
    "n_edges",
)


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


@dataclass(frozen=True, slots=True)
class _ImportRules:
    """What every source of one import shares: the unit of its counts, how
    its depth is defined, and the fields that counts carry.

    Attributes:
        field_paths: For each count field, the first source, in canonical
            order, whose counts carry it.
    """

    count_unit: CountUnit
    depth_definition: str
    depth_roles: frozenset[str] | None
    reference_from_depth: bool
    open_alleles: bool
    count_fields: tuple[str, ...]
    field_paths: dict[str, str]


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
    """Append rows to one-dimensional arrays of an artifact in row bands.

    Each array is given its dtype and the most rows it can receive, so a
    short array is one chunk of about its own size rather than a band of
    ``_ROW_BAND`` rows, which every read would decompress whole.
    """

    def __init__(
        self, group: zarr.Group, arrays: Mapping[str, tuple[Any, int]]
    ) -> None:
        compressors = get_compressors(
            resolve_storage_profile(group.store),
            zarrFormat=int(group.metadata.zarr_format),
        )
        self.arrays = {
            name: group.create_array(
                name,
                shape=(0,),
                chunks=(min(_ROW_BAND, max(1, int(most_rows))),),
                dtype=dtype,
                compressors=compressors,
            )
            for name, (dtype, most_rows) in arrays.items()
        }
        self.n_rows = {name: 0 for name in arrays}

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


def _stored_row_bounds(sorter: CellRowSorter) -> tuple[int, int]:
    """The most rows that the stored depth and counts, and the calls, hold.

    A band derives depth from counts and reference counts from depth, so
    neither stored table outnumbers the depth and count rows given.
    """
    entries = int(sorter.rows_per_cell("depth").sum()) + int(
        sorter.rows_per_cell("counts").sum()
    )
    return entries, int(sorter.rows_per_cell("calls").sum())


def _listed(values: Any) -> Any:
    """A NumPy array or a pandas index or series as a list, other values as given."""
    if isinstance(values, np.ndarray | pd.Index | pd.Series):
        return values.tolist()
    return values


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
    sites = _listed(sites)
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


def _band_chunk_bytes(group: zarr.Group, tables: Mapping[str, Sequence[str]]) -> int:
    """Bytes that reading a band decompresses beyond its own rows.

    A read decompresses every chunk it touches, so a band's first and last
    rows bring in the rest of their chunks: at most two chunks of the widest
    array that ``tables`` names.
    """
    widest = 0
    for names in tables.values():
        for name in names:
            array = as_zarr_array(group[name], name=name)
            widest = max(widest, int(array.chunks[0]) * int(array.dtype.itemsize))
    return 2 * widest


def _n_counts_cells(group: zarr.Group) -> int:
    """The cells of a ``variant_counts`` artifact, read or not."""
    return int(as_zarr_array(group["listed"], name="listed").shape[0])


def _variant_bands(
    group: zarr.Group,
    tables: Mapping[str, Sequence[str]],
    positions: np.ndarray,
    max_rows: int,
    *,
    max_cells: int | None = None,
) -> Iterator[tuple[np.ndarray, dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]]]:
    """Yield bands of ``positions``' cells with their rows of long tables.

    ``tables`` names each long table, such as ``"depth"``, with the arrays
    to read from it. A band covers consecutive cells whose rows in every
    table together number at most ``max_rows``, or one cell, and at most
    ``max_cells`` cells, and gives the positions among them and, per table,
    each of their rows' cell position and the arrays, so a band holds every
    row of its chosen cells. Rows of cells between chosen ones are read
    through and dropped while they number fewer than ``max_rows`` and than a
    chunk of the arrays: reading a gap decompresses its chunks, while
    skipping it reads at most one chunk again, so scattered cells take few
    reads and distant ones are read apart.
    """
    if not len(positions):
        return
    positions = np.asarray(positions, dtype=np.int64)
    offsets = {name: _read(group, f"{name}_offsets") for name in tables}
    combined = np.sum(list(offsets.values()), axis=0)
    chosen = np.zeros(len(combined) - 1, dtype=bool)
    chosen[positions] = True
    chunk_rows = min(
        int(as_zarr_array(group[name], name=name).chunks[0])
        for names in tables.values()
        for name in names
    )
    gaps = combined[positions[1:]] - combined[positions[:-1] + 1]
    breaks = np.flatnonzero(gaps >= min(max_rows, chunk_rows)) + 1
    for run in np.split(positions, breaks):
        start_cell = int(run[0])
        last = int(run[-1])
        while start_cell <= last:
            start = int(combined[start_cell])
            stop_cell = int(
                np.searchsorted(combined, start + max_rows, side="right") - 1
            )
            stop_cell = min(last + 1, max(start_cell + 1, stop_cell))
            if max_cells is not None:
                stop_cell = min(stop_cell, start_cell + max_cells)
            cells = np.arange(start_cell, stop_cell, dtype=np.int64)
            # A band of chosen cells only keeps its rows without a copy.
            whole = bool(chosen[start_cell:stop_cell].all())
            band: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
            for table, names in tables.items():
                table_offsets = offsets[table]
                first = int(table_offsets[start_cell])
                stop = int(table_offsets[stop_cell])
                cells_of_rows = np.repeat(
                    cells, np.diff(table_offsets[start_cell : stop_cell + 1])
                )
                kept = slice(None) if whole else chosen[cells_of_rows]
                band[table] = (
                    cells_of_rows[kept],
                    {
                        name: np.asarray(
                            as_zarr_array(group[name], name=name)[first:stop]
                        )[kept]
                        for name in names
                    },
                )
            yield cells if whole else cells[chosen[cells]], band
            start_cell = stop_cell
            # The next band starts at the next chosen cell.
            following = np.searchsorted(run, start_cell)
            if following < len(run):
                start_cell = int(run[following])


class _VariantOperationsMixin(_VariantOperationsBase):
    def _variant_selection_indices(self, cell_selection: ArtifactRef) -> np.ndarray:
        cell_index = cell_selection_indices(self.zw, cell_selection)
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
        another length or sequence raises ``ValueError``, as do one sequence
        under two contig names and a cell that two files count at the same
        site. The paths must also count in one unit, define depth alike and
        give their counts the same fields, such as base qualities; a table of
        calls alone, such as GoT-Multi's refined table, gives no counts.

        The canonical ``mean_depth`` holds each cell's depth summed over the
        sites and divided by their number, NaN for a selected cell that no
        file names; ``covered_sites`` counts its sites with depth above 0 and
        ``listed`` marks the cells a file names. ``ds.select_cells(counts,
        low=10)`` keeps cells with a mean depth above 10.

        Rows that outgrow the memory budget are sorted in runs on local
        disk, in the system's temporary folder (``TMPDIR`` on Linux and
        macOS), and removed when the import ends.

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
                not state, since it depends on how they were run, and what a
                MiTo matrix counted when its ``uns`` does not say. Other
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
                references, count in different units, define depth
                differently or give their counts different fields, a cell is
                counted twice at a site, an option lists a value for another
                number of paths, or the selection holds no cell.
            FileNotFoundError: If a path or a file it needs is missing.
            PermissionError: If the store is read-only and the counts are not
                saved yet.
            MemoryError: If the import does not fit the memory budget.
        """
        from ...readers._variant_files import VARIANT_FORMATS, source_sha256

        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        bool_argument(swap_strands, "swap_strands")
        bool_argument(invalidate_cache, "invalidate_cache")
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
        threads = self.nthreads if self.memoryBytes >= _THREADED_PARSE_BYTES else 1

        def identify(path: str | os.PathLike[str]) -> str:
            key = os.fspath(path)
            if key not in readers:
                reader = _read_source(key, format, swap_strands, threads)
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
        rules = self._variant_import_rules(sources, count_unit)
        site_table, site_maps, allele_maps = merge_sites(
            [source.reader.sites() for source in sources],
            open_alleles=rules.open_alleles,
        )
        cell_index = self._variant_selection_indices(cell_selection)
        n_cells = len(cell_index)
        arguments = VariantCountImportArguments(
            cell_selection=cell_selection,
            sources=tuple(source.record() for source in sources),
            count_unit=rules.count_unit,
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
        # What the import holds for the selected cells, and the read of their
        # IDs as text, which decodes a stored chunk of them, are checked before
        # the IDs are read.
        reading = (
            LOOKUP_BYTES_PER_CELL + _IMPORT_BYTES_PER_CELL
        ) * n_cells + text_read_bytes(self.cells._get_array("ids"), n_cells)
        if reading > self.memoryBytes:
            raise MemoryError(
                f"Reading the IDs of {n_cells} selected cells holds about {reading} "
                f"bytes, more than the memory budget of {self.memoryBytes} bytes. "
                "Raise mem_budget, or pass a smaller cell_selection."
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
        # The parsed chunks, the sort buffer and the bands share what the join
        # leaves: it holds its lookup, the cells' IDs as text and the bytes
        # held for each cell.
        held = join.held_bytes
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
        counts_dtype = count_dtype(rules.count_fields)
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
                        path=source.reader.path,
                        field_paths=rules.field_paths,
                    )
            if not n_rows:
                raise ValueError(
                    f"No row of counts, depth, calls or cells is in {files_text}"
                )
            # A band's rows sit beside the chunks that appending them rewrites.
            n_entry_rows, _n_call_rows = _stored_row_bounds(sorter)
            rewritten = (
                _REWRITTEN_CHUNKS * min(_ROW_BAND, max(1, n_entry_rows)) * _STORED_BYTES
            )
            band_bytes = int(room / _BAND_SHARE) - rewritten
            if band_bytes < _BAND_COPIES * counts_dtype.itemsize:
                raise MemoryError(
                    f"Writing the variant counts rewrites stored chunks of about "
                    f"{rewritten} bytes, more than the memory budget of "
                    f"{self.memoryBytes} bytes leaves beside the sorted rows. Raise "
                    "mem_budget."
                )
            band_rows = band_bytes // (_BAND_COPIES * counts_dtype.itemsize)
            attributes = {
                **join.attributes(),
                "format": formats[0],
                "count_unit": rules.count_unit,
                "depth_definition": rules.depth_definition,
                "stranded": "forward" in rules.count_fields,
                "count_fields": list(rules.count_fields),
                "contigs": [contig.record() for contig in site_table.contigs],
                "n_sorted_runs": 0,
            }
            self._write_variant_counts(
                planned,
                sorter,
                sites=site_table,
                rules=rules,
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
                    f"{reader.path} does not say whether {reader.format} counted "
                    "reads, UMIs or molecules, which depends on how it ran; pass "
                    "count_unit='reads', count_unit='umis' or count_unit='molecules'"
                )
            return cast(CountUnit, count_unit)
        if count_unit is not None and count_unit != reader.count_unit:
            raise ValueError(
                f"{reader.format} counts {reader.count_unit}, not {count_unit}; "
                "count_unit names the unit of formats whose files do not state it"
            )
        return cast(CountUnit, reader.count_unit)

    @classmethod
    def _variant_import_rules(
        cls, sources: Sequence[_Source], count_unit: str | None
    ) -> _ImportRules:
        """The unit, depth rules and count fields that every source shares.

        Each source's unit and depth rules must agree with the others', so
        the artifact records them once whatever order the sources take. The
        count fields are those of all sources together: a table of calls
        alone gives none, and counts that lack a field another source gives
        are refused as they arrive.

        Raises:
            ValueError: If a source's unit is missing or differs from the
                given one, or two sources count in different units or define
                depth differently.
        """
        first = sources[0].reader
        unit = cls._variant_count_unit(first, count_unit)
        depth_rules = (
            first.depth_definition,
            first.depth_roles,
            first.reference_from_depth,
            first.open_alleles,
        )
        field_paths: dict[str, str] = {}
        for source in sources:
            reader = source.reader
            other = cls._variant_count_unit(reader, count_unit)
            if other != unit:
                raise ValueError(
                    f"The files count in different units: {first.path} counts {unit} "
                    f"and {reader.path} {other}; import them separately"
                )
            if (
                reader.depth_definition,
                reader.depth_roles,
                reader.reference_from_depth,
                reader.open_alleles,
            ) != depth_rules:
                raise ValueError(
                    f"The files define depth differently: {first.path} counts "
                    f"{first.depth_definition!r} and {reader.path} "
                    f"{reader.depth_definition!r}, as cellsnp-lite folders with "
                    "and without OTH do; import them separately"
                )
            for name in reader.count_fields:
                field_paths.setdefault(name, reader.path)
        return _ImportRules(
            count_unit=unit,
            depth_definition=first.depth_definition,
            depth_roles=first.depth_roles,
            reference_from_depth=first.reference_from_depth,
            open_alleles=first.open_alleles,
            count_fields=tuple(name for name in COUNT_FIELDS if name in field_paths),
            field_paths=field_paths,
        )

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
        path: str,
        field_paths: Mapping[str, str],
    ) -> None:
        """Join a batch's rows to cells and add those of selected cells.

        Raises:
            ValueError: If counts lack a field that another source's counts
                carry, or a value lies outside the stored range.
        """
        if batch.kind == "counts":
            missing = [
                name
                for name in counts_dtype.names or ()
                if name in COUNT_FIELDS and name not in batch.fields
            ]
            if missing:
                raise ValueError(
                    "The files give their counts different fields: "
                    f"{field_paths[missing[0]]} gives {missing[0]} and {path} does "
                    "not; import them separately"
                )
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
        rules: _ImportRules,
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

        n_entry_rows, n_call_rows = _stored_row_bounds(sorter)
        with artifact_transaction(self.zw, planned) as group:
            appender = _Appender(
                group,
                {
                    "depth_sites": (np.int32, n_entry_rows),
                    "depth": (np.uint32, n_entry_rows),
                    "count_alleles": (np.int32, n_entry_rows),
                    "counts": (np.uint32, n_entry_rows),
                    **{
                        f"counts_{name}": (COUNT_FIELDS[name], n_entry_rows)
                        for name in fields
                    },
                    "call_alleles": (np.int32, n_call_rows),
                    "call_sources": (np.int32, n_call_rows),
                    "call_values": (np.int32, n_call_rows),
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
                # The rows came sorted by the order in which sources first
                # appeared; stored, they follow the sorted source names.
                call_rows = call_rows.copy()
                call_rows["source"] = source_remap[call_rows["source"]]
                call_rows = sort_rows(call_rows, ("cell", "key", "source"))
                depth_out, counts_out = finish_band(
                    counts_rows,
                    depth_rows,
                    sites=sites,
                    depth_roles=rules.depth_roles,
                    reference_from_depth=rules.reference_from_depth,
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
                appender.append("call_sources", call_rows["source"])
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
        return cell_selection_indices(self.zw, selection)

    def _variant_cell_positions(
        self, counts: ArtifactRef, cell_index: np.ndarray, cells: Any
    ) -> np.ndarray:
        """The artifact's cells that ``cells`` names, as ascending positions."""
        if cells is None:
            return np.arange(len(cell_index), dtype=np.int64)
        if isinstance(cells, ArtifactRef):
            chosen = cell_selection_indices(self.zw, cells)
            positions = np.searchsorted(cell_index, chosen)
            inside = positions < len(cell_index)
            inside[inside] = cell_index[positions[inside]] == chosen[inside]
            if not inside.all():
                raise ValueError(
                    f"cells selects {int((~inside).sum())} cells that the counts do "
                    "not hold; pass a selection within the counts' cell_selection"
                )
            return np.asarray(positions, dtype=np.int64)
        cells = _listed(cells)
        if isinstance(cells, str | bytes) or not isinstance(cells, Sequence):
            raise TypeError(
                "cells must be a cell selection ArtifactRef, a sequence of cell IDs, "
                "or None"
            )
        wanted = np.asarray(list(cells), dtype=object)
        if not len(wanted):
            return np.empty(0, dtype=np.int64)
        # Cells named by ID are found among the IDs of every cell of the
        # counts, read as text into a lookup.
        lookup = (
            text_read_bytes(self.cells._get_array("ids"), len(cell_index))
            + LOOKUP_BYTES_PER_CELL * len(cell_index)
            + sum(map(sys.getsizeof, wanted.tolist()))
        )
        if lookup > self.memoryBytes:
            raise MemoryError(
                f"Finding {len(wanted)} cells among the {len(cell_index)} cells of "
                f"the counts holds about {lookup} bytes, more than the memory budget "
                f"of {self.memoryBytes} bytes. Raise mem_budget, or pass a cell "
                "selection."
            )
        ids = pd.Index(
            np.asarray(
                read_metadata_rows_chunkwise(self.cells, "ids", cell_index),
                dtype=object,
            )
        )
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
        for _cells, tables in _variant_bands(
            group, {table: names}, positions, max_rows
        ):
            yield tables[table]

    @validation_scoped
    def _variants_load_counts(
        self,
        counts: ArtifactRef,
        *,
        sites: Sequence[str | int] | np.ndarray | pd.Index | None = None,
        cells: ArtifactRef | Sequence[str] | np.ndarray | pd.Index | None = None,
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

        The budget counts each row read, at about 120 bytes, the two stored
        chunks that a read's first and last rows decompress whole, the cells'
        IDs as text, whose read decodes a stored chunk of them, and 40 bytes
        for each cell of the counts, whose offsets and per-cell arrays are
        read whole; cells named by ID are found among the IDs of every cell
        of the counts.

        Args:
            counts: A ``variant_counts`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.import_counts`.
            sites: Site labels as the ``sites`` table names them, such as
                ``"chrM:310"`` or ``"IDH2_Mut1"``, or positions of an
                artifact of one contig, such as ``310``, as a list, a NumPy
                array or a pandas index; None for all sites.
            cells: A cell selection within the counts' cells, or cell IDs as
                a list, a NumPy array or a pandas index; None for all cells.

        Returns:
            A :class:`~cytearc.variants.VariantCounts` record.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``counts`` is not a ``variant_counts`` artifact, or
                a site or cell is not in it.
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as an import that
                stopped before it finished leaves it.
            MemoryError: If the tables do not fit the memory budget; pass
                fewer sites or cells.
        """
        group = self._variant_group(counts)
        table = _stored_sites(group)
        kept_sites = _site_positions(table, sites)
        cell_index = self._variant_cell_index(counts)
        positions = self._variant_cell_positions(counts, cell_index, cells)
        fields = [str(name) for name in cast(list[str], group.attrs["count_fields"])]
        count_arrays = (
            "count_alleles",
            "counts",
            *(f"counts_{name}" for name in fields),
        )
        resident = _LOADED_BYTES_PER_COUNTS_CELL * len(
            cell_index
        ) + _LOADED_BYTES_PER_CELL * len(positions)
        if len(positions):
            # The cells' IDs are read as text, a stored chunk at a time.
            resident += text_read_bytes(self.cells._get_array("ids"), len(positions))
        if len(positions) and len(kept_sites):
            # Reads decompress whole chunks, as the bands of the statistics do.
            resident += _band_chunk_bytes(
                group,
                {
                    "depth": ("depth_sites", "depth"),
                    "count": count_arrays,
                    "call": ("call_alleles", "call_sources", "call_values"),
                },
            )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Loading the counts of {len(positions)} cells holds about {resident} "
                "bytes before it keeps a row, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget, or pass fewer cells."
            )
        rank = np.full(len(cell_index), -1, dtype=np.int64)
        rank[positions] = np.arange(len(positions))
        site_kept = np.zeros(table.n_sites, dtype=bool)
        site_kept[kept_sites] = True
        allele_kept = site_kept[table.allele_site]
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

    def _selection_group(self, informative: ArtifactRef) -> zarr.Group:
        """Return a complete ``variant_selection`` artifact's group."""
        if not isinstance(informative, ArtifactRef):
            raise TypeError("variants must be an ArtifactRef")
        if informative.kind != VariantSelectionArguments.artifact_kind:
            raise ValueError(
                "variants must be a variant_selection artifact from "
                f"ds.variants.informative, not {informative.kind!r}"
            )
        require_complete_artifact(self.zw, informative)
        return artifact_group(self.zw, informative)

    def _statistic_positions(
        self, counts: ArtifactRef, group: zarr.Group, cells: ArtifactRef | None
    ) -> np.ndarray:
        """The positions of the cells that statistics cover: the chosen cells
        that a file names."""
        cell_index = self._variant_cell_index(counts)
        positions = self._variant_cell_positions(counts, cell_index, cells)
        listed = _read(group, "listed").astype(bool)
        positions = np.asarray(positions[listed[positions]], dtype=np.int64)
        if not len(positions):
            raise ValueError(
                "No chosen cell is named by the counts' files, so no variant can be "
                "scored; check the cells and the import's unmatched barcodes"
            )
        return positions

    @validation_scoped
    def _variants_informative(
        self,
        counts: ArtifactRef,
        *,
        cells: ArtifactRef | None = None,
        preset: str | None = None,
        min_cells: int | None = None,
        min_strand_correlation: float | None = None,
        min_vmr: float | None = None,
        min_depth: float | None = None,
        min_quality: float | None = None,
        mask: str | Sequence[str] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Score every variant across cells and select the informative ones.

        A variant is an alternative allele at a site, such as ``310T>C``; the
        variants scored are those that some chosen cell shows. Each variant
        gets mgatk's per-variant statistics, computed over the chosen cells
        in bands of cells, with an allele frequency only where a cell is
        covered: the pooled ``mean`` frequency, the ``variance`` of the
        covered cells' frequencies and their ratio ``vmr``, the cells with 2
        or more reads of the allele on each strand
        (``n_cells_conf_detected``), the Pearson correlation of the forward
        and reverse counts over the cells that show the allele
        (``strand_correlation``), the cells whose frequency reaches 5, 10,
        20 and 95%, the highest frequency, the ``mean_coverage`` at the site
        and MAESTER's mean base quality of the allele (``mean_quality``).

        A variant is informative when it passes every threshold that is set
        and no mask lists it. ``preset="atac"`` sets Signac's thresholds for
        mtscATAC-seq: at least 5 confidently detected cells, a strand
        correlation of at least 0.65 and a VMR above 0.01. ``preset="rna"``
        sets MAESTER's for RNA libraries, whose strands follow the
        transcript: a mean depth above 20 and a mean base quality above 30,
        with a VMR above 0.01. A threshold given by name replaces the
        preset's. VMR, depth and quality must exceed their thresholds; the
        cell count and strand correlation must reach theirs, as the
        published rules compare them, and a statistic that is undefined,
        such as the strand correlation of a variant that one cell shows,
        fails.

        Signac computes the same statistics but sets the frequency of an
        uncovered cell to 0 and, by default, replaces the frequency of a cell
        with depth below 10 by the pooled mean before taking the variance;
        here an uncovered cell has no frequency, as in mgatk, so variances
        and VMRs differ where coverage is sparse.

        Args:
            counts: A ``variant_counts`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.import_counts`.
            cells: A cell selection within the counts' cells whose evidence
                the statistics use, such as ``ds.select_cells(counts,
                low=10)`` for the cells whose mean depth exceeds 10, as mgatk
                scores them; None uses every cell that the counts' files
                name. Cells that no file names are left out either way.
            preset: ``"atac"``, ``"rna"`` or None.
            min_cells: The fewest cells with 2 or more reads of the allele on
                each strand.
            min_strand_correlation: The lowest strand correlation.
            min_vmr: The VMR that a variant must exceed.
            min_depth: The mean depth at the site that a variant must exceed.
            min_quality: The mean base quality that a variant must exceed.
            mask: Names of lists of variants that artifacts mimic, which are
                never informative: ``"numt"``, the 25 false positives of two
                common NUMTs (Laricchia et al. 2022); ``"rna_editing"``, the
                RNA-DNA differences at 295, 2617 and 13710 (Bar-Yaacov et al.
                2013); and ``"artifact_prone"``, every variant at 301, 302,
                310, 316, 3107 and 16182 (Laricchia et al. 2022). Masks list
                rCRS positions, so they need a 16,569-base contig.
            invalidate_cache: Score again instead of reusing saved statistics.

        Returns:
            A complete datastore-scoped ``variant_selection`` artifact. Read
            it with
            :meth:`~cytearc.datastore.namespaces.VariantsAccessor.load_informative`,
            plot it with ``ds.plots.variant_statistics`` or call clones from
            it with :meth:`~cytearc.datastore.namespaces.VariantsAccessor.clones`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``counts`` is not a ``variant_counts`` artifact,
                the preset, a threshold or a mask is unknown or invalid, no
                threshold is set, a threshold tests a statistic that the
                counts cannot give (strand tests without strand counts,
                quality without base qualities), a mask meets a contig that
                is not rCRS-sized, or no chosen cell is named by the counts'
                files.
            KeyError: If the store holds no such ``counts`` artifact.
            RuntimeError: If the ``counts`` artifact is incomplete, as an
                import that stopped before it finished leaves it.
            PermissionError: If the store is read-only and the statistics are
                not saved yet.
            MemoryError: If the statistics do not fit the memory budget.
        """
        group = self._variant_group(counts)
        if cells is not None and not isinstance(cells, ArtifactRef):
            raise TypeError("cells must be a cell selection ArtifactRef or None")
        bool_argument(invalidate_cache, "invalidate_cache")
        thresholds = resolve_thresholds(
            preset,
            {
                "min_cells": min_cells,
                "min_strand_correlation": min_strand_correlation,
                "min_vmr": min_vmr,
                "min_depth": min_depth,
                "min_quality": min_quality,
            },
        )
        masks = _mask_names(mask)
        fields = [str(name) for name in cast(list[str], group.attrs["count_fields"])]
        stranded = "forward" in fields
        qualities = "quality_forward" in fields
        # A test that the counts cannot make fails before any row is read.
        available = {"n_cells_conf_detected": stranded, "strand_correlation": stranded}
        available["mean_quality"] = qualities
        informative_variants(
            {
                column: np.empty(0)
                for column in (
                    "n_cells_conf_detected",
                    "strand_correlation",
                    "vmr",
                    "mean_coverage",
                    "mean_quality",
                )
                if available.get(column, True)
            },
            thresholds,
            n_variants=0,
        )
        sites = _stored_sites(group)
        positions = self._statistic_positions(counts, group, cells)
        arguments = VariantSelectionArguments(
            counts=counts,
            cells=cells,
            preset=cast(Any, preset),
            min_cells=thresholds["min_cells"],
            min_strand_correlation=thresholds["min_strand_correlation"],
            min_vmr=thresholds["min_vmr"],
            min_depth=thresholds["min_depth"],
            min_quality=thresholds["min_quality"],
            mask=masks,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_variant_selection(arguments)
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        count_arrays = ["count_alleles", "counts"]
        if stranded:
            count_arrays += ["counts_forward", "counts_reverse"]
        if qualities:
            count_arrays += ["counts_quality_forward", "counts_quality_reverse"]
        tables = {"depth": ("depth_sites", "depth"), "count": tuple(count_arrays)}
        # The statistics are written beside what Zarr's writes hold in file
        # buffers.
        held = (
            _STATISTIC_BYTES_PER_VARIANT * sites.n_alleles
            + 16 * sites.n_sites
            + _band_chunk_bytes(group, tables)
            + _BAND_BYTES_PER_COUNTS_CELL * _n_counts_cells(group)
            + zarr_write_buffer_bytes()
        )
        max_rows = (self.memoryBytes - held) // _STATISTIC_BYTES_PER_ROW
        if max_rows < 1:
            raise MemoryError(
                f"Scoring the variants of {sites.n_alleles} alleles holds about "
                f"{held} bytes, more than the memory budget of {self.memoryBytes} "
                "bytes. Raise mem_budget."
            )
        accumulator = VariantStatisticsAccumulator(
            sites, stranded=stranded, qualities=qualities
        )
        for _band, rows in _variant_bands(group, tables, positions, int(max_rows)):
            depth_cells, depth = rows["depth"]
            count_cells, values = rows["count"]
            accumulator.add(
                depth_cells=depth_cells,
                depth_sites=depth["depth_sites"],
                depth=depth["depth"],
                count_cells=count_cells,
                count_alleles=values["count_alleles"],
                counts=values["counts"],
                forward=values.get("counts_forward"),
                reverse=values.get("counts_reverse"),
                quality_forward=values.get("counts_quality_forward"),
                quality_reverse=values.get("counts_quality_reverse"),
            )
        statistics = accumulator.finish(len(positions))
        variant_sites = sites.allele_site[statistics.alleles]
        contig_lengths = np.asarray(
            [
                -1
                if code < 0 or sites.contigs[code].length is None
                else int(cast(int, sites.contigs[code].length))
                for code in sites.site_contig[variant_sites].tolist()
            ],
            dtype=np.int64,
        )
        masked = masked_variants(
            masks,
            positions=sites.site_position[variant_sites],
            alternatives=sites.allele[statistics.alleles],
            contig_lengths=contig_lengths,
        )
        columns = statistics.arrays()
        informative = (
            informative_variants(columns, thresholds, n_variants=statistics.n_variants)
            & ~masked
        )
        attributes = {
            "preset": preset,
            "thresholds": thresholds,
            "masks": list(masks),
            "n_cells": statistics.n_cells,
            "n_variants": statistics.n_variants,
            "n_informative": int(informative.sum()),
            "n_masked": int(masked.sum()),
            "stranded": stranded,
            "qualities": qualities,
        }
        with artifact_transaction(self.zw, planned) as output:
            _write_array(output, "variant_alleles", statistics.alleles.astype(np.int32))
            _write_array(output, "informative", informative)
            _write_array(output, "masked", masked)
            for name, column in columns.items():
                _write_array(output, name, np.asarray(column))
            output.attrs.update(attributes)
        logger.info(
            f"Scored {statistics.n_variants} variants in {statistics.n_cells} cells; "
            f"{int(informative.sum())} are informative"
        )
        return planned.ref

    def _plan_variant_selection(
        self, arguments: VariantSelectionArguments
    ) -> PlannedArtifact:
        record = arguments.to_record()

        def has_statistics(_ref: ArtifactRef, group: zarr.Group) -> bool:
            names = set(group.array_keys())
            required = {
                "variant_alleles",
                "informative",
                "masked",
                "n_cells_covered",
                "mean",
                "variance",
                "vmr",
                "max_heteroplasmy",
                "mean_coverage",
                *HETEROPLASMY_LEVELS,
            }
            # Counts with strands or qualities give their statistics too.
            if group.attrs.get("stranded"):
                required |= {"n_cells_conf_detected", "strand_correlation"}
            if group.attrs.get("qualities"):
                required.add("mean_quality")
            if not required <= names:
                return False
            lengths = {as_zarr_array(group[name], name=name).shape for name in required}
            return len(lengths) == 1

        return plan_artifact(
            self.zw,
            scope="datastore",
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=arguments.invalidate_cache,
            required_attributes=(
                "thresholds",
                "masks",
                "n_cells",
                "n_variants",
                "stranded",
                "qualities",
            ),
            reuse_validator=has_statistics,
        )

    @validation_scoped
    def _variants_load_informative(self, informative: ArtifactRef) -> VariantSelection:
        """Read saved per-variant statistics and the informative variants.

        Args:
            informative: A ``variant_selection`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.informative`.

        Returns:
            A :class:`~cytearc.variants.VariantSelection` record whose
            ``variants`` table holds one row per scored variant, with mgatk's
            ``variant_stats`` columns and the ``informative`` flag.

        Raises:
            TypeError: If ``informative`` is not an ArtifactRef.
            ValueError: If it is not a ``variant_selection`` artifact.
            KeyError: If the store holds no such artifact, or not the counts
                that it scored.
            RuntimeError: If the artifact or the counts that it scored are
                incomplete.
        """
        group = self._selection_group(informative)
        counts = inspect_artifact(self.zw, informative).input_ref("counts")
        sites = _stored_sites(self._variant_group(counts))
        alleles = _read(group, "variant_alleles").astype(np.int64)
        attributes = group.attrs
        frame = variant_selection_frame(
            sites,
            alleles,
            {name: _read(group, name) for name in group.array_keys()},
        )
        thresholds = cast(dict[str, float | None], attributes["thresholds"])
        status = inspect_artifact(self.zw, informative)
        return VariantSelection(
            variants=frame,
            counts=counts,
            cells=None
            if (status.inputs or {}).get("cells") is None
            else status.input_ref("cells"),
            preset=cast(str | None, attributes["preset"]),
            thresholds=dict(thresholds),
            masks=tuple(str(name) for name in cast(list[str], attributes["masks"])),
            n_cells=int(cast(int, attributes["n_cells"])),
            stranded=bool(attributes["stranded"]),
            qualities=bool(attributes["qualities"]),
        )

    @validation_scoped
    def _variants_clones(
        self,
        counts: ArtifactRef,
        *,
        variants: ArtifactRef,
        n_neighbors: int = 10,
        resolution: float = 1.0,
        within: str | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call clones from the allele frequencies of informative variants.

        Cells that descend from one ancestor share its mitochondrial
        variants, at similar frequencies. As Signac's ``FindClonotypes``
        does, each cell's neighbourhood holds itself and the
        ``n_neighbors - 1`` cells most similar to it by the cosine similarity
        of the square roots of their allele frequencies of the informative
        variants; two cells that share a neighbour are joined with the
        Jaccard index of their neighbourhoods, dropping joins below 1/15, as
        Seurat's ``FindNeighbors`` builds its shared-neighbour graph; and the
        Leiden algorithm partitions the graph at ``resolution`` where Signac
        uses Seurat's ``FindClusters``.

        Every cell is compared with every other cell, so the time grows with
        the square of the cells; ``within`` compares the cells of each group
        apart. Cells that tie, such as cells that show one informative
        variant each, take the same few neighbours and join every other such
        cell, so the graph also grows with the square of their number, and a
        graph that does not fit the memory budget raises ``MemoryError``.

        A cell's frequency is undefined where it was not covered, so the
        similarity of two cells counts only the variants that both cover,
        where Signac counts an uncovered variant as frequency 0. Cells with
        no coverage at any informative variant, cells that show none of
        them and cells similar to no other cell carry no evidence of a clone
        and get none; the artifact counts them in its
        ``n_cells_without_coverage``, ``n_cells_without_alternative`` and
        ``n_cells_without_neighbors`` attributes. The clones cover the cells
        that the statistics of ``variants`` covered.

        Args:
            counts: The ``variant_counts`` artifact that ``variants`` scored.
            variants: A ``variant_selection`` artifact from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.informative`.
            n_neighbors: The cells of a full neighbourhood, the cell itself
                included as Seurat counts it, at least 2.
            resolution: The Leiden resolution; larger values give more,
                smaller clones.
            within: A cell metadata column, such as the donor or the mouse:
                clones never join cells of two groups, and each clone's label
                starts with its group. Cells without a group get no clone.
            invalidate_cache: Call clones again instead of reusing them.

        Returns:
            A complete datastore-scoped ``clones`` artifact over the counts'
            cells, which every clone summary reads. Read it with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_clones`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``counts`` is not a ``variant_counts`` artifact,
                ``variants`` is not a ``variant_selection`` artifact or did
                not score ``counts``, it selects no informative variant,
                ``n_neighbors`` is below 2, the resolution is not finite and
                positive, or ``within`` is not a cell metadata column.
            KeyError: If the store holds no such ``counts`` or ``variants``
                artifact.
            RuntimeError: If the ``counts`` or ``variants`` artifact is
                incomplete.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If grouping the cells by ``within``, the frequencies,
                similarities or graph, or the clone labels as the store saves
                them do not fit the memory budget.
        """
        from ...clustering.leiden import canonical_resolution

        group = self._variant_group(counts)
        selection = self._selection_group(variants)
        status = inspect_artifact(self.zw, variants)
        if status.input_ref("counts") != counts:
            raise ValueError(
                "variants scored other counts; pass the counts that "
                "ds.variants.informative read"
            )
        if isinstance(n_neighbors, bool) or not isinstance(
            n_neighbors, int | np.integer
        ):
            raise TypeError("n_neighbors must be an integer")
        if int(n_neighbors) < 2:
            raise ValueError(
                "n_neighbors must be at least 2: a neighbourhood holds the cell "
                "itself and its most similar cells"
            )
        resolution = canonical_resolution(resolution)
        if within is not None and not isinstance(within, str):
            raise TypeError("within must be a cell metadata column or None")
        if within is not None and within not in self.cells.columns:
            raise ValueError(f"{within!r} is not a cell metadata column")
        bool_argument(invalidate_cache, "invalidate_cache")
        chosen = _read(selection, "informative").astype(bool)
        alleles = _read(selection, "variant_alleles").astype(np.int64)[chosen]
        if not len(alleles):
            raise ValueError(
                "variants selects no informative variant, so no clone can be called; "
                "relax its thresholds"
            )
        cell_selection = inspect_artifact(self.zw, counts).input_ref("cell_selection")
        cell_index = self._variant_cell_index(counts)
        n_cells = len(cell_index)
        # The statistics covered the chosen cells, or every listed cell.
        chosen_cells = (
            None
            if (status.inputs or {}).get("cells") is None
            else status.input_ref("cells")
        )
        positions = self._statistic_positions(counts, group, chosen_cells)
        group_codes: np.ndarray | None = None
        group_names: list[str] = []
        fingerprint = None
        if within is not None:
            # The groups are those that the clone summaries group cells into,
            # read and coded within the memory budget, and are identified by
            # each cell's group and the groups' labels.
            grouped = cell_groups(
                self.zw,
                self.cells,
                cell_selection,
                np.ones(n_cells, dtype=bool),
                group_by=within,
                groups=None,
                cell_selection=None,
                memory_bytes=self.memoryBytes,
            )
            group_codes = grouped.codes
            group_names = [key[within] for key in grouped.keys]
            fingerprint = grouped.fingerprint()
        arguments = VariantCloneArguments(
            cell_selection=cell_selection,
            counts=counts,
            variants=variants,
            within_fingerprint=fingerprint,
            n_neighbors=int(n_neighbors),
            resolution=resolution,
            within=within,
            random_seed=_CLONE_SEED,
            shared_neighbor_prune=_SHARED_NEIGHBOR_PRUNE,
            invalidate_cache=invalidate_cache,
        )
        planned = plan_clones(
            self.zw,
            arguments,
            cell_selection,
            n_cells,
            with_probability=False,
            required_attributes=_CLONE_ATTRIBUTES,
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        sites = _stored_sites(group)
        matrix_bytes = _FREQUENCY_BYTES_PER_ENTRY * len(positions) * len(alleles)
        tables = {
            "depth": ("depth_sites", "depth"),
            "count": ("count_alleles", "counts"),
        }
        # Beside the frequencies, every cell of the counts holds its rows and
        # group, and while bands are read its offsets.
        cell_bytes = _CLONE_BYTES_PER_COUNTS_CELL * n_cells
        held = (
            matrix_bytes
            + cell_bytes
            + _BAND_BYTES_PER_COUNTS_CELL * n_cells
            + _band_chunk_bytes(group, tables)
        )
        # A band holds at most as many cells as rows, each with its dense
        # depths, counts and frequencies at the informative variants.
        per_row = _STATISTIC_BYTES_PER_ROW + _BAND_FREQUENCY_BYTES * len(alleles)
        max_rows = (self.memoryBytes - held) // per_row
        if max_rows < 1:
            raise MemoryError(
                f"The allele frequencies of {len(positions)} cells at "
                f"{len(alleles)} variants take about {held + per_row} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget, or select fewer informative variants."
            )
        frequencies = np.empty((len(positions), len(alleles)), dtype=np.float64)
        row_of_position = np.full(n_cells, -1, dtype=np.int64)
        row_of_position[positions] = np.arange(len(positions))
        for band, rows in _variant_bands(
            group, tables, positions, int(max_rows), max_cells=int(max_rows)
        ):
            depth_cells, depth = rows["depth"]
            count_cells, values = rows["count"]
            # A band's cells need not be consecutive.
            frequencies[row_of_position[band]] = band_frequencies(
                n_rows=len(band),
                depth_rows=np.searchsorted(band, depth_cells),
                depth_sites=depth["depth_sites"],
                depth=depth["depth"],
                count_rows=np.searchsorted(band, count_cells),
                count_alleles=values["count_alleles"],
                counts=values["counts"],
                alleles=alleles,
                sites=sites,
            )
        called = frequency_clones(
            frequencies,
            groups=None if group_codes is None else group_codes[positions],
            n_neighbors=int(n_neighbors),
            resolution=resolution,
            random_seed=_CLONE_SEED,
            memory_bytes=self.memoryBytes - matrix_bytes - cell_bytes,
            nthreads=self.nthreads,
        )
        del frequencies
        assignment = _clone_assignment(
            n_cells, positions, called, group_names if within is not None else None
        )
        counts_by_status = clone_status_counts(called.status)
        attributes = {
            "n_clones": len(assignment.clone_ids),
            "n_variants": len(alleles),
            "n_cells_scored": len(positions),
            "n_cells_clustered": counts_by_status["clustered"],
            "n_cells_without_coverage": counts_by_status["no_coverage"],
            "n_cells_without_alternative": counts_by_status["no_alternative"],
            "n_cells_without_neighbors": counts_by_status["no_neighbor"],
            "n_cells_without_group": counts_by_status["no_group"],
            "n_edges": called.n_edges,
        }
        check_clone_write_budget(
            self.memoryBytes,
            _CLONE_WRITE_BYTES_PER_CELL * n_cells
            + sum(map(sys.getsizeof, assignment.clone_ids.tolist())),
            assignment.clone_ids,
            n_cells,
        )
        write_clones(self.zw, planned, assignment, attributes)
        log_clones(assignment, n_cells)
        return planned.ref


def _mask_names(mask: str | Sequence[str] | None) -> tuple[str, ...]:
    """The masks that ``mask`` names, sorted and once each."""
    if mask is None:
        return ()
    names = [mask] if isinstance(mask, str) else mask
    if not isinstance(names, Sequence) or not all(
        isinstance(name, str) for name in names
    ):
        raise TypeError("mask must be a mask name, a sequence of mask names, or None")
    unknown = sorted({name for name in names if name not in VARIANT_MASKS})
    if unknown:
        raise ValueError(
            f"Unknown masks {unknown}; masks are {', '.join(VARIANT_MASKS)}"
        )
    return tuple(sorted(set(names)))


def _clone_assignment(
    n_cells: int,
    positions: np.ndarray,
    called: FrequencyClones,
    group_names: Sequence[str] | None,
) -> CloneAssignment:
    """Rank called clones over all of the counts' cells."""
    clustered = called.clusters > 0
    codes = np.full(n_cells, -1, dtype=np.int64)
    if group_names is None:
        labels = [
            str(cluster)
            for cluster in range(1, int(called.clusters.max(initial=0)) + 1)
        ]
        codes[positions[clustered]] = called.clusters[clustered] - 1
    else:
        pairs = np.stack([called.groups[clustered], called.clusters[clustered]], axis=1)
        distinct, inverse = np.unique(pairs, axis=0, return_inverse=True)
        labels = [
            f"{group_names[group]}:{cluster}" for group, cluster in distinct.tolist()
        ]
        codes[positions[clustered]] = np.asarray(inverse, dtype=np.int64).reshape(-1)
    ranked, clone_ids, clone_sizes = ranked_clones(codes, labels)
    return CloneAssignment(
        codes=ranked,
        clone_ids=clone_ids,
        clone_sizes=clone_sizes,
        probabilities=None,
        n_cells_in_several_clones=0,
    )
