import math
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import zarr

from ...clones.assignments import CloneAssignment
from ...clones.assignments import ranked_clone_labels
from ...clones.calling import cell_sets, set_clones
from ...metadata.arguments import (
    FrequencyCloneArguments,
    GenotypeCloneArguments,
    VariantCountImportArguments,
    VariantGenotypeArguments,
    VariantSelectionArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact
from ...metadata.rows import (
    read_metadata_rows_chunkwise,
    read_text_rows,
    read_whole,
    text_read_bytes,
)
from ...metadata.selection import (
    CellSelectionRead,
    cell_selection_indices,
    plan_cell_selection_read,
    read_cell_selection,
)
from ...storage.arrays import create_metadata_column, zarr_write_buffer_bytes
from ...storage.artifact_writer import (
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
    planned_diagnostics,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
    require_complete_artifact,
)
from ...storage.diagnostics import (
    DIAGNOSTICS_ATTRIBUTE,
    EXAMPLES,
    DiagnosticRule,
    warn_diagnostics,
)
from ...storage.layout import get_compressors
from ...storage.profiles import resolve_storage_profile
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument, quoted_value
from ...utils.arrays import sorted_unique
from ...utils.files import is_hdf5
from ...utils.logging import logger
from ...utils.memory_budget import MemoryLedger
from ...utils.numba import numba_threads
from ...variants.counts import (
    CALL_DTYPE,
    CALL_FIELDS,
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
    band_frequencies,
    clone_status_counts,
    frequency_clones,
)
from ...variants.genotypes import (
    CALLS,
    FILTERED,
    GENOTYPE_RULES,
    MISSING,
    MUTANT_CALLS,
    SOURCE_CALLS,
    SOURCE_PRECEDENCE,
    WILD_TYPE,
    GenotypeRules,
    VariantGenotypes,
    gapfill_wild_type_probabilities,
    genotype_frames,
    genotype_rules,
    gift_calls,
    gift_posteriors,
    source_call_codes,
    threshold_calls,
    unit_swap_probabilities,
)
from ...variants.masks import VARIANT_MASKS, masked_variants
from ...variants.sorting import CellRowSorter, sort_rows
from ...variants.statistics import (
    HETEROPLASMY_LEVELS,
    THRESHOLDS,
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
    unmatched_barcode_rules,
)
from .cell_groups import group_columns, within_groups
from .clone_artifacts import SITE_ARRAYS, log_clones, plan_clones, write_clones

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
_SPILL_BUFFERS = 3
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
# A band holds at least this share of a stored chunk's rows, or of every
# row where fewer, so appending bands writes each chunk about 33 times at
# most, where bands of a few cells would rewrite every chunk once a band.
# 8,000 Tapestri cells of 40 variants, 838,970 rows in one chunk of each
# array, then need 18,026,255 bytes, hold 9,976,376 and run in 2 to 3 s;
# with a whole chunk to a band they need 101,623,601 bytes.
_BAND_CHUNK_PARTS = 32
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
_CELL_ARRAYS = ("mean_depth", "n_covered_sites", "listed")
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
# Bytes held for each cell while called clones are saved, beside what saving
# them holds: its position, group, cluster and status of the call.
_CLONE_CALL_BYTES_PER_CELL = 34
# The smallest Jaccard index of two neighbourhoods that Seurat's shared
# neighbour graph keeps.
_SHARED_NEIGHBOR_PRUNE = 1 / 15
# The attributes that saved variant clones need to be reused.
_CLONE_ATTRIBUTES = ("n_clones",)
# Bytes that calling genotypes, or the clones they define, holds whatever the
# counts: zarr's objects for the arrays that planning, writing and validating
# the artifact open, the chunk of the cell selection that planning validates,
# and the garbage that appending bands leaves until Python collects it.
_GENOTYPE_FIXED_BYTES = 1 << 20
# A band of genotypes holds at least this many rows, or every row, so that a
# small budget does not append the calls a few rows at a time.
_MIN_GENOTYPE_BAND_ROWS = 1_000
# Bytes per call row read while the sites of a source's calls are found: its
# allele, site and source, as read and stacked, and their distinct pairs.
_SOURCE_SITE_BYTES_PER_ROW = 64
# Calling genotypes holds, for each cell of the counts, its mutant and called
# sites and its offset as they are written.
_GENOTYPE_BYTES_PER_CELL = 24
# And for each of the counts' alleles, its site's rank and its gapfill's
# probability of the wild type.
_GENOTYPE_BYTES_PER_ALLELE = 24
# Bytes per stored row of a band that genotypes read, by method: the rows of
# the depth, counts or calls with their cells, the keys of their cells and
# sites and their order, and what the method derives per key, the UMIs and
# calls, the source's labels and the order of its columns, or GIFT's scores
# and posteriors; at most 110, 129 and 277 bytes as traced between bands of
# 20,000 and 80,000 rows on two shapes of counts.
_GENOTYPE_BYTES_PER_ROW = {"threshold": 144, "source": 168, "gift": 360}
# Bytes per call row of loaded genotypes: its row of the table of calls and
# the stored arrays read beside it, 174 as traced on 128,000 rows.
_LOADED_GENOTYPE_BYTES_PER_ROW = 220
# The attributes that saved genotype clones need to be reused: the exact
# rule's graph sizes, which describe the method, and the clones of each.
_EXACT_CLONE_ATTRIBUTES = ("n_clones", "n_nodes", "n_edges")
_LIKELIHOOD_CLONE_ATTRIBUTES = ("n_clones",)
# Bytes per cell that reading the genotype calls in bands holds beside the
# packed calls: each cell's position, and the offsets, their sum, the flags
# of the chosen cells, the gaps between them and each band's cells and their
# rows' counts that the bands read.
_CALL_BAND_BYTES_PER_CELL = 64
# Bytes per cell that ordering the likelihood's cells by their within group
# holds beside their positions: each cell's group, the order of the stable
# sort and its scratch, and the positions as found beside them reordered,
# 24 as traced on 90,000 and 180,000 cells of 2, 4 and 50 groups.
_GROUP_ORDER_BYTES = 32
# Bytes that numbering genotype clones holds for each cell: each clustered
# cell's key of group and clone, their sorted copy, first cells and order,
# and each cell's rank.
_CLONE_RANK_BYTES = 64


# The formats whose files hold a chosen subset of variants.
_VARIANT_CHOICE_FORMATS = ("tapestri", "dab_seq")
# The diagnostic of each call's count among a genotypes' calls.
_CALL_DIAGNOSTICS = {
    "wild_type": "wildTypeCalls",
    "heterozygous": "heterozygousCalls",
    "homozygous_alternative": "homozygousAlternativeCalls",
    "alternative_present": "alternativePresentCalls",
    "missing": "missingCalls",
    "filtered": "filteredCalls",
}
# The warning of genotypes whose alleles name an alternative no cell shows.
_GENOTYPE_RULES = (
    DiagnosticRule(
        "alternativesNotShown",
        lambda saved: (saved["alternativesNotShown"] or 0) > 0,
        lambda saved: (
            f"No cell shows the alternative that alleles gives "
            f"{saved['alternativesNotShown']} "
            f"{'site' if saved['alternativesNotShown'] == 1 else 'sites'}, so no cell "
            "is called mutant there; name each alternative in the orientation that "
            "the files give, such as a GIFT probe's"
        ),
    ),
)


def _skipped_variants(sources: Sequence["_Source"]) -> dict[str, Any]:
    """The diagnostics of the chosen variants that Tapestri and DAb-seq sources skip.

    A variant is counted once however many files of its panel skip it, and
    the examples are the first five in the order of the sources and their
    files; sources of other formats choose no variants, so their counts are
    None and they list no example.
    """
    from ...readers.tapestri import TapestriSource

    skipped = [
        source.reader.skipped
        for source in sources
        if isinstance(source.reader, TapestriSource)
    ]
    if not skipped:
        return {
            "skippedEmptyAlternatives": None,
            "skippedOtherAlleles": None,
            "skippedVariantExamples": [],
        }
    names = list(dict.fromkeys(name for entry in skipped for name in entry.names))
    empty = frozenset().union(*(entry.empty_alternatives for entry in skipped))
    n_empty = sum(name in empty for name in names)
    return {
        "skippedEmptyAlternatives": n_empty,
        "skippedOtherAlleles": len(names) - n_empty,
        "skippedVariantExamples": names[:EXAMPLES],
    }


def _skipped_variant_rule(files: str) -> DiagnosticRule:
    """The warning about the chosen variants that a Tapestri or DAb-seq import skips."""
    return DiagnosticRule(
        "skippedVariantExamples",
        lambda saved: bool(saved["skippedVariantExamples"]),
        lambda saved: (
            f"{files[0].upper()}{files[1:]} {'have' if files.endswith('s') else 'has'} "
            f"{saved['skippedEmptyAlternatives']} chosen variants with an empty ALT "
            f"and {saved['skippedOtherAlleles']} whose alleles are not bases, which "
            "no documentation explains, so they are skipped, such as "
            + ", ".join(map(repr, saved["skippedVariantExamples"]))
        ),
    )


def _variant_names(variants: Any) -> tuple[str, ...]:
    """The variants or positions that ``include_variants`` names, sorted and once each."""
    if variants is None:
        return ()
    variants = _listed(variants)
    if (
        isinstance(variants, str | bytes)
        or not isinstance(variants, Sequence)
        or not all(isinstance(item, str) for item in variants)
    ):
        raise TypeError(
            "include_variants must be a sequence of variant or position names, "
            "such as ['chr1:115256513:G/T'], or None"
        )
    return tuple(sorted(set(variants)))


def _read_source(
    path: str,
    format: str | None,
    swap_strands: bool,
    threads: int,
    include_variants: tuple[str, ...] = (),
    memory_bytes: int | None = None,
) -> VariantSource:
    """Read the description of one file or folder of variant counts.

    Tapestri and DAb-seq files check their description against
    ``memory_bytes`` before they read their candidate variants and cells.
    """
    from ...readers import _variant_files as files

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    detected = format
    if detected is None:
        if os.path.isdir(path):
            detected = files.folder_format(path)
        elif is_hdf5(path):
            detected = files.hdf5_format(path)
        else:
            from ...readers.got import got_table_format

            detected = got_table_format(path)
    if detected not in ("mgatk", "maegatk") and swap_strands:
        raise ValueError(
            "swap_strands applies to mgatk files from call or bcall of mgatk 0.4.0 "
            f"to 0.5.8, not to {detected} files"
        )
    if include_variants and detected not in _VARIANT_CHOICE_FORMATS:
        raise ValueError(
            "include_variants chooses the variants of Tapestri and DAb-seq files, "
            f"which import their passing variants by default, not of {detected} "
            "files"
        )
    if detected == "tapestri":
        from ...readers.tapestri import tapestri_source

        return tapestri_source(path, include_variants, memory_bytes)
    if detected == "dab_seq":
        from ...readers.tapestri import dab_seq_source

        return dab_seq_source(path, include_variants, memory_bytes)
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
    its depth is defined, and the fields that counts and calls carry.

    Attributes:
        field_paths: For each count field, the first source, in canonical
            order, whose counts carry it.
        call_fields: The fields of the calls beside the call, ``("quality",)``
            when a source gives each call's quality, as Tapestri does.
    """

    count_unit: CountUnit
    depth_definition: str
    depth_roles: frozenset[str] | None
    reference_from_depth: bool
    alleles_overlap: bool
    open_alleles: bool
    count_fields: tuple[str, ...]
    field_paths: dict[str, str]
    call_fields: tuple[str, ...]


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
        The kept rows; for repeats that differ in their call or its quality,
        the row of the first such repeat, or None; and the number of rows
        dropped.
    """
    if len(rows) < 2:
        return rows, None, 0
    same = (
        (rows["cell"][1:] == rows["cell"][:-1])
        & (rows["key"][1:] == rows["key"][:-1])
        & (rows["source"][1:] == rows["source"][:-1])
    )
    quality = rows["quality"]
    same_quality = (quality[1:] == quality[:-1]) | (
        np.isnan(quality[1:]) & np.isnan(quality[:-1])
    )
    differing = same & ((rows["value"][1:] != rows["value"][:-1]) | ~same_quality)
    if differing.any():
        return rows, int(np.flatnonzero(differing)[0]) + 1, 0
    kept = np.r_[True, ~same]
    return rows[kept], None, int(np.count_nonzero(same))


def _factorized(texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each text's position among the distinct texts, and the distinct texts.

    Raises:
        ValueError: If a value is missing rather than text.
    """
    inverse, distinct = pd.factorize(np.asarray(texts, dtype=object).reshape(-1))
    if np.any(inverse < 0):
        raise ValueError("A call or label is missing where text was expected")
    return np.asarray(inverse, dtype=np.int64), np.asarray(distinct, dtype=object)


class _Vocabulary:
    """Text coded in order of first appearance, sorted when stored."""

    def __init__(self) -> None:
        self._codes: dict[str, int] = {}

    def codes(self, texts: np.ndarray) -> np.ndarray:
        # Hashing the texts takes a pass over them, where sorting millions of
        # Python strings, as dense calls hold, takes seconds.
        inverse, distinct = _factorized(texts)
        lookup = np.asarray(
            [self.code(str(text)) for text in distinct.tolist()], dtype=np.int32
        )
        return np.asarray(lookup[inverse], dtype=np.int32)

    def code(self, text: str) -> int:
        """The code of one text."""
        return self._codes.setdefault(text, len(self._codes))

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


def _band_budget(budget: int, room: int, left: int, needed: int) -> int:
    """The least budget above ``budget`` at which a band of ``needed`` bytes fits beside what the variant import holds.

    The import held its sort buffers, ``_SPILL_BUFFERS`` of a
    ``_BUFFER_SHARE``-th of the ``room`` that its budget left once the join
    held its lookup, and ``left`` bytes are left now; every byte more of
    budget adds a byte to the room, of which the buffers take their share.
    """

    def left_at(extra: int) -> int:
        grown = (room + extra) // _BUFFER_SHARE - room // _BUFFER_SHARE
        return left + extra - _SPILL_BUFFERS * grown

    extra = max(1, math.ceil((needed - left) / (1 - _SPILL_BUFFERS / _BUFFER_SHARE)))
    while left_at(extra) < needed:
        extra += 1
    # The buffers grow a step every _BUFFER_SHARE bytes, so a smaller budget
    # just below may fit too.
    for fewer in range(extra - 1, max(0, extra - 2 * _BUFFER_SHARE), -1):
        if left_at(fewer) >= needed:
            extra = fewer
    return budget + extra


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
            f"The artifact has no site {', '.join(quoted_value(item) for item in unknown[:5])}"
        )
    return sorted_unique(np.asarray(chosen, dtype=np.int64))


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
    def _variant_selection_read(self, cell_selection: ArtifactRef) -> CellSelectionRead:
        """Check and count a cell selection, whose cells the import reads within its ledger."""
        planned = plan_cell_selection_read(self.zw, cell_selection)
        if not planned.cells:
            raise ValueError("cell_selection selects no cell")
        return planned

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
        include_variants: Sequence[str] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import per-cell allele counts of mitochondrial or targeted variants.

        Mitochondrial lineage tracing (mtscATAC-seq, MAESTER) and targeted
        genotyping (GoT, GIFT, Tapestri) count, for each cell, the reads or
        UMIs of each allele at variant sites: the positions of the
        mitochondrial genome or of a targeted panel, or targets and probes
        without coordinates. The import reads those counts,
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
        the GoT tables of IronThrone, GoT-Multi, Gotcha and GEO, a Tapestri
        Pipeline ``.h5`` file of DNA variants and a DAb-seq ``.hdf5`` file of
        genotypes. The variants API reference lists the files and columns of
        each format. Six-column mgatk files look like maegatk's, which counts
        strands the other way round, so they need ``format``.

        A Tapestri file imports the variants that pass the pipeline's default
        filters, and a DAb-seq file every variant, with those that
        ``include_variants`` names. Each position is a site, whose alleles are its
        reference and the alternatives of its variants; Tapestri gives each
        variant's alternative reads as ``AF × DP / 100``, its depth ``DP`` and
        its genotype ``NGT`` as a call with the genotype quality ``GQ``, and
        DAb-seq its ``AD``, ``RD``, ``DP`` and ``GT`` with ``GQ``. Variants
        with an empty ``ALT``, or alleles that are not bases, are skipped,
        counted in the ``skippedEmptyAlternatives`` and
        ``skippedOtherAlleles`` diagnostics, each variant once however many
        files of its panel skip it, with up to five
        ``skippedVariantExamples``, and named in a warning, which a reused
        import gives again; other formats choose no variants, so those
        counts are None. A Tapestri file's
        cells are named ``<sample>__<barcode>``, as
        :class:`~cytearc.TapestriToZarr` names them.

        A cell's ID is its file's ``barcode_prefix``, the barcode and
        ``barcode_suffix``, matched exactly; ``barcode_pattern`` keeps the
        part of each barcode that its ``barcode`` group matches first.
        Barcodes that match no selected cell are counted by the cell ID they
        name in the artifact's diagnostics, which
        ``ds.artifacts.inspect(ref).diagnostics`` returns: ``fileBarcodes``,
        ``unmatchedBarcodes``, those that name no cell of the store,
        ``unselectedBarcodes``, those of cells of the store outside the
        selection, and up to five examples of each, which show the barcode
        beside the cell ID it derives. Barcodes of no cell are named in a
        warning, which a reused import gives again. A source is identified by the
        SHA-256 of each file it reads, so moved files reuse the saved counts
        and the order of the paths does not matter. The diagnostics also
        hold ``listedCells``, the selected cells that the files list,
        ``sortedRuns``, the sorted runs of rows that the import wrote to
        disk because the rows did not fit the memory budget at once, and
        ``repeatedCalls``, the calls that a source repeats for a cell's
        allele, which are kept once.

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
        file names; ``n_covered_sites`` counts its sites with depth above 0 and
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
                ``"ironthrone"``, ``"got"``, ``"got_multi"``, ``"gotcha"``,
                ``"tapestri"`` and ``"dab_seq"``, or None to recognise each
                path's format.
            count_unit: ``"reads"``, ``"umis"`` or ``"molecules"``: what
                cellsnp-lite, MQuad and VarTrix counted, which their files do
                not state, since it depends on how they were run, and what a
                MiTo matrix counted when its ``uns`` does not say. Other
                formats count in a unit of their own, which a given value must
                match.
            swap_strands: Swap the forward and reverse counts of mgatk files
                written by ``call`` or ``bcall`` of mgatk 0.4.0 to 0.5.8, which
                counted reverse-aligned reads as forward.
            include_variants: Variants of Tapestri or DAb-seq files to import
                beside those that pass, named as the file names them, such as
                ``"chr1:115256513:G/T"``, or positions, such as
                ``"chr1:115256513"`` for every variant there; None for the
                passing variants alone. The source record counts the variants
                added, so names that add the same variants reuse the import.
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
                number of paths, ``include_variants`` is given for another format or
                names a variant that a file lacks or cannot be read, a file
                has no variant to import, or the selection holds no cell.
            FileNotFoundError: If a path or a file it needs is missing.
            PermissionError: If the store is read-only and the counts are not
                saved yet.
            MemoryError: If the import does not fit the memory budget.
        """
        swap_strands = bool_argument(swap_strands, "swap_strands")
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        from ...readers._variant_files import VARIANT_FORMATS, source_sha256

        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        chosen_variants = _variant_names(include_variants)
        if format is not None and format not in VARIANT_FORMATS:
            raise ValueError(
                f"format must be one of {', '.join(VARIANT_FORMATS)}, or None, not "
                f"{quoted_value(format)}"
            )
        if count_unit is not None and count_unit not in COUNT_UNITS:
            raise ValueError(
                f"count_unit must be one of {', '.join(COUNT_UNITS)}, or None, not "
                f"{quoted_value(count_unit)}"
            )
        readers: dict[str, tuple[VariantSource, str, dict[str, str]]] = {}
        threads = self.nthreads if self.memoryBytes >= _THREADED_PARSE_BYTES else 1

        def source_digest(path: str | os.PathLike[str]) -> str:
            """Read a path's source once, and return the SHA-256 of its files."""
            key = os.fspath(path)
            if key not in readers:
                reader = _read_source(
                    key,
                    format,
                    swap_strands,
                    threads,
                    chosen_variants,
                    self.memoryBytes,
                )
                digest, digests = source_sha256(reader.files)
                readers[key] = (reader, digest, digests)
            return readers[key][1]

        files = barcode_files(
            paths,
            barcode_prefix=barcode_prefix,
            barcode_suffix=barcode_suffix,
            barcode_pattern=barcode_pattern,
            what="variant file",
            digest=source_digest,
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
        selection = self._variant_selection_read(cell_selection)
        n_cells = selection.cells
        arguments = VariantCountImportArguments(
            cell_selection=cell_selection,
            sources=tuple(source.record() for source in sources),
            count_unit=rules.count_unit,
            swap_strands=swap_strands,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_variant_counts(arguments, cell_selection, n_cells)
        files_text = _FILES_TEXT[0] if len(sources) == 1 else _FILES_TEXT[1]
        warnings = (
            *unmatched_barcode_rules(files_text),
            _skipped_variant_rule(files_text),
        )
        if planned.reused:
            warn_diagnostics(
                arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
            )
            return planned.ref
        self._require_writable(arguments.operation)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Importing variant counts for {n_cells} selected cells",
            remedy="pass a smaller cell_selection",
            codec_threads=self.nthreads,
        )
        ledger.hold("each selected cell's state", _IMPORT_BYTES_PER_CELL * n_cells)
        cell_index = read_cell_selection(
            ledger, self.zw, selection, item="the selected cells' positions"
        )
        cell_ids = read_text_rows(
            ledger,
            self.cells._get_array("ids"),
            cell_index,
            item="the selected cells' IDs",
        )
        join = CellBarcodeJoin(
            cell_ids, ledger=ledger, store_ids=self.cells._get_array("ids")
        )
        # The parsed chunks, the sort buffer and the bands share what the
        # ledger's room leaves once the join holds its lookup; the hashes of
        # barcodes that match no cell grow in what the shares leave.
        room = int(self.memoryBytes if ledger.room is None else ledger.room)
        chunk_rows = int(
            min(room // (_PARSE_SHARE * _PARSE_BYTES_PER_ROW), _MAX_CHUNK_ROWS)
        )
        parsing = _PARSE_BYTES_PER_ROW * max(chunk_rows, _MIN_CHUNK_ROWS)
        if chunk_rows < _MIN_CHUNK_ROWS:
            # The least chunk, at its share of the room.
            ledger.peak(
                f"a chunk of {_MIN_CHUNK_ROWS} parsed rows", _PARSE_SHARE * parsing
            )
        buffer_bytes = room // _BUFFER_SHARE
        ledger.hold("the sorted rows' buffers", _SPILL_BUFFERS * buffer_bytes)
        counts_dtype = count_dtype(rules.count_fields)
        sorter = CellRowSorter(
            {
                "counts": (counts_dtype, ("cell", "key")),
                "depth": (DEPTH_DTYPE, ("cell", "key")),
                "calls": (CALL_DTYPE, ("cell", "key", "source")),
            },
            n_cells=n_cells,
            buffer_bytes=buffer_bytes,
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
                batches = source.reader.batches(chunk_rows)
                while True:
                    ledger.hold(f"a chunk of {chunk_rows} parsed rows", parsing)
                    batch = next(batches, None)
                    if batch is None:
                        ledger.release(parsing)
                        break
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
                        call_fields=rules.call_fields,
                    )
                    del batch
                    ledger.release(parsing)
            if not n_rows:
                raise ValueError(
                    f"No row of counts, depth, calls or cells is in {files_text}"
                )
            # A band's rows sit beside the chunks that appending them rewrites.
            n_entry_rows, n_call_rows = _stored_row_bounds(sorter)
            rewritten = (
                _REWRITTEN_CHUNKS * min(_ROW_BAND, max(1, n_entry_rows)) * _STORED_BYTES
            )
            # The bands take their share of the room, or what the room leaves
            # now that the rows are parsed, if less. Appending a band rewrites
            # the stored chunks that it ends in, so a band holds at least a
            # share of a chunk's rows, _BAND_CHUNK_PARTS.
            left = room if ledger.room is None else ledger.room
            band_bytes = int(min(room / _BAND_SHARE, left)) - rewritten
            least_rows = math.ceil(
                min(_ROW_BAND, max(1, n_entry_rows + n_call_rows)) / _BAND_CHUNK_PARTS
            )
            row_bytes = _BAND_COPIES * counts_dtype.itemsize
            band_bytes = max(least_rows * row_bytes, band_bytes)
            item = (
                f"a band of at least {least_rows} sorted rows and the stored chunks "
                "that it rewrites"
            )
            if ledger.room is not None and rewritten + band_bytes > ledger.room:
                # The sort buffers grow with the budget, so the refusal names
                # the least budget at which the band fits beside them.
                ledger.refuse_beyond(
                    item,
                    _band_budget(
                        self.memoryBytes, room, ledger.room, rewritten + band_bytes
                    ),
                )
            ledger.peak(item, rewritten + band_bytes)
            band_rows = band_bytes // row_bytes
            attributes = {
                "format": formats[0],
                "count_unit": rules.count_unit,
                "depth_definition": rules.depth_definition,
                "stranded": "forward" in rules.count_fields,
                "count_fields": list(rules.count_fields),
                "call_fields": list(rules.call_fields),
                "contigs": [contig.record() for contig in site_table.contigs],
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
                diagnostics={**join.diagnostics(), **_skipped_variants(sources)},
            )
        warn_diagnostics(
            arguments, inspect_artifact(self.zw, planned.ref).diagnostics, warnings
        )
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
        """The unit, depth rules, count fields and call fields that every
        source shares.

        Each source's unit and depth rules must agree with the others', so
        the artifact records them once whatever order the sources take. The
        count fields are those of all sources together: a table of calls
        alone gives none, and counts that lack a field another source gives
        are refused as they arrive. The call fields are those of all sources
        together, so the calls' quality is stored only when a source gives it.

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
            first.alleles_overlap,
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
                reader.alleles_overlap,
                reader.open_alleles,
            ) != depth_rules:
                raise ValueError(
                    f"The files define depth differently: {first.path} counts "
                    f"{quoted_value(first.depth_definition)} and {reader.path} "
                    f"{quoted_value(reader.depth_definition)}, as cellsnp-lite folders with "
                    "and without OTH do; import them separately"
                )
            for name in reader.count_fields:
                field_paths.setdefault(name, reader.path)
        given = {name for source in sources for name in source.reader.call_fields}
        return _ImportRules(
            count_unit=unit,
            depth_definition=first.depth_definition,
            depth_roles=first.depth_roles,
            reference_from_depth=first.reference_from_depth,
            alleles_overlap=first.alleles_overlap,
            open_alleles=first.open_alleles,
            count_fields=tuple(name for name in COUNT_FIELDS if name in field_paths),
            field_paths=field_paths,
            call_fields=tuple(name for name in CALL_FIELDS if name in given),
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
            call_fields = group.attrs.get("call_fields")
            if not isinstance(call_fields, list) or not set(call_fields) <= set(
                CALL_FIELDS
            ):
                return False
            if not {f"call_{name}" for name in call_fields} <= names:
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
                "n_covered_sites": ((n_cells,), "i"),
                "listed": ((n_cells,), "b"),
            },
            invalidate_cache=arguments.invalidate_cache,
            required_attributes=(
                "format",
                "count_unit",
                "depth_definition",
                "count_fields",
                "call_fields",
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
        call_fields: tuple[str, ...],
    ) -> None:
        """Join a batch's rows to cells and add those of selected cells.

        Raises:
            ValueError: If counts lack a field that another source's counts
                carry, calls carry a quality that their source does not list,
                or a value lies outside the stored range.
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
            rows["source"] = call_sources.code(batch.call_source)
            rows["value"] = call_values.codes(batch.calls[kept])
            if len(batch.call_quality):
                if "quality" not in call_fields:
                    raise ValueError(
                        f"{path} gives call qualities, which its format does not "
                        "list among its call fields"
                    )
                if len(batch.call_quality) != batch.n_rows:
                    raise ValueError(
                        f"{path} gives {len(batch.call_quality)} call qualities for "
                        f"{batch.n_rows} calls"
                    )
                rows["quality"] = np.asarray(batch.call_quality)[kept]
            else:
                rows["quality"] = np.nan
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
        diagnostics: dict[str, Any],
    ) -> None:
        """Write the sorted rows of the counts in bands, then each cell's and site's arrays.

        ``diagnostics`` holds the join's counts and the skipped variants; the
        counts' own, the sorted runs, the repeated calls and the listed cells,
        are known once every band is written and join them then.
        """
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
            return (
                f"{quoted_value(sites.allele[allele])} at the site {sites.label(site)}"
            )

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
                    **{
                        f"call_{name}": (np.float32, n_call_rows)
                        for name in rules.call_fields
                    },
                },
            )
            for first, stop, rows in sorter.bands(band_rows):
                counts_rows = rows["counts"]
                repeated = _repeated_row(counts_rows, ("cell", "key"))
                if repeated is not None:
                    raise ValueError(
                        f"The cell {quoted_value(name_cell(int(counts_rows['cell'][repeated])))} "
                        "is counted twice for the allele "
                        f"{allele_text(int(counts_rows['key'][repeated]))}, by two "
                        "files or two rows of one file"
                    )
                depth_rows = rows["depth"]
                repeated = _repeated_row(depth_rows, ("cell", "key"))
                if repeated is not None:
                    raise ValueError(
                        f"The cell {quoted_value(name_cell(int(depth_rows['cell'][repeated])))} "
                        "has two depths at the site "
                        f"{sites.label(int(depth_rows['key'][repeated]))}, from two "
                        "files or two rows of one file"
                    )
                # A call that a source repeats is kept once; two different
                # calls of one source about one cell's allele are refused.
                call_rows, conflict, n_repeats = _distinct_calls(rows["calls"])
                if conflict is not None:
                    raise ValueError(
                        f"The cell {quoted_value(name_cell(int(call_rows['cell'][conflict])))} "
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
                    alleles_overlap=rules.alleles_overlap,
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
                for name in rules.call_fields:
                    appender.append(f"call_{name}", call_rows[name])
            mean_depth = np.where(listed, depth_total / max(sites.n_sites, 1), np.nan)
            _write_array(group, "mean_depth", mean_depth)
            _write_array(group, "n_covered_sites", covered)
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
            attributes["n_sites"] = sites.n_sites
            attributes["n_alleles"] = sites.n_alleles
            attributes[DIAGNOSTICS_ATTRIBUTE] = planned_diagnostics(
                planned,
                {
                    **diagnostics,
                    "listedCells": int(listed.sum()),
                    "sortedRuns": sorter.n_runs,
                    "repeatedCalls": repeated_calls,
                },
            )
            group.attrs.update(attributes)

    def _variant_group(self, counts: ArtifactRef) -> zarr.Group:
        """Return a complete ``variant_counts`` artifact's group."""
        if not isinstance(counts, ArtifactRef):
            raise TypeError("counts must be an ArtifactRef")
        if counts.kind != VariantCountImportArguments.artifact_kind:
            raise ValueError(
                f"counts must be a variant_counts artifact, not {quoted_value(counts.kind)}"
            )
        require_complete_artifact(self.zw, counts)
        group = artifact_group(self.zw, counts)
        missing = [name for name in _CELL_ARRAYS if name not in group]
        if missing:
            raise ValueError(
                "counts was saved by an earlier version of CyteArc, without the "
                f"arrays {', '.join(missing)}; import the counts again with "
                "ds.variants.import_counts, which saves them anew"
            )
        if "call_fields" not in group.attrs:
            raise ValueError(
                "counts was saved by an earlier version of CyteArc, without the "
                "attribute call_fields; import the counts again with "
                "ds.variants.import_counts, which saves them anew"
            )
        return group

    def _variant_cell_index(self, counts: ArtifactRef) -> np.ndarray:
        """The store rows of a variant_counts artifact's cells, in its order."""
        selection = inspect_artifact(self.zw, counts).input_ref("cell_selection")
        return cell_selection_indices(self.zw, selection)

    def _variant_cell_positions(
        self, counts: ArtifactRef, cell_index: np.ndarray, cell_selection: Any
    ) -> np.ndarray:
        """The artifact's cells that ``cell_selection`` names, as ascending positions."""
        cells = cell_selection
        if cells is None:
            return np.arange(len(cell_index), dtype=np.int64)
        if isinstance(cells, ArtifactRef):
            chosen = cell_selection_indices(self.zw, cells)
            positions = np.searchsorted(cell_index, chosen)
            inside = positions < len(cell_index)
            inside[inside] = cell_index[positions[inside]] == chosen[inside]
            if not inside.all():
                raise ValueError(
                    f"cell_selection selects {int((~inside).sum())} cells that the "
                    "counts do not hold; pass a selection within the counts' "
                    "cell_selection"
                )
            return np.asarray(positions, dtype=np.int64)
        cells = _listed(cells)
        if isinstance(cells, str | bytes) or not isinstance(cells, Sequence):
            raise TypeError(
                "cell_selection must be a cell selection ArtifactRef, a sequence of "
                "cell IDs, or None"
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
                + ", ".join(quoted_value(str(cell)) for cell in missing)
            )
        return sorted_unique(positions)

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
        cell_selection: ArtifactRef
        | Sequence[str]
        | np.ndarray
        | pd.Index
        | None = None,
    ) -> VariantCounts:
        """Read saved variant counts as tables of sites, alleles, cells and rows.

        The long tables hold one row per stored entry: a cell's depth at a
        site where it was covered, its count of an allele it showed, and the
        calls its source made. A cell and site without a depth row was not
        covered, so a frequency there is undefined; one with depth but
        without a count row of an allele showed none of it. ``sites`` and
        ``cell_selection`` keep the tables to some sites and cells;
        ``cell_selection=[]`` reads only the sites and alleles, with each
        site's depth summed over all cells.

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
            cell_selection: A cell selection within the counts' cells, or cell
                IDs as a list, a NumPy array or a pandas index; None for all
                cells.

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
        positions = self._variant_cell_positions(counts, cell_index, cell_selection)
        fields = [str(name) for name in cast(list[str], group.attrs["count_fields"])]
        count_arrays = (
            "count_alleles",
            "counts",
            *(f"counts_{name}" for name in fields),
        )
        call_fields = [
            str(name) for name in cast(list[str], group.attrs["call_fields"])
        ]
        call_arrays = (
            "call_alleles",
            "call_sources",
            "call_values",
            *(f"call_{name}" for name in call_fields),
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
                    "call": call_arrays,
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
            "call", call_arrays, "call_alleles", allele_kept
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
            n_covered_sites=_read(group, "n_covered_sites")[positions],
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
                {name: call_values[f"call_{name}"] for name in call_fields},
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
            raise TypeError("informative must be an ArtifactRef")
        if informative.kind != VariantSelectionArguments.artifact_kind:
            raise ValueError(
                "informative must be a variant_selection artifact from "
                f"ds.variants.informative, not {quoted_value(informative.kind)}"
            )
        status = require_complete_artifact(self.zw, informative)
        if status.operation != VariantSelectionArguments.operation:
            raise ValueError(
                "informative was saved by an earlier version of CyteArc, by the "
                f"operation {quoted_value(status.operation)}; select the informative variants "
                "again with ds.variants.informative, which saves them anew"
            )
        return artifact_group(self.zw, informative)

    def _statistic_positions(
        self,
        counts: ArtifactRef,
        group: zarr.Group,
        cell_selection: ArtifactRef | None,
    ) -> np.ndarray:
        """The positions of the cells that statistics cover: the chosen cells
        that a file names."""
        cell_index = self._variant_cell_index(counts)
        positions = self._variant_cell_positions(counts, cell_index, cell_selection)
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
        cell_selection: ArtifactRef | None = None,
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
        preset's. Every statistic must reach its threshold, so
        ``min_vmr=0.01`` keeps a VMR of exactly 0.01; a preset's rule that a
        statistic exceed its value holds the next floating-point number above
        it, as ``cytearc.variants.INFORMATIVE_PRESETS`` shows, so the presets
        keep exactly the variants of the published rules. A statistic that is
        undefined, such as the strand correlation of a variant that one cell
        shows, fails.

        Signac computes the same statistics but sets the frequency of an
        uncovered cell to 0 and, by default, replaces the frequency of a cell
        with depth below 10 by the pooled mean before taking the variance;
        here an uncovered cell has no frequency, as in mgatk, so variances
        and VMRs differ where coverage is sparse.

        The artifact's diagnostics, which
        ``ds.artifacts.inspect(ref).diagnostics`` returns, count the
        ``cells`` whose evidence was scored, the ``variants`` scored, the
        ``informativeVariants`` and the ``maskedVariants``.

        Args:
            counts: A ``variant_counts`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.import_counts`.
            cell_selection: A cell selection within the counts' cells whose
                evidence the statistics use, such as ``ds.select_cells(counts,
                low=10)`` for the cells whose mean depth exceeds 10, as mgatk
                scores them; None uses every cell that the counts' files
                name. Cells that no file names are left out either way.
            preset: ``"atac"``, ``"rna"`` or None.
            min_cells: The fewest cells with 2 or more reads of the allele on
                each strand.
            min_strand_correlation: The lowest strand correlation.
            min_vmr: The lowest VMR.
            min_depth: The lowest mean depth at the site.
            min_quality: The lowest mean base quality.
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
            it with
            :meth:`~cytearc.datastore.namespaces.VariantsAccessor.frequency_clones`.

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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        group = self._variant_group(counts)
        if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
            raise TypeError(
                "cell_selection must be a cell selection ArtifactRef or None"
            )
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
        positions = self._statistic_positions(counts, group, cell_selection)
        arguments = VariantSelectionArguments(
            counts=counts,
            cell_selection=cell_selection,
            preset=cast(Any, preset),
            min_cells=cast(int | None, thresholds["min_cells"]),
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
            DIAGNOSTICS_ATTRIBUTE: planned_diagnostics(
                planned,
                {
                    "cells": statistics.n_cells,
                    "variants": statistics.n_variants,
                    "informativeVariants": int(informative.sum()),
                    "maskedVariants": int(masked.sum()),
                },
            ),
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
            required_attributes=("stranded", "qualities"),
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
        status = inspect_artifact(self.zw, informative)
        counts = status.input_ref("counts")
        sites = _stored_sites(self._variant_group(counts))
        alleles = _read(group, "variant_alleles").astype(np.int64)
        attributes = group.attrs
        frame = variant_selection_frame(
            sites,
            alleles,
            {name: _read(group, name) for name in group.array_keys()},
        )
        # The settings come from the provenance parameters, their one home.
        parameters = dict(status.parameters or {})
        return VariantSelection(
            variants=frame,
            counts=counts,
            cell_selection=None
            if (status.inputs or {}).get("cell_selection") is None
            else status.input_ref("cell_selection"),
            preset=cast(str | None, parameters["preset"]),
            thresholds={name: parameters[name] for name in THRESHOLDS},
            mask=tuple(str(name) for name in cast(list[str], parameters["mask"])),
            n_cells=int(cast(int, status.diagnostics["cells"])),
            stranded=bool(attributes["stranded"]),
            qualities=bool(attributes["qualities"]),
        )

    @validation_scoped
    def _variants_frequency_clones(
        self,
        informative: ArtifactRef,
        *,
        n_neighbors: int = 10,
        resolution: float = 1.0,
        within: str | Sequence[str] | None = None,
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
        and get none; the artifact's diagnostics, which
        ``ds.artifacts.inspect(clones).diagnostics`` returns, count them in
        ``cellsWithoutCoverage``, ``cellsWithoutAlternative`` and
        ``cellsWithoutNeighbors``, beside the informative ``variants``, the
        ``cellsScored`` and the ``edges`` of the shared-neighbour graph, and,
        as every source's clones, ``cellsWithClone`` and
        ``largestCloneShare``. The clones cover the cells
        of the counts that ``informative`` scored, and its statistics cover
        the cells that the clones are called from.

        Clones are numbered by size over every ``within`` group, so
        ``clone_1`` is the largest, and ties follow their first cell.

        Args:
            informative: A ``variant_selection`` artifact from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.informative`,
                whose counts the frequencies are read from.
            n_neighbors: The cells of a full neighbourhood, the cell itself
                included as Seurat counts it, at least 2.
            resolution: The Leiden resolution; larger values give more,
                smaller clones.
            within: Cell metadata columns whose labels keep clones apart, such
                as the donor or the mouse: clones never join cells of two
                groups. Every cell that the statistics covered needs a label.
            invalidate_cache: Call clones again instead of reusing them.

        Returns:
            A complete datastore-scoped ``clones`` artifact over the counts'
            cells, which every clone summary reads. Read it with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_clones`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``informative`` is not a ``variant_selection``
                artifact or selects no informative variant, ``n_neighbors`` is
                below 2, the resolution is not finite and positive, or a
                within column is missing, is named twice or has no label for
                a cell that the statistics covered.
            KeyError: If the store holds no such ``informative`` artifact, or
                not the counts that it scored.
            RuntimeError: If the ``informative`` artifact or its counts are
                incomplete.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If grouping the cells by ``within``, the frequencies,
                similarities or graph, or the clone labels as the store saves
                them do not fit the memory budget.
        """
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        from ...clustering.leiden import canonical_resolution

        selection = self._selection_group(informative)
        status = inspect_artifact(self.zw, informative)
        counts = status.input_ref("counts")
        group = self._variant_group(counts)
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
        columns = group_columns(within, "within")
        chosen = _read(selection, "informative").astype(bool)
        alleles = _read(selection, "variant_alleles").astype(np.int64)[chosen]
        if not len(alleles):
            raise ValueError(
                "informative selects no informative variant, so no clone can be "
                "called; relax its thresholds"
            )
        cell_selection = inspect_artifact(self.zw, counts).input_ref("cell_selection")
        cell_index = self._variant_cell_index(counts)
        n_cells = len(cell_index)
        # The statistics covered the chosen cells, or every listed cell.
        chosen_cells = (
            None
            if (status.inputs or {}).get("cell_selection") is None
            else status.input_ref("cell_selection")
        )
        positions = self._statistic_positions(counts, group, chosen_cells)
        # The groups are those that the clone summaries group cells into,
        # read and coded within the memory budget, and are identified by
        # each scored cell's group and the groups' labels.
        scored = np.zeros(n_cells, dtype=bool)
        scored[positions] = True
        group_codes, _keys, fingerprint = within_groups(
            self.zw,
            self.cells,
            cell_selection,
            columns,
            scored,
            ledger=MemoryLedger(
                self.memoryBytes,
                f"Grouping the {n_cells} cells of the variant counts",
                codec_threads=self.nthreads,
            ),
            what="counts that the statistics scored",
        )
        del scored
        arguments = FrequencyCloneArguments(
            cell_selection=cell_selection,
            informative=informative,
            within_fingerprint=fingerprint,
            n_neighbors=int(n_neighbors),
            resolution=resolution,
            within=columns,
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
            groups=group_codes[positions] if columns else None,
            n_neighbors=int(n_neighbors),
            resolution=resolution,
            random_seed=_CLONE_SEED,
            memory_bytes=self.memoryBytes - matrix_bytes - cell_bytes,
            nthreads=self.nthreads,
        )
        del frequencies, group_codes
        assignment = _clone_assignment(
            n_cells, positions, groups=called.groups, clusters=called.clusters
        )
        counts_by_status = clone_status_counts(called.status)
        saving = MemoryLedger(
            self.memoryBytes,
            f"Calling clones from the variant frequencies of {n_cells} cells",
            codec_threads=self.nthreads,
        )
        saving.hold(
            "each cell's call and the clones' labels",
            _CLONE_CALL_BYTES_PER_CELL * n_cells
            + sum(map(sys.getsizeof, assignment.clone_ids.tolist())),
        )
        write_clones(
            self.zw,
            planned,
            assignment,
            {"n_clones": len(assignment.clone_ids)},
            ledger=saving,
            diagnostics={
                "variants": len(alleles),
                "cellsScored": len(positions),
                "cellsWithoutCoverage": counts_by_status["no_coverage"],
                "cellsWithoutAlternative": counts_by_status["no_alternative"],
                "cellsWithoutNeighbors": counts_by_status["no_neighbor"],
                "edges": called.n_edges,
            },
        )
        log_clones(assignment, n_cells)
        return planned.ref

    def _genotype_group(self, genotypes: ArtifactRef) -> zarr.Group:
        """Return a complete ``variant_genotypes`` artifact's group."""
        if not isinstance(genotypes, ArtifactRef):
            raise TypeError("genotypes must be an ArtifactRef")
        if genotypes.kind != VariantGenotypeArguments.artifact_kind:
            raise ValueError(
                f"genotypes must be a variant_genotypes artifact, not "
                f"{quoted_value(genotypes.kind)}"
            )
        require_complete_artifact(self.zw, genotypes)
        return artifact_group(self.zw, genotypes)

    @validation_scoped
    def _variants_genotypes(
        self,
        counts: ArtifactRef,
        *,
        method: str,
        sites: Sequence[str | int] | np.ndarray | pd.Index | None = None,
        alleles: Mapping[str, Sequence[str]] | None = None,
        min_umis: int | None = None,
        min_mutant_umis: int | None = None,
        min_fraction: float | None = None,
        min_posterior: float | None = None,
        swap_probabilities: Mapping[int, float] | None = None,
        min_quality: float | None = None,
        min_depth: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call each cell's genotype at targeted sites, in one vocabulary of calls.

        A call says what a cell carries at a site: ``wild_type``,
        ``heterozygous``, ``homozygous_alternative``, ``alternative_present``
        (the alternative, without a claim of how many copies carry it),
        ``missing`` (no evidence) or ``filtered`` (evidence that the rule did
        not call). Three methods call them:

        - ``"source"`` adopts the calls that the tool wrote into the imported
          files, such as GoT's ``Genotype``, GoT-Multi's ``genotype_merged``
          or ``genotype_final`` and Gotcha's ``genotype_pred``, keeping each
          call's own label. Where the files wrote several call columns for a
          site, each cell takes its call from the first that holds one for
          it: GoT-Multi's refined call, then its merged call, then a single
          call column, and its channels' calls last. A label of no call where
          the cell holds evidence at the site or at its channels, such as
          ``Unprofiled`` beside ambiguous UMIs, is ``filtered``, and a cell
          with evidence but no call is ``filtered`` too. Tapestri and DAb-seq
          call each variant of a position apart, so each alternative of a
          position is genotyped as a site of its own, labelled like
          ``chr1:115256513:G/T``. ``min_quality`` and ``min_depth`` filter
          the adopted calls, as Tapestri's pipeline filters genotypes by
          their quality and depth: a call whose quality, such as Tapestri's
          ``GQ``, is below ``min_quality`` or missing, or whose cell's depth
          at the site, or at the deepest of its channels, is below
          ``min_depth``, is ``filtered``. Only Tapestri's and DAb-seq's calls
          carry a quality, so counts of other formats refuse ``min_quality``,
          and a genotyped site at which no cell holds depth, as in a GoT
          table of calls alone, refuses ``min_depth``.
        - ``"threshold"`` calls each cell from its UMIs of the site's wild type
          and alternative by the rules given, since GoT studies apply
          different ones: ``min_mutant_umis=1`` calls a cell
          ``alternative_present`` with one mutant UMI and ``wild_type``
          with none, as for patients' cells, and ``min_fraction`` calls a
          cell mutant or wild type when that allele's share of its UMIs
          reaches it. Nam et al. (2019) called cell lines whose allele held
          more than 90% of their UMIs, which
          ``min_fraction=float(np.nextafter(0.9, 1))`` reproduces exactly;
          ``min_fraction=0.9`` also calls a cell whose allele holds exactly
          90%, such as 9 of 10 UMIs. ``min_umis`` UMIs of the two, 1 by
          default, are needed for any call.
        - ``"gift"`` applies GIFT's model to giftwrap's counts: each UMI's
          gapfill goes to the probe's wild type or mutant by its edit
          distance to them, each UMI may come from template switching with
          the probability that ``swap_probabilities`` gives for its reads,
          and the UMIs of a cell give the posterior of the wild type, the
          heterozygote and the homozygous mutant under a uniform prior. The
          most probable genotype is called when its posterior reaches
          ``min_posterior`` and the cell holds at least ``min_umis`` UMIs, 1
          by default, whose swap probability lies below 0.1. By default the
          posterior must exceed 0.6, as GIFT calls cells, which the artifact
          records as the next number above 0.6, 0.6000000000000001. Without
          ``swap_probabilities`` every UMI counts, as GIFT treats cell lines
          once their UMIs of few reads are dropped.

        ``"threshold"`` and ``"gift"`` compare each site's wild type, the
        allele of role ``"reference"``, with its one alternative; ``alleles``
        names them for a site whose files do not, such as a GIFT probe whose
        manifest lacks its gapfills, as ``{"JAK2 c.1849G>T": ("CAC",
        "AAC")}``. Without ``sites``, every site that the method can call is
        genotyped: those with calls for ``"source"``, those with one wild
        type and one alternative otherwise.

        The artifact's diagnostics, which
        ``ds.artifacts.inspect(ref).diagnostics`` returns, count the calls of
        each kind, ``wildTypeCalls``, ``heterozygousCalls``,
        ``homozygousAlternativeCalls``, ``alternativePresentCalls``,
        ``missingCalls`` and ``filteredCalls``, and
        ``alternativesNotShown``, the sites whose alternative that
        ``alleles`` names no cell shows, None for ``"source"``, which
        compares no alleles; such sites give a warning, which reused
        genotypes give again.

        Args:
            counts: A ``variant_counts`` artifact from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.import_counts`.
            method: ``"source"``, ``"threshold"`` or ``"gift"``.
            sites: Site labels to genotype, such as ``["CALR"]`` or
                ``["chrM:310"]``, or positions of counts of one contig, such
                as ``[310]``, as a list, a NumPy array or a pandas index; None
                for every site the method can call.
            alleles: Each site's wild-type and alternative allele, by label.
            min_umis: The fewest UMIs that a call needs, for ``"threshold"``
                and ``"gift"``.
            min_mutant_umis: The fewest mutant UMIs of a mutant call, for
                ``"threshold"``.
            min_fraction: The share, above 0.5 and at most 1, that the
                called allele's UMIs must reach, for ``"threshold"``.
            min_posterior: The posterior, from 0 to 1, that a ``"gift"`` call
                must reach; None for GIFT's rule, a posterior above 0.6.
            swap_probabilities: For ``"gift"``, the probability that a UMI of
                each number of reads came from template switching, from 1
                read up; a UMI takes the value of the most reads given that
                it reaches, and its reads are its entry's reads per UMI.
            min_quality: For ``"source"``, the quality that an adopted call
                needs, such as Tapestri's ``GQ``, or None to adopt calls
                whatever their quality.
            min_depth: For ``"source"``, the depth at the site that an adopted
                call needs, or None to adopt calls whatever their depth.
                Every genotyped site needs depth in some cell.
            invalidate_cache: Call the genotypes again instead of reusing them.

        Returns:
            A complete datastore-scoped, cell-aligned ``variant_genotypes``
            artifact over the counts' cells. Its canonical ``n_mutant_sites``
            counts each cell's mutant sites. Read it with
            :meth:`~cytearc.datastore.namespaces.VariantsAccessor.load_genotypes`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``counts`` is not a ``variant_counts`` artifact,
                the method is unknown, a rule is given to a method that does
                not use it or is out of range, ``min_quality`` is given for
                counts whose calls carry no quality or ``min_depth`` for a site
                at which no cell holds depth, a named site is unknown or
                cannot be called by the method, ``alleles`` names an unknown
                site, a source's call label is unknown, or no site can be
                genotyped.
            KeyError: If the store holds no such ``counts`` artifact.
            RuntimeError: If the ``counts`` artifact is incomplete.
            PermissionError: If the store is read-only and the genotypes are
                not saved yet.
            MemoryError: If the counts' rows, the calls or the cells' arrays
                do not fit the memory budget.
        """
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        group = self._variant_group(counts)
        rules = genotype_rules(
            method,
            min_umis=min_umis,
            min_mutant_umis=min_mutant_umis,
            min_fraction=min_fraction,
            min_posterior=min_posterior,
            swap_probabilities=swap_probabilities,
            alleles=alleles,
            min_quality=min_quality,
            min_depth=min_depth,
        )
        if method == "gift" and str(group.attrs["format"]) != "giftwrap":
            raise ValueError(
                "method='gift' weighs the gapfills of giftwrap's probes, and these "
                f"counts come from {quoted_value(str(group.attrs['format']))} files; use "
                "method='threshold' for the UMIs of wild type and mutant"
            )
        if method == "threshold" and str(group.attrs["format"]) == "tapestri":
            raise ValueError(
                "method='threshold' compares the reads of the wild type and the "
                "alternative, and Tapestri's files hold no reads of the reference; "
                "use method='source' to adopt its NGT calls"
            )
        if rules.min_quality is not None and "quality" not in cast(
            list[str], group.attrs["call_fields"]
        ):
            raise ValueError(
                "min_quality tests call_quality, which these counts cannot give: "
                f"their {quoted_value(str(group.attrs['format']))} calls carry no quality, "
                "while Tapestri's and DAb-seq's carry their GQ; leave min_quality out"
            )
        table = _stored_sites(group)
        explicit = sites is not None
        chosen = _site_positions(table, sites)
        if explicit and not len(chosen):
            raise ValueError("sites names no site")
        labels = table.labels()
        if rules.alleles:
            unknown = sorted(set(rules.alleles) - set(labels.tolist()))
            if unknown:
                raise ValueError(
                    f"alleles names {quoted_value(unknown[0])}, which no site is"
                )
        n_cells = _n_counts_cells(group)
        # Each genotyped unit is a site or, for source calls at a site with
        # coordinates, one of its alternative alleles; -1 marks a whole site.
        if method == "source":
            genotyped, unit_alleles, site_columns = self._source_call_sites(
                group, table, chosen, explicit, n_cells
            )
            site_sources = [", ".join(columns) for columns in site_columns]
            site_alleles = _unit_alleles(table, genotyped, unit_alleles)
            reference_alleles = alternative_alleles = np.empty(0, dtype=np.int64)
        else:
            genotyped, site_alleles, reference_alleles, alternative_alleles = (
                _compared_alleles(table, chosen, explicit, rules)
            )
            unit_alleles = np.full(len(genotyped), -1, dtype=np.int64)
            site_columns = [() for _ in genotyped]
            site_sources = ["" for _ in genotyped]
        if not len(genotyped):
            raise ValueError(
                f"No site of the counts can be genotyped with method={quoted_value(method)}: "
                + (
                    "no site holds calls of the source"
                    if method == "source"
                    else "no site has one wild type and one alternative; name them "
                    "with alleles"
                )
            )
        if rules.min_depth is not None:
            unread = _sites_without_depth(group, table, genotyped)
            if unread:
                raise ValueError(
                    "min_depth tests each cell's depth at the site, which these counts "
                    f"cannot give at {len(unread)} genotyped "
                    f"{'site' if len(unread) == 1 else 'sites'}, such as "
                    f"{quoted_value(unread[0])}: no cell holds depth there or at the site's "
                    "channels, as when a source wrote calls alone; leave min_depth "
                    "out or name other sites"
                )
        cell_selection = inspect_artifact(self.zw, counts).input_ref("cell_selection")
        arguments = VariantGenotypeArguments(
            cell_selection=cell_selection,
            counts=counts,
            method=cast(Literal["source", "threshold", "gift"], rules.method),
            sites=_unit_labels(labels, genotyped, unit_alleles, site_alleles),
            alleles=tuple(site_alleles) if method != "source" else (),
            min_umis=rules.min_umis,
            min_mutant_umis=rules.min_mutant_umis,
            min_fraction=rules.min_fraction,
            min_posterior=rules.min_posterior,
            swap_probabilities=rules.swap_record,
            min_quality=rules.min_quality,
            min_depth=rules.min_depth,
            invalidate_cache=invalidate_cache,
        )
        planned = self._plan_variant_genotypes(arguments, cell_selection, n_cells)
        if planned.reused:
            # Saved genotypes warn as they did when they were called.
            warn_diagnostics(
                arguments,
                inspect_artifact(self.zw, planned.ref).diagnostics,
                _GENOTYPE_RULES,
            )
            return planned.ref
        self._require_writable(arguments.operation)
        tables: dict[str, tuple[str, ...]]
        if method == "source":
            tables = {
                "call": (
                    "call_alleles",
                    "call_sources",
                    "call_values",
                    *(("call_quality",) if rules.min_quality is not None else ()),
                ),
                "depth": ("depth_sites", "depth"),
            }
        elif method == "threshold":
            tables = {
                "count": ("count_alleles", "counts"),
                "depth": ("depth_sites", "depth"),
            }
        else:
            tables = {
                "count": (
                    "count_alleles",
                    "counts",
                    *(("counts_reads",) if rules.swap is not None else ()),
                )
            }
        # Every cell of the counts holds its offsets while bands are read, and
        # its call counts and offsets as they are written; each allele its
        # site, rank and gapfill probability; each call label its text;
        # appending a band rewrites the chunks it ends in; and Zarr's writes
        # hold their file buffers.
        n_alleles = table.n_alleles
        # Each call row's label refers to one of these texts rather than a copy.
        call_value_names = np.asarray(
            _read(group, "call_value_names").astype(str), dtype=object
        )
        most_rows = sum(
            int(as_zarr_array(group[names[0]], name=names[0]).shape[0])
            for names in tables.values()
        )
        held = (
            _GENOTYPE_FIXED_BYTES
            + (_BAND_BYTES_PER_COUNTS_CELL + _GENOTYPE_BYTES_PER_CELL) * n_cells
            + _GENOTYPE_BYTES_PER_ALLELE * n_alleles
            + sum(map(sys.getsizeof, call_value_names.tolist()))
            + _band_chunk_bytes(group, tables)
            + _REWRITTEN_CHUNKS * min(_ROW_BAND, max(1, most_rows)) * _STORED_BYTES
            + zarr_write_buffer_bytes()
        )
        site_rank, allele_rank, site_units, max_units = _unit_ranks(
            table, genotyped, unit_alleles
        )
        # A depth row of a site of several units stands for each of them.
        per_row = _GENOTYPE_BYTES_PER_ROW[method] * max_units
        max_rows = (self.memoryBytes - held) // per_row
        # A band holds at least _MIN_GENOTYPE_BAND_ROWS rows, or every row,
        # and every row of the cell that holds the most.
        fewest = max(1, min(most_rows, _MIN_GENOTYPE_BAND_ROWS))
        if max_rows >= fewest:
            fewest = max(fewest, _most_cell_rows(group, tables))
        if max_rows < fewest:
            raise MemoryError(
                f"Calling the genotypes of {n_cells} cells holds about "
                f"{held + fewest * per_row} bytes for a band of "
                f"{fewest} rows, more than the memory budget of {self.memoryBytes} "
                "bytes. Raise mem_budget."
            )
        cover_rank = _cover_ranks(table, site_rank)
        wild_type_probabilities = np.zeros(0)
        if method == "gift":
            # Each stored gapfill of a genotyped probe, against its probe's
            # wild type and mutant.
            of_probe = np.flatnonzero(allele_rank >= 0)
            wild_type_probabilities = np.zeros(n_alleles, dtype=np.float64)
            wild_type_probabilities[of_probe] = gapfill_wild_type_probabilities(
                [str(table.allele[allele]) for allele in of_probe],
                [site_alleles[allele_rank[allele]][0] for allele in of_probe],
                [site_alleles[allele_rank[allele]][1] for allele in of_probe],
            )
        # Each source column's place among a site's adopted columns, -1 where
        # the site does not adopt it.
        source_codes = _read(group, "call_source_names").astype(str)
        column_place = np.full((len(genotyped), len(source_codes)), -1, dtype=np.int64)
        for rank, columns in enumerate(site_columns):
            for place, column in enumerate(columns):
                column_place[rank, int(np.flatnonzero(source_codes == column)[0])] = (
                    place
                )
        # The labels the calls can take, coded in an order that the bands do
        # not change: the source's labels, or none.
        label_names = sorted(
            {""} | (set(call_value_names.tolist()) if method == "source" else set())
        )
        label_index = {text: code for code, text in enumerate(label_names)}
        n_mutant = np.zeros(n_cells, dtype=np.int32)
        n_called = np.zeros(n_cells, dtype=np.int32)
        offsets = np.zeros(n_cells + 1, dtype=np.int64)
        call_counts = np.zeros(len(CALLS), dtype=np.int64)
        positions = np.arange(n_cells, dtype=np.int64)
        with artifact_transaction(self.zw, planned) as out:
            appender = _Appender(
                out,
                {
                    "call_sites": (np.int32, most_rows),
                    "calls": (np.int8, most_rows),
                    "call_labels": (np.int32, most_rows),
                    "call_reference": (np.int32, most_rows),
                    "call_alternative": (np.int32, most_rows),
                    "posterior_wild_type": (np.float32, most_rows),
                    "posterior_heterozygous": (np.float32, most_rows),
                    "posterior_homozygous_alternative": (np.float32, most_rows),
                },
            )
            for band, rows in _variant_bands(group, tables, positions, int(max_rows)):
                first = int(band[0])
                width = int(band[-1]) - first + 1
                result = _band_genotypes(
                    rows,
                    method=method,
                    rules=rules,
                    first=first,
                    n_genotyped=len(genotyped),
                    cover_rank=cover_rank,
                    allele_rank=allele_rank,
                    reference_alleles=reference_alleles,
                    alternative_alleles=alternative_alleles,
                    wild_type_probabilities=wild_type_probabilities,
                    column_place=column_place,
                    source_columns=source_codes,
                    call_value_names=call_value_names,
                    site_units=site_units,
                )
                cells_of_rows = result.cells
                per_cell = np.bincount(cells_of_rows - first, minlength=width)
                offsets[first + 1 : first + width + 1] = offsets[first] + np.cumsum(
                    per_cell
                )
                mutant = np.isin(result.calls, list(MUTANT_CALLS))
                called = mutant | (result.calls == WILD_TYPE)
                n_mutant[first : first + width] += np.bincount(
                    cells_of_rows[mutant] - first, minlength=width
                ).astype(np.int32)
                n_called[first : first + width] += np.bincount(
                    cells_of_rows[called] - first, minlength=width
                ).astype(np.int32)
                call_counts += np.bincount(result.calls, minlength=len(CALLS))
                appender.append("call_sites", result.sites.astype(np.int32))
                appender.append("calls", result.calls.astype(np.int8))
                appender.append("call_labels", _label_codes(label_index, result.labels))
                appender.append("call_reference", result.reference.astype(np.int32))
                appender.append("call_alternative", result.alternative.astype(np.int32))
                for index, name in enumerate(
                    (
                        "posterior_wild_type",
                        "posterior_heterozygous",
                        "posterior_homozygous_alternative",
                    )
                ):
                    appender.append(
                        name, result.posteriors[:, index].astype(np.float32)
                    )
            _write_array(out, "n_mutant_sites", n_mutant)
            _write_array(out, "n_called_sites", n_called)
            _write_array(out, "call_offsets", offsets)
            _write_array(out, "genotype_sites", np.asarray(genotyped, dtype=np.int32))
            _write_array(out, "site_labels", np.asarray(arguments.sites, dtype=object))
            _write_array(
                out,
                "site_reference",
                np.asarray([pair[0] for pair in site_alleles], dtype=object),
            )
            _write_array(
                out,
                "site_alternative",
                np.asarray([pair[1] for pair in site_alleles], dtype=object),
            )
            _write_array(out, "site_sources", np.asarray(site_sources, dtype=object))
            _write_array(out, "call_label_names", np.asarray(label_names, dtype=object))
            out.attrs.update(
                {
                    DIAGNOSTICS_ATTRIBUTE: planned_diagnostics(
                        planned,
                        {
                            **{
                                _CALL_DIAGNOSTICS[name]: int(call_counts[code])
                                for code, name in enumerate(CALLS)
                            },
                            # Source calls compare no alleles of their own.
                            "alternativesNotShown": (
                                None
                                if method == "source"
                                else int(np.count_nonzero(alternative_alleles < 0))
                            ),
                        },
                    ),
                    "n_sites": len(genotyped),
                    "count_unit": str(group.attrs["count_unit"]),
                }
            )
        warn_diagnostics(
            arguments,
            inspect_artifact(self.zw, planned.ref).diagnostics,
            _GENOTYPE_RULES,
        )
        logger.info(
            f"Called the genotypes of {int(np.count_nonzero(n_called))} of {n_cells} "
            f"cells at {len(genotyped)} sites by method={quoted_value(method)}; "
            f"{int(np.count_nonzero(n_mutant))} cells carry a mutant call"
        )
        return planned.ref

    def _source_call_sites(
        self,
        group: zarr.Group,
        table: SiteTable,
        chosen: np.ndarray,
        explicit: bool,
        n_cells: int,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[str, ...]]]:
        """The units of the chosen sites that hold source calls, with their columns.

        A named site, such as a GoT target, is one unit. A site with
        coordinates gives one unit per alternative allele that holds calls,
        since Tapestri and DAb-seq call each variant of a position apart. A
        unit's columns come in the order of ``SOURCE_PRECEDENCE``, which a
        cell's call follows: the first column that holds one for the cell.

        Returns:
            Each unit's site, its allele or -1 for a whole site, and its
            call columns.
        """
        alleles = as_zarr_array(group["call_alleles"], name="call_alleles")
        sources = as_zarr_array(group["call_sources"], name="call_sources")
        names = _read(group, "call_source_names").astype(str)
        n_rows = int(alleles.shape[0])
        # The call rows are read in steps of at least _MIN_GENOTYPE_BAND_ROWS
        # rows, or all of them, beside the chunks that a step decompresses.
        held = _GENOTYPE_FIXED_BYTES + _band_chunk_bytes(
            group, {"call": ("call_alleles", "call_sources")}
        )
        step = (self.memoryBytes - held) // _SOURCE_SITE_BYTES_PER_ROW
        fewest = max(1, min(n_rows, _MIN_GENOTYPE_BAND_ROWS))
        if step < fewest:
            raise MemoryError(
                f"Finding the sites of the source calls of {n_cells} cells holds "
                f"about {held + fewest * _SOURCE_SITE_BYTES_PER_ROW} bytes, more "
                f"than the memory budget of {self.memoryBytes} bytes. Raise "
                "mem_budget."
            )
        allele_site = np.asarray(table.allele_site, dtype=np.int64)
        pairs: set[tuple[int, int]] = set()
        for start in range(0, n_rows, int(step)):
            stop = min(n_rows, start + int(step))
            allele_codes = np.asarray(alleles[start:stop], dtype=np.int64)
            codes = np.asarray(sources[start:stop], dtype=np.int64)
            # One integer per pair sorts fast, where unique rows sort slowly.
            keys = sorted_unique(allele_codes * len(names) + codes)
            pairs.update(
                (int(key) // len(names), int(key) % len(names)) for key in keys.tolist()
            )
        # The columns of each unit: (site, allele) at a site with coordinates,
        # (site, -1) at a named site.
        columns: dict[tuple[int, int], list[str]] = {}
        for allele, code in pairs:
            site = int(allele_site[allele])
            unit = (site, allele if table.site_contig[site] >= 0 else -1)
            columns.setdefault(unit, []).append(str(names[code]))
        units_of_site: dict[int, list[int]] = {}
        for site, allele in columns:
            units_of_site.setdefault(site, []).append(allele)
        labels = table.labels()
        kept: list[int] = []
        kept_alleles: list[int] = []
        chosen_columns: list[tuple[str, ...]] = []
        for site in chosen.tolist():
            if site not in units_of_site:
                if explicit:
                    raise ValueError(
                        f"The site {quoted_value(labels[site])} holds no call of the source, so "
                        "method='source' cannot genotype it; use method='threshold'"
                    )
                continue
            for allele in sorted(units_of_site[site]):
                ranked = sorted(
                    columns[(site, allele)],
                    key=lambda column: (
                        SOURCE_PRECEDENCE.index(column)
                        if column in SOURCE_PRECEDENCE
                        else len(SOURCE_PRECEDENCE),
                        column,
                    ),
                )
                unknown = [column for column in ranked if column not in SOURCE_CALLS]
                if unknown:
                    raise ValueError(
                        f"The calls of the column {quoted_value(unknown[0])} at the site "
                        f"{quoted_value(labels[site])} have no vocabulary; the known call "
                        f"columns are {', '.join(SOURCE_CALLS)}"
                    )
                kept.append(site)
                kept_alleles.append(allele)
                chosen_columns.append(tuple(ranked))
        return (
            np.asarray(kept, dtype=np.int64),
            np.asarray(kept_alleles, dtype=np.int64),
            chosen_columns,
        )

    def _plan_variant_genotypes(
        self,
        arguments: VariantGenotypeArguments,
        cell_selection: ArtifactRef,
        n_cells: int,
    ) -> PlannedArtifact:
        record = arguments.to_record()
        n_sites = len(arguments.sites)

        def has_calls(_ref: ArtifactRef, group: zarr.Group) -> bool:
            names = set(group.array_keys())
            long_arrays = {
                "call_sites",
                "calls",
                "call_labels",
                "call_reference",
                "call_alternative",
                "posterior_wild_type",
                "posterior_heterozygous",
                "posterior_homozygous_alternative",
            }
            site_arrays = {
                "genotype_sites",
                "site_labels",
                "site_reference",
                "site_alternative",
                "site_sources",
            }
            if (
                not (long_arrays | site_arrays | {"call_offsets", "call_label_names"})
                <= names
            ):
                return False
            if as_zarr_array(group["call_offsets"], name="call_offsets").shape != (
                n_cells + 1,
            ):
                return False
            lengths = {
                as_zarr_array(group[name], name=name).shape for name in long_arrays
            }
            sites = {
                as_zarr_array(group[name], name=name).shape for name in site_arrays
            }
            return len(lengths) == 1 and sites == {(n_sites,)}

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
                "n_mutant_sites": ((n_cells,), "i"),
                "n_called_sites": ((n_cells,), "i"),
            },
            invalidate_cache=arguments.invalidate_cache,
            required_attributes=("n_sites",),
            reuse_validator=has_calls,
        )

    @validation_scoped
    def _variants_load_genotypes(
        self,
        genotypes: ArtifactRef,
        *,
        sites: Sequence[str | int] | np.ndarray | pd.Index | None = None,
    ) -> VariantGenotypes:
        """Read saved genotype calls as tables of sites, cells and calls.

        A cell that the calls table does not list at a site holds no evidence
        there, the call ``missing``.

        Args:
            genotypes: A ``variant_genotypes`` artifact from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.genotypes`.
            sites: The genotyped sites to read the calls of, by label as the
                ``sites`` table names them, such as ``"CALR"`` or
                ``"chr1:115256513:G/T"``, or by position on counts of one
                contig, such as ``310``, which takes every genotyped site
                there; None for every genotyped site.

        Returns:
            A :class:`~cytearc.variants.VariantGenotypes` record.

        Raises:
            TypeError: If ``genotypes`` is not an ArtifactRef or ``sites`` not
                a sequence of labels or positions.
            ValueError: If ``genotypes`` is not a ``variant_genotypes``
                artifact, ``sites`` names a site it did not genotype, or a
                position is given for counts of several contigs.
            KeyError: If the store holds no such artifact, or not its counts
                when ``sites`` names a position.
            RuntimeError: If the artifact is incomplete, or its counts when
                ``sites`` names a position.
            MemoryError: If the tables do not fit the memory budget.
        """
        group = self._genotype_group(genotypes)
        site_labels = _read(group, "site_labels").astype(str)
        chosen = self._chosen_genotype_sites(genotypes, group, site_labels, sites)
        cell_index = cell_selection_indices(
            self.zw, inspect_artifact(self.zw, genotypes).input_ref("cell_selection")
        )
        n_cells = len(cell_index)
        n_rows = int(as_zarr_array(group["calls"], name="calls").shape[0])
        needed = (
            _LOADED_BYTES_PER_CELL * n_cells
            + text_read_bytes(self.cells._get_array("ids"), n_cells)
            + _LOADED_GENOTYPE_BYTES_PER_ROW * n_rows
        )
        if needed > self.memoryBytes:
            raise MemoryError(
                f"Loading the genotype calls of {n_cells} cells, {n_rows} rows, "
                f"holds about {needed} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget."
            )
        offsets = _read(group, "call_offsets").astype(np.int64)
        cells_of_rows = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(offsets))
        call_sites = _read(group, "call_sites").astype(np.int64)
        keep = np.isin(call_sites, chosen)
        ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", cell_index), dtype=object
        )
        # Each row's label refers to one of a few texts rather than a copy.
        label_names = np.asarray(
            _read(group, "call_label_names").astype(str), dtype=object
        )
        frames = genotype_frames(
            cell_ids=ids,
            site_labels=site_labels,
            site_reference=_read(group, "site_reference").astype(str),
            site_alternative=_read(group, "site_alternative").astype(str),
            site_sources=_read(group, "site_sources").astype(str),
            chosen=chosen,
            cells=cells_of_rows[keep],
            sites=call_sites[keep],
            calls=_read(group, "calls")[keep],
            labels=label_names[_read(group, "call_labels")[keep].astype(np.int64)]
            if len(label_names)
            else np.empty(0, dtype=object),
            reference=_read(group, "call_reference")[keep],
            alternative=_read(group, "call_alternative")[keep],
            posteriors=np.stack(
                [
                    _read(group, name)[keep]
                    for name in (
                        "posterior_wild_type",
                        "posterior_heterozygous",
                        "posterior_homozygous_alternative",
                    )
                ],
                axis=1,
            ),
        )
        status = inspect_artifact(self.zw, genotypes)
        # The method and its rules come from the provenance parameters, their
        # one home.
        parameters = dict(status.parameters or {})
        return VariantGenotypes(
            sites=frames["sites"],
            cells=frames["cells"],
            calls=frames["calls"],
            counts=status.input_ref("counts"),
            method=str(parameters["method"]),
            rules={name: parameters[name] for name in GENOTYPE_RULES},
        )

    def _chosen_genotype_sites(
        self,
        genotypes: ArtifactRef,
        group: zarr.Group,
        site_labels: np.ndarray,
        sites: Any,
    ) -> np.ndarray:
        """The genotyped sites that ``sites`` names, as ascending positions.

        A label names the genotyped site of that label. A position names every
        genotyped site at that position of the counts' one contig, so it is
        read from the counts only when ``sites`` holds one.
        """
        if sites is None:
            return np.arange(len(site_labels), dtype=np.int64)
        sites = _listed(sites)
        if isinstance(sites, str | bytes | int) or not isinstance(sites, Sequence):
            raise TypeError(
                "sites must be a sequence of site labels, such as ['CALR'], or of "
                "positions, or None"
            )
        by_label = {
            str(label): index for index, label in enumerate(site_labels.tolist())
        }
        unit_positions: np.ndarray | None = None
        has_contig: np.ndarray | None = None
        chosen: list[int] = []
        unknown: list[Any] = []
        for item in sites:
            if isinstance(item, str):
                found = [by_label[item]] if item in by_label else []
            elif isinstance(item, int | np.integer) and not isinstance(item, bool):
                if unit_positions is None or has_contig is None:
                    counts = inspect_artifact(self.zw, genotypes).input_ref("counts")
                    table = _stored_sites(self._variant_group(counts))
                    if len(table.contigs) != 1:
                        raise ValueError(
                            "Sites are named by position only on genotypes of counts "
                            "of one contig; name them by label, such as 'chrM:310'"
                        )
                    units = _read(group, "genotype_sites").astype(np.int64)
                    unit_positions = table.site_position[units]
                    # A named site, such as a GoT target, has no position.
                    has_contig = table.site_contig[units] >= 0
                found = np.flatnonzero(
                    has_contig & (unit_positions == int(item))
                ).tolist()
            else:
                raise TypeError(
                    f"sites must hold site labels or positions, not {type(item).__name__}"
                )
            if found:
                chosen.extend(found)
            else:
                unknown.append(item)
        if unknown:
            raise ValueError(
                "The genotypes have no site "
                + ", ".join(quoted_value(item) for item in unknown[:5])
            )
        if not chosen:
            raise ValueError("sites names no site")
        return sorted_unique(np.asarray(chosen, dtype=np.int64))

    @validation_scoped
    def _variants_genotype_clones(
        self,
        genotypes: ArtifactRef,
        *,
        sites: Sequence[str | int] | np.ndarray | pd.Index | None = None,
        within: str | Sequence[str] | None = None,
        method: str = "likelihood",
        dropout_probability: float | None = None,
        error_probability: float | None = None,
        min_size: int | None = None,
        min_posterior: float | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call clones from the combinations of mutations that cells carry, tolerating allelic dropout.

        Cells that descend from one mutated ancestor share its mutations.
        Each cell's calls at the chosen sites are mutant (heterozygous,
        homozygous alternative or alternative present), wild type, or
        unknown where the cell was ``missing`` or ``filtered``. Single-cell
        DNA amplification loses one allele of a cell at some sites, allelic
        dropout, which turns a heterozygous mutant call into a wild-type one,
        so two cells of one clone can disagree.

        ``method="likelihood"``, the default, fits SCClone's mixture model
        (Yu, Du and Song, Frontiers in Genetics 2022,
        doi:10.3389/fgene.2022.823941): each clone is a set of mutations, a
        cell of a clone reads wild type at one of its mutations with the
        site's dropout probability and mutant at another site with the
        site's error probability, and an unknown call adds nothing. Each
        site's two probabilities are estimated from the cells by EM, shrunk
        toward the probabilities pooled over every site by priors worth 20
        and 200 calls, a site's error probability stays at most five times
        the pooled one, and the pooled one at most 0.1, so that a mutation
        that every cell carries cannot pass for errors;
        ``dropout_probability`` or ``error_probability`` holds one
        probability at every site instead. A greedy search chooses the
        number of clones by Schwarz's criterion, in which a clone costs its
        weight and the sites at which it differs from its neighbour on a
        minimum spanning tree of the clones rooted at the wild type. The genotype without mutations is always a
        component: its cells, like cells without a mutant call, take no
        clone. Each cell takes its most probable clone, its posterior saved
        as ``clone_probability``; a cell with a mutant call whose most
        probable component has a posterior below ``min_posterior`` is
        ambiguous and takes none, and a clone of fewer than ``min_size``
        cells leaves its cells without one. Every cell with a call at the
        chosen sites takes part in the fit. A subclone that differs from
        its parent at one site cannot be told from that site's dropout by
        calls alone: both explanations fit the calls equally, so such a
        subclone may join its parent.

        ``method="exact"`` links two cells when they agree at every site
        that both were called at, a Jaccard index of 1 over the sites both
        know. Cells whose calls agree with two cells that disagree at a site
        they were not called at cannot be placed and join neither: they are
        left out as ambiguous, however many there are. The linked groups are
        the clones, and a cell without a mutant call gets no clone. One
        dropout splits a cell from its clone under this rule.

        Clones are numbered by size as ``clone_1`` and on. ``within`` names
        cell metadata columns, such as the donor: clones never join cells of
        two groups, and the likelihood fits each group apart, with its own
        rates. The artifact's diagnostics, which
        ``ds.artifacts.inspect(clones).diagnostics`` returns, count the
        ``sites`` compared, the ``mutantCells``, the ``ambiguousCells`` and,
        for the likelihood, the ``cellsInSmallClones``, and record the pooled
        ``dropoutProbability`` and ``errorProbability`` that the fit used,
        weighed over the groups by the calls that estimate them, beside
        every source's ``cellsWithClone`` and
        ``largestCloneShare``; the rule that does not run records None. The
        likelihood also saves each site's label and each group's dropout and
        error probability at each site.

        Args:
            genotypes: A ``variant_genotypes`` artifact from
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.genotypes`.
            sites: The genotyped sites whose calls define the clones, by label
                or by position on counts of one contig, as
                :meth:`~cytearc.datastore.namespaces.VariantsAccessor.load_genotypes`
                takes them, or None for every genotyped site.
            within: Cell metadata columns to call clones within, or None;
                every cell with a call, for ``"likelihood"``, or with a mutant
                call, for ``"exact"``, needs a label.
            method: ``"likelihood"`` or ``"exact"``.
            dropout_probability: For ``"likelihood"``, the chance that a cell
                carrying a site's mutation reads wild type there, held at every
                site, from 0 up to 1; None to estimate each site's.
            error_probability: For ``"likelihood"``, the chance that a cell
                without a site's mutation reads mutant there, held at every
                site, from 0 up to 1; None to estimate each site's.
            min_size: For ``"likelihood"``, the fewest cells of a clone, None
                for 2.
            min_posterior: For ``"likelihood"``, the posterior that a cell's
                most probable component must reach, else the cell is
                ambiguous, from 0 to 1; None for 0.6.
            invalidate_cache: Call the clones again instead of reusing them.

        Returns:
            A complete datastore-scoped ``clones`` artifact over the
            genotypes' cells, which every clone summary reads. Its
            ``ambiguous`` array marks the cells left out as ambiguous and,
            for the likelihood, its ``clone_probability`` array holds each
            cell's posterior of its clone. Read it with
            :meth:`~cytearc.datastore.namespaces.ClonesAccessor.load_clones`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``genotypes`` is not a ``variant_genotypes``
                artifact, ``sites`` names a site it did not genotype or a
                position of counts of several contigs, ``method`` is unknown,
                a rule is given to ``"exact"`` or is out of range, ``within``
                names a column twice or no cell metadata column, or a cell
                that needs a label has none in it.
            KeyError: If the store holds no such artifact, or not its counts
                when ``sites`` names a position.
            RuntimeError: If the artifact is incomplete, or its counts when
                ``sites`` names a position.
            PermissionError: If the store is read-only and the clones are not
                saved yet.
            MemoryError: If reading the calls, fitting or linking the cells,
                or saving the clone labels does not fit the memory budget.
        """
        from ...variants.genotype_clones import genotype_clone_rules
        from ...variants.genotype_mixture import (
            DROPOUT_PRIOR_CALLS,
            ERROR_PRIOR_CALLS,
            MAX_POOLED_ERROR,
            MAX_SITE_ERROR_RATIO,
            any_bits,
        )

        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        rules = genotype_clone_rules(
            method,
            dropout_probability=dropout_probability,
            error_probability=error_probability,
            min_size=min_size,
            min_posterior=min_posterior,
        )
        likelihood = rules.method == "likelihood"
        group = self._genotype_group(genotypes)
        columns = group_columns(within, "within")
        cell_selection = inspect_artifact(self.zw, genotypes).input_ref(
            "cell_selection"
        )
        n_cells = (
            int(as_zarr_array(group["call_offsets"], name="call_offsets").shape[0]) - 1
        )
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Calling clones from the genotypes of {n_cells} cells",
            codec_threads=self.nthreads,
        )
        ledger.hold(
            "Zarr's objects for the arrays that the call opens and writes",
            _GENOTYPE_FIXED_BYTES,
        )
        site_labels = read_whole(ledger, group, "site_labels", item="the sites' labels")
        chosen = self._chosen_genotype_sites(genotypes, group, site_labels, sites)
        n_sites = len(chosen)
        calls = self._packed_genotype_calls(
            ledger, group, len(site_labels), chosen, n_cells
        )
        flags_held = ledger.hold(
            "whether each cell has a call and a mutant call", 3 * n_cells
        )
        has_mutant = np.empty(n_cells, dtype=bool)
        has_wild = np.empty(n_cells, dtype=bool)
        with numba_threads(self.nthreads):
            any_bits(calls.mutant, has_mutant)
            any_bits(calls.wild, has_wild)
        members = has_mutant | has_wild if likelihood else has_mutant.copy()
        del has_wild
        group_codes, group_keys, within_fingerprint = within_groups(
            self.zw,
            self.cells,
            cell_selection,
            columns,
            members,
            ledger=ledger,
            what="a call" if likelihood else "a mutant call",
        )
        arguments = GenotypeCloneArguments(
            cell_selection=cell_selection,
            genotypes=genotypes,
            within_fingerprint=within_fingerprint,
            sites=tuple(str(site_labels[site]) for site in chosen),
            within=columns,
            method="likelihood" if likelihood else "exact",
            dropout_probability=rules.dropout_probability,
            error_probability=rules.error_probability,
            min_size=rules.min_size,
            min_posterior=rules.min_posterior,
            dropout_prior_calls=DROPOUT_PRIOR_CALLS if likelihood else None,
            error_prior_calls=ERROR_PRIOR_CALLS if likelihood else None,
            max_site_error_ratio=MAX_SITE_ERROR_RATIO if likelihood else None,
            max_pooled_error=MAX_POOLED_ERROR if likelihood else None,
            similarity=None if likelihood else 1.0,
            inclusive=None if likelihood else True,
            ambiguous=None if likelihood else True,
            invalidate_cache=invalidate_cache,
        )
        planned = plan_clones(
            self.zw,
            arguments,
            cell_selection,
            n_cells,
            with_probability=likelihood,
            required_attributes=(
                _LIKELIHOOD_CLONE_ATTRIBUTES if likelihood else _EXACT_CLONE_ATTRIBUTES
            ),
            source_arrays={"ambiguous": "b"},
            site_arrays=SITE_ARRAYS if likelihood else (),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)
        n_mutant_cells = int(np.count_nonzero(has_mutant))
        del has_mutant
        labels_text = np.asarray(site_labels, dtype=object)[chosen]
        site_arrays: dict[str, np.ndarray] = {}
        if likelihood:
            called = self._likelihood_genotype_clones(
                ledger, calls, members, group_codes, len(group_keys), n_sites, rules
            )
            assignment = called.assignment
            ambiguous = called.ambiguous
            attributes: dict[str, Any] = {"n_clones": len(assignment.clone_ids)}
            site_arrays = {
                "site_labels": labels_text,
                "dropout_probability": called.dropout,
                "error_probability": called.error,
            }
            own = {
                "cellsInSmallClones": called.cells_in_small_clones,
                "dropoutProbability": called.dropout_probability,
                "errorProbability": called.error_probability,
            }
        else:
            assignment, ambiguous, attributes = self._exact_genotype_clones(
                ledger, calls, members, group_codes, n_sites
            )
            own = {
                "cellsInSmallClones": None,
                "dropoutProbability": None,
                "errorProbability": None,
            }
        del members, group_codes
        ledger.hold(
            "the clones' labels and the sites' labels",
            sum(map(sys.getsizeof, assignment.clone_ids.tolist()))
            + sum(map(sys.getsizeof, labels_text.tolist())),
        )
        write_clones(
            self.zw,
            planned,
            assignment,
            attributes,
            source_arrays={"ambiguous": ambiguous},
            site_arrays=site_arrays,
            ledger=ledger,
            diagnostics={
                "sites": n_sites,
                "mutantCells": n_mutant_cells,
                "ambiguousCells": int(np.count_nonzero(ambiguous)),
                **own,
            },
        )
        ledger.release(flags_held)
        log_clones(assignment, n_cells)
        return planned.ref

    def _packed_genotype_calls(
        self,
        ledger: MemoryLedger,
        group: zarr.Group,
        n_labels: int,
        chosen: np.ndarray,
        n_cells: int,
    ) -> "_PackedCalls":
        """Each cell's mutant and wild-type calls at the chosen sites, packed 64 sites to a word, read in bands of cells.

        The packed calls stay charged to ``ledger`` until the caller lets
        them go.
        """
        from ...variants.genotype_mixture import n_words, pack_calls

        words = n_words(len(chosen))
        held = ledger.hold(
            f"the calls of {n_cells} cells at {len(chosen)} sites, packed",
            16 * n_cells * words,
        )
        calls = _PackedCalls(
            np.zeros((n_cells, words), dtype=np.uint64),
            np.zeros((n_cells, words), dtype=np.uint64),
            held,
        )
        # Each stored site's column among the chosen ones, whether each call
        # is mutant, and what the bands read for every cell.
        lookup = ledger.hold(
            "each stored site's chosen column",
            8 * n_labels + _CALL_BAND_BYTES_PER_CELL * n_cells + 64,
        )
        rank = np.full(n_labels, -1, dtype=np.int64)
        rank[chosen] = np.arange(len(chosen))
        mutant = np.zeros(len(CALLS), dtype=bool)
        mutant[sorted(MUTANT_CALLS)] = True
        positions = np.arange(n_cells, dtype=np.int64)
        tables = {"call": ("call_sites", "calls")}
        n_rows = int(as_zarr_array(group["calls"], name="calls").shape[0])
        # A band holds each row's cell beside its site and call as stored,
        # and the chunks at its ends that its read decompresses. It holds at
        # least _MIN_GENOTYPE_BAND_ROWS rows, or every row, and every row of
        # the cell that holds the most.
        per_row = 8 + sum(
            int(as_zarr_array(group[name], name=name).dtype.itemsize)
            for name in tables["call"]
        )
        chunk_bytes = _band_chunk_bytes(group, tables)
        fewest = max(
            1, min(n_rows, _MIN_GENOTYPE_BAND_ROWS), _most_cell_rows(group, tables)
        )
        ledger.peak("a band of the genotype calls", chunk_bytes + fewest * per_row)
        # The room that the budget leaves sizes the bands.
        max_rows = (self.memoryBytes - ledger.held_bytes - chunk_bytes) // per_row
        for _band, rows in _variant_bands(group, tables, positions, max_rows):
            cells_of_rows, values = rows["call"]
            pack_calls(
                cells_of_rows,
                values["call_sites"],
                values["calls"],
                rank,
                mutant,
                calls.mutant,
                calls.wild,
            )
            del cells_of_rows, values, rows
        del rank, mutant, positions
        ledger.release(lookup)
        return calls

    def _likelihood_genotype_clones(
        self,
        ledger: MemoryLedger,
        calls: "_PackedCalls",
        members: np.ndarray,
        group_codes: np.ndarray,
        n_groups: int,
        n_sites: int,
        rules: Any,
    ) -> "_LikelihoodCall":
        """Fit each within group's cells apart and number their clones by size over every group.

        The packed calls of the fitted cells are gathered in order of their
        groups, unless every cell is fitted in one group, and the packed
        calls are let go once every group is fitted.
        """
        from ...variants.genotype_clones import likelihood_clones

        n_cells = len(members)
        words = calls.mutant.shape[1]
        n_fitted = int(np.count_nonzero(members))
        if n_fitted == n_cells and n_groups == 1:
            order_held = ledger.hold("each fitted cell's position", 8 * n_fitted + 16)
            fitted = np.flatnonzero(members)
            bounds = np.array([0, n_cells], dtype=np.int64)
        else:
            order_held = ledger.hold(
                f"the calls of {n_fitted} cells in order of their groups",
                16 * n_fitted * words + 8 * n_fitted + 8 * (n_groups + 1),
            )
            ledger.peak("ordering the cells by group", _GROUP_ORDER_BYTES * n_fitted)
            fitted, bounds = _cells_by_group(members, group_codes, n_groups)
            calls.gather(ledger, fitted)
        # Each cell's clone within its group, from 1, its posterior and its
        # flag, and each group's rates at each site.
        ledger.hold(
            f"the clones of {n_cells} cells",
            17 * n_cells + 16 * n_groups * n_sites,
        )
        clusters = np.zeros(n_cells, dtype=np.int64)
        probability = np.full(n_cells, np.nan)
        ambiguous = np.zeros(n_cells, dtype=bool)
        dropout = np.full((n_groups, n_sites), np.nan)
        error = np.full((n_groups, n_sites), np.nan)
        # Each group's pooled rates, weighed by the calls that estimate them.
        weighed = np.zeros(4)
        cells_in_small_clones = 0
        offset = 0
        for index in range(n_groups):
            first, stop = int(bounds[index]), int(bounds[index + 1])
            if first == stop:
                continue
            group_cells = fitted[first:stop]
            result = likelihood_clones(
                calls.mutant[first:stop],
                calls.wild[first:stop],
                n_sites,
                dropout_probability=rules.dropout_probability,
                error_probability=rules.error_probability,
                min_size=rules.min_size,
                min_posterior=rules.min_posterior,
                ledger=ledger,
                nthreads=self.nthreads,
            )
            ledger.peak(
                f"placing the clones of {stop - first} cells", 41 * (stop - first)
            )
            clustered = result.codes >= 0
            clusters[group_cells[clustered]] = result.codes[clustered] + 1 + offset
            probability[group_cells[clustered]] = result.probability[clustered]
            ambiguous[group_cells] = result.ambiguous
            offset += len(result.sizes)
            dropout[index] = result.dropout
            error[index] = result.error
            weighed += (
                result.pooled_dropout * result.carried_calls,
                result.carried_calls,
                result.pooled_error * result.absent_calls,
                result.absent_calls,
            )
            cells_in_small_clones += result.cells_in_small_clones
            held = result.held_bytes
            del clustered, group_cells, result
            ledger.release(held)
        del fitted, bounds
        calls.let_go(ledger)
        ledger.release(order_held)
        # Clones are numbered by size over every group, ties by their first
        # cell, as frequency clones are.
        ledger.peak(
            f"numbering the clones of {n_cells} cells", _CLONE_RANK_BYTES * n_cells
        )
        positions = np.flatnonzero(clusters > 0)
        assignment = _clone_assignment(
            n_cells,
            positions,
            groups=np.zeros(len(positions), dtype=np.int64),
            clusters=clusters[positions],
        )
        del positions, clusters
        assignment = replace(
            assignment,
            probabilities=np.where(assignment.codes >= 0, probability, np.nan),
        )
        del probability
        # Each cell's code and probability take the place of its clone
        # within its group and its posterior, which the results' charge
        # holds, and the clones' sizes and labels come beside them.
        ledger.hold(
            f"the sizes of {len(assignment.clone_ids)} clones",
            24 * len(assignment.clone_ids),
        )
        return _LikelihoodCall(
            assignment=assignment,
            ambiguous=ambiguous,
            dropout=dropout,
            error=error,
            cells_in_small_clones=cells_in_small_clones,
            dropout_probability=(
                rules.dropout_probability
                if rules.dropout_probability is not None
                else (float(weighed[0] / weighed[1]) if weighed[1] > 0 else None)
            ),
            error_probability=(
                rules.error_probability
                if rules.error_probability is not None
                else (float(weighed[2] / weighed[3]) if weighed[3] > 0 else None)
            ),
        )

    def _exact_genotype_clones(
        self,
        ledger: MemoryLedger,
        calls: "_PackedCalls",
        members: np.ndarray,
        group_codes: np.ndarray,
        n_sites: int,
    ) -> tuple[CloneAssignment, np.ndarray, dict[str, Any]]:
        """Link the cells that agree at every site both were called at, with each mutant cell's unread sites as unknown items.

        The packed calls are let go once each mutant cell's mutant and
        unknown sites are listed.
        """
        from ...variants.genotype_mixture import bit_pairs

        n_cells = len(members)
        ledger.peak("finding the mutant cells", 8 * n_cells)
        member_cells = np.flatnonzero(members)
        held = ledger.hold("the mutant cells", member_cells.nbytes)
        mutant_cells, mutant_sites, mutant_held = bit_pairs(
            ledger, calls.mutant, calls.wild, member_cells, n_sites, unknown=False
        )
        unknown_cells, unknown_sites, unknown_held = bit_pairs(
            ledger, calls.mutant, calls.wild, member_cells, n_sites, unknown=True
        )
        del member_cells
        calls.let_go(ledger)
        ledger.release(held)
        mutant_sets = cell_sets(mutant_cells, mutant_sites, n_cells, ledger=ledger)
        sets_held = ledger.hold("each cell's mutant sites", mutant_sets.nbytes)
        del mutant_cells, mutant_sites
        ledger.release(mutant_held)
        unknown = cell_sets(unknown_cells, unknown_sites, n_cells, ledger=ledger)
        sets_held += ledger.hold("each mutant cell's unknown sites", unknown.nbytes)
        del unknown_cells, unknown_sites
        ledger.release(unknown_held)
        called = set_clones(
            mutant_sets,
            unknown=unknown,
            groups=group_codes,
            method="jaccard",
            similarity=1.0,
            inclusive=True,
            ambiguous=True,
            ledger=ledger,
        )
        del mutant_sets, unknown
        ledger.release(sets_held)
        labels = ranked_clone_labels(len(called.sizes))
        assignment = CloneAssignment(
            codes=called.codes,
            clone_ids=labels,
            clone_sizes=called.sizes,
            probabilities=None,
            n_cells_in_several_clones=0,
        )
        # The sizes of the graph stay attributes; they describe the method.
        attributes = {
            "n_clones": len(labels),
            "n_nodes": called.n_nodes,
            "n_edges": called.n_edges,
        }
        return assignment, called.ambiguous.astype(bool, copy=False), attributes


def _cells_by_group(
    members: np.ndarray, group_codes: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """The positions of the member cells in order of their groups, cells in order within each, and where each group starts and stops among them.

    Beside the positions, this holds ``_GROUP_ORDER_BYTES`` a member cell.
    """
    fitted = np.flatnonzero(members)
    fitted_groups = group_codes[fitted]
    order = np.argsort(fitted_groups, kind="stable")
    fitted = fitted[order]
    bounds = np.searchsorted(fitted_groups[order], np.arange(n_groups + 1), side="left")
    return fitted, bounds


@dataclass(slots=True)
class _PackedCalls:
    """Each cell's mutant and wild-type calls packed 64 sites to a word, with the bytes that they are charged as.

    One holder owns the packed calls, so that letting them go frees them.
    """

    mutant: np.ndarray
    wild: np.ndarray
    held: int

    def gather(self, ledger: MemoryLedger, rows: np.ndarray) -> None:
        """Keep the calls of ``rows`` alone, in their order; the caller has charged them."""
        self.mutant = self.mutant[rows]
        self.wild = self.wild[rows]
        ledger.release(self.held)
        self.held = 0

    def let_go(self, ledger: MemoryLedger) -> None:
        """Let the packed calls go and release their charge."""
        self.mutant = self.wild = np.empty((0, 1), dtype=np.uint64)
        ledger.release(self.held)
        self.held = 0


@dataclass(frozen=True, slots=True)
class _LikelihoodCall:
    """The clones that the likelihood calls over every within group, with what the artifact saves of them."""

    assignment: CloneAssignment
    ambiguous: np.ndarray
    dropout: np.ndarray
    error: np.ndarray
    cells_in_small_clones: int
    dropout_probability: float | None
    error_probability: float | None


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
    n_cells: int, positions: np.ndarray, *, groups: np.ndarray, clusters: np.ndarray
) -> CloneAssignment:
    """Number called clones by size over every group, ties by their first cell.

    The clones are labelled ``clone_1`` and on by rank, as every clone that
    CyteArc calls is.

    Args:
        n_cells: The cells of the artifact.
        positions: The called cells, as ascending positions among them.
        groups: Each called cell's group.
        clusters: Each called cell's clone within its group, from 1, or 0
            for none.
    """
    clustered = np.asarray(clusters) > 0
    cells = np.asarray(positions, dtype=np.int64)[clustered]
    clusters = np.asarray(clusters, dtype=np.int64)[clustered]
    keys = (
        np.asarray(groups, dtype=np.int64)[clustered]
        * (int(clusters.max(initial=0)) + 1)
        + clusters
    )
    # A clone's first index among the clustered cells is its first cell, since
    # the positions ascend.
    _distinct, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    sizes = np.bincount(inverse.reshape(-1), minlength=len(first)).astype(np.int64)
    order = np.lexsort((first, -sizes))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[cells] = rank[inverse.reshape(-1)]
    return CloneAssignment(
        codes=codes,
        clone_ids=ranked_clone_labels(len(order)),
        clone_sizes=sizes[order],
        probabilities=None,
        n_cells_in_several_clones=0,
    )


def _compared_alleles(
    table: SiteTable, chosen: np.ndarray, explicit: bool, rules: GenotypeRules
) -> tuple[np.ndarray, list[tuple[str, str]], np.ndarray, np.ndarray]:
    """The chosen sites that a counting method can genotype, with their alleles.

    A site is compared at its wild type and its one alternative: those that
    ``rules.alleles`` names for it, or its alleles of roles ``"reference"``
    and ``"alternative"``. Each allele's position among the counts' alleles
    is -1 where the counts hold no such allele at the site: an alternative
    that no cell shows, which the genotypes count and warn of.

    Raises:
        ValueError: If a site that ``explicit`` sites name has no wild type
            or not one alternative, or ``rules.alleles`` names a wild type
            that no cell shows at its site.
    """
    labels = table.labels()
    allele_site = np.asarray(table.allele_site, dtype=np.int64)
    roles = np.asarray(table.allele_role, dtype=object)
    texts = np.asarray(table.allele, dtype=object)
    kept: list[int] = []
    pairs: list[tuple[str, str]] = []
    reference: list[int] = []
    alternative: list[int] = []
    for site in chosen.tolist():
        own = np.flatnonzero(allele_site == site)
        label = str(labels[site])
        if label in rules.alleles:
            wild_type, mutant = rules.alleles[label]
        else:
            wild = own[roles[own] == "reference"]
            mutants = own[roles[own] == "alternative"]
            if len(wild) != 1 or len(mutants) != 1:
                if explicit:
                    raise ValueError(
                        f"The site {quoted_value(label)} has {len(wild)} wild types and "
                        f"{len(mutants)} alternatives in the counts; name its "
                        "wild type and alternative with alleles"
                    )
                continue
            wild_type, mutant = str(texts[wild[0]]), str(texts[mutants[0]])
        positions = {str(texts[allele]): int(allele) for allele in own}
        if label in rules.alleles:
            if wild_type not in positions:
                raise ValueError(
                    f"alleles gives the site {quoted_value(label)} the wild type {quoted_value(wild_type)}, "
                    "which no cell shows there; its alleles are "
                    f"{', '.join(map(repr, positions)) or 'none'}, in the "
                    "orientation that the files give, such as a GIFT probe's"
                )
        kept.append(site)
        pairs.append((wild_type, mutant))
        reference.append(positions.get(wild_type, -1))
        alternative.append(positions.get(mutant, -1))
    return (
        np.asarray(kept, dtype=np.int64),
        pairs,
        np.asarray(reference, dtype=np.int64),
        np.asarray(alternative, dtype=np.int64),
    )


def _unit_alleles(
    table: SiteTable, genotyped: np.ndarray, unit_alleles: np.ndarray
) -> list[tuple[str, str]]:
    """Each unit's reference and alternative: empty for a whole site."""
    references = table.reference_alleles()
    pairs: list[tuple[str, str]] = []
    for site, allele in zip(genotyped.tolist(), unit_alleles.tolist(), strict=True):
        if allele < 0:
            pairs.append(("", ""))
            continue
        reference = int(references[site])
        pairs.append(
            (
                str(table.allele[reference]) if reference >= 0 else "",
                str(table.allele[allele]),
            )
        )
    return pairs


def _unit_labels(
    labels: np.ndarray,
    genotyped: np.ndarray,
    unit_alleles: np.ndarray,
    site_alleles: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Each unit's label: its site's, and for one allele of a site with
    coordinates, its reference and alternative as Tapestri names variants,
    such as ``chr1:115256513:G/A``."""
    named: list[str] = []
    for site, allele, (reference, alternative) in zip(
        genotyped.tolist(), unit_alleles.tolist(), site_alleles, strict=True
    ):
        label = str(labels[site])
        if allele < 0:
            named.append(label)
        elif reference:
            named.append(f"{label}:{reference}/{alternative}")
        else:
            named.append(f"{label}:{alternative}")
    return tuple(named)


def _unit_ranks(
    table: SiteTable, genotyped: np.ndarray, unit_alleles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None, int]:
    """Where each site's and allele's evidence goes among the genotyped units.

    A whole site's unit takes every allele of the site, and a unit of one
    allele that allele. A site of one unit ranks it; a site of several
    units ranks none and lists them, since its depth covers each.

    Returns:
        Each site's unit or -1, each allele's unit or -1, the units of the
        sites of several as offsets over the sites and the units, or None,
        and the most units of a site.
    """
    genotyped = np.asarray(genotyped, dtype=np.int64)
    unit_alleles = np.asarray(unit_alleles, dtype=np.int64)
    ranks = np.arange(len(genotyped), dtype=np.int64)
    whole = unit_alleles < 0
    whole_rank = np.full(table.n_sites, -1, dtype=np.int64)
    whole_rank[genotyped[whole]] = ranks[whole]
    allele_rank = whole_rank[np.asarray(table.allele_site, dtype=np.int64)]
    allele_rank[unit_alleles[~whole]] = ranks[~whole]
    per_site = np.bincount(genotyped, minlength=table.n_sites)
    single = per_site[genotyped] == 1
    site_rank = np.full(table.n_sites, -1, dtype=np.int64)
    site_rank[genotyped[single]] = ranks[single]
    site_units: tuple[np.ndarray, np.ndarray] | None = None
    if not single.all():
        counts = np.where(per_site > 1, per_site, 0)
        offsets = np.zeros(table.n_sites + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        order = np.lexsort((ranks[~single], genotyped[~single]))
        site_units = (offsets, ranks[~single][order])
    return site_rank, allele_rank, site_units, int(per_site.max(initial=1))


def _sites_without_depth(
    group: zarr.Group, table: SiteTable, sites: np.ndarray
) -> list[str]:
    """The labels of the sites among ``sites`` at which no cell holds depth.

    A named target's channels, such as GoT-Multi's ``BCOR (geneseq)``, read
    its evidence, so depth at a channel covers its target.
    """
    read = _read(group, "site_cells").astype(np.int64) > 0
    names = [str(name) for name in table.site_name]
    channels = [str(channel) for channel in table.site_channel]
    read_targets = {
        name
        for name, channel, covered in zip(names, channels, read.tolist(), strict=True)
        if channel and covered
    }
    labels = table.labels()
    return [
        str(labels[site])
        for site in sorted_unique(np.asarray(sites, dtype=np.int64)).tolist()
        if not read[site] and not (names[site] in read_targets and not channels[site])
    ]


def _cover_ranks(table: SiteTable, site_rank: np.ndarray) -> np.ndarray:
    """Each site's genotyped site whose evidence its reads are, or -1.

    A genotyped site's reads are its own; a channel of a named target, such
    as GoT-Multi's ``BCOR (geneseq)``, that is not genotyped itself reads
    the evidence of its target, which holds the calls.
    """
    cover = np.asarray(site_rank, dtype=np.int64).copy()
    names = [str(name) for name in table.site_name]
    channels = [str(channel) for channel in table.site_channel]
    targets = {
        name: site
        for site, (name, channel) in enumerate(zip(names, channels, strict=True))
        if name and not channel
    }
    for site, (name, channel) in enumerate(zip(names, channels, strict=True)):
        if channel and cover[site] < 0 and name in targets:
            cover[site] = site_rank[targets[name]]
    return cover


def _label_codes(index: Mapping[str, int], labels: np.ndarray) -> np.ndarray:
    """Each label's position in the vocabulary that ``index`` numbers."""
    inverse, distinct = _factorized(labels)
    lookup = np.asarray(
        [index[str(text)] for text in distinct.tolist()], dtype=np.int32
    )
    return np.asarray(lookup[inverse], dtype=np.int32)


def _most_cell_rows(group: zarr.Group, tables: Mapping[str, Sequence[str]]) -> int:
    """The most rows that one cell holds in ``tables`` together."""
    combined = np.sum([_read(group, f"{name}_offsets") for name in tables], axis=0)
    return int(np.diff(combined).max(initial=0))


@dataclass(frozen=True, slots=True)
class _BandCalls:
    """The calls of one band of cells, sorted by cell and then site."""

    cells: np.ndarray
    sites: np.ndarray
    calls: np.ndarray
    labels: np.ndarray
    reference: np.ndarray
    alternative: np.ndarray
    posteriors: np.ndarray


def _band_keys(
    cells: np.ndarray, ranks: np.ndarray, first: int, n_genotyped: int
) -> np.ndarray:
    """Each row's cell and genotyped site as one sortable key."""
    return np.asarray(
        (cells.astype(np.int64) - first) * n_genotyped + ranks, dtype=np.int64
    )


def _covered_keys(
    cells: np.ndarray,
    sites: np.ndarray,
    chosen: np.ndarray,
    *,
    cover_rank: np.ndarray,
    first: int,
    n_genotyped: int,
    site_units: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """The keys of the genotyped units that the ``chosen`` depth rows cover.

    A row covers its site's unit, or each unit of a site of several units
    that ``site_units`` lists.

    Returns:
        The distinct keys, ascending.
    """
    ranks = cover_rank[sites]
    on = (ranks >= 0) & chosen
    keys = _band_keys(cells[on], ranks[on], first, n_genotyped)
    if site_units is not None:
        offsets, units = site_units
        width = (offsets[sites + 1] - offsets[sites]) * chosen
        if width.any():
            ends = np.cumsum(width)
            within = np.arange(int(ends[-1]), dtype=np.int64) - np.repeat(
                ends - width, width
            )
            keys = np.concatenate(
                [
                    keys,
                    _band_keys(
                        np.repeat(cells, width),
                        units[np.repeat(offsets[sites], width) + within],
                        first,
                        n_genotyped,
                    ),
                ]
            )
    return sorted_unique(keys)


def _band_genotypes(
    rows: Mapping[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    *,
    method: str,
    rules: GenotypeRules,
    first: int,
    n_genotyped: int,
    cover_rank: np.ndarray,
    allele_rank: np.ndarray,
    reference_alleles: np.ndarray,
    alternative_alleles: np.ndarray,
    wild_type_probabilities: np.ndarray,
    column_place: np.ndarray,
    source_columns: np.ndarray,
    call_value_names: np.ndarray,
    site_units: tuple[np.ndarray, np.ndarray] | None = None,
) -> _BandCalls:
    """Call the genotypes of a band's cells at the genotyped units.

    ``site_units`` lists the units of the sites of several, each of which a
    depth row of the site covers. A cell's adopted source call is filtered
    where its quality or depth falls below the rules' ``min_quality`` or
    ``min_depth``.
    """
    covered = np.empty(0, dtype=np.int64)
    deep = np.empty(0, dtype=np.int64)
    if "depth" in rows:
        depth_cells, depth = rows["depth"]
        depth_sites = depth["depth_sites"].astype(np.int64)
        covered = _covered_keys(
            depth_cells,
            depth_sites,
            depth["depth"] > 0,
            cover_rank=cover_rank,
            first=first,
            n_genotyped=n_genotyped,
            site_units=site_units,
        )
        if rules.min_depth is not None:
            deep = _covered_keys(
                depth_cells,
                depth_sites,
                depth["depth"] >= rules.min_depth,
                cover_rank=cover_rank,
                first=first,
                n_genotyped=n_genotyped,
                site_units=site_units,
            )
    labels: np.ndarray
    if method == "source":
        call_cells, values = rows["call"]
        ranks = allele_rank[values["call_alleles"].astype(np.int64)]
        columns = values["call_sources"].astype(np.int64)
        place = np.full(len(ranks), -1, dtype=np.int64)
        on = ranks >= 0
        place[on] = column_place[ranks[on], columns[on]]
        adopted = np.flatnonzero(place >= 0)
        # Each cell's call at a site comes from the first of the site's
        # columns that holds one.
        all_keys = _band_keys(call_cells[adopted], ranks[adopted], first, n_genotyped)
        order = np.lexsort((place[adopted], all_keys))
        leading = np.ones(len(order), dtype=bool)
        leading[1:] = all_keys[order[1:]] != all_keys[order[:-1]]
        picked = adopted[order[leading]]
        call_keys = all_keys[order[leading]]
        del all_keys, order, leading, adopted
        call_labels = call_value_names[values["call_values"][picked].astype(np.int64)]
        call_columns = columns[picked]
        keys = sorted_unique(np.concatenate([covered, call_keys]))
        calls = np.full(len(keys), FILTERED, dtype=np.int8)
        labels = np.full(len(keys), "", dtype=object)
        at = np.searchsorted(keys, call_keys)
        is_covered = np.isin(call_keys, covered)
        for code in sorted_unique(call_columns).tolist():
            of_column = call_columns == code
            calls[at[of_column]] = source_call_codes(
                str(source_columns[code]), call_labels[of_column], is_covered[of_column]
            )
        labels[at] = call_labels
        if rules.min_quality is not None or rules.min_depth is not None:
            failing = np.zeros(len(call_keys), dtype=bool)
            if rules.min_quality is not None:
                # A missing quality, NaN, fails every threshold.
                failing |= ~(values["call_quality"][picked] >= rules.min_quality)
            if rules.min_depth is not None:
                failing |= ~np.isin(call_keys, deep)
            gated = at[failing]
            calls[gated[calls[gated] != MISSING]] = FILTERED
        reference = alternative = np.full(len(keys), -1, dtype=np.int64)
        posteriors = np.full((len(keys), 3), np.nan)
    else:
        count_cells, values = rows["count"]
        alleles = values["count_alleles"].astype(np.int64)
        ranks = allele_rank[alleles]
        on = ranks >= 0
        alleles, ranks = alleles[on], ranks[on]
        umis = values["counts"][on].astype(np.int64)
        count_keys = _band_keys(count_cells[on], ranks, first, n_genotyped)
        keys = sorted_unique(np.concatenate([covered, count_keys]))
        at = np.searchsorted(keys, count_keys)
        is_reference = alleles == reference_alleles[ranks]
        is_alternative = alleles == alternative_alleles[ranks]
        reference = np.bincount(
            at[is_reference], weights=umis[is_reference], minlength=len(keys)
        ).astype(np.int64)
        alternative = np.bincount(
            at[is_alternative], weights=umis[is_alternative], minlength=len(keys)
        ).astype(np.int64)
        labels = np.full(len(keys), "", dtype=object)
        if method == "threshold":
            assert rules.min_umis is not None
            calls = threshold_calls(
                reference,
                alternative,
                min_umis=rules.min_umis,
                min_mutant_umis=rules.min_mutant_umis,
                min_fraction=rules.min_fraction,
            )
            posteriors = np.full((len(keys), 3), np.nan)
        else:
            assert rules.min_umis is not None and rules.min_posterior is not None
            swap = (
                np.zeros(len(umis))
                if rules.swap is None
                else unit_swap_probabilities(
                    umis, values["counts_reads"][on].astype(np.int64), rules.swap
                )
            )
            posteriors, confident = gift_posteriors(
                at, len(keys), umis, wild_type_probabilities[alleles], swap
            )
            calls = gift_calls(
                posteriors,
                confident,
                min_posterior=rules.min_posterior,
                min_umis=rules.min_umis,
            )
    return _BandCalls(
        cells=first + keys // n_genotyped,
        sites=keys % n_genotyped,
        calls=calls,
        labels=labels,
        reference=reference,
        alternative=alternative,
        posteriors=posteriors,
    )
