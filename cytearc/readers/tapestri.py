"""Read Mission Bio Tapestri single-cell DNA and protein files, and DAb-seq files.

Tapestri genotypes hundreds of amplicons in each cell, often beside surface
proteins. Tapestri Pipeline 2.0 writes ``<sample>.dna+protein.h5``, or
``<sample>.dna.h5`` without proteins, whose ``assays`` group holds one group
per assay. Each assay group holds ``ra``, one value per cell (``barcode``,
``sample_name`` and, for the DNA variants, ``filtered``), ``ca``, one value
per feature, ``layers`` of cells by features, and ``metadata``, such as the
panel, genome and pipeline version:

- ``dna_read_counts``: each cell's reads of each amplicon, with the
  amplicon's ``CHROM``, ``start_pos`` and ``end_pos``, which copy number
  analysis normalizes;
- ``protein_read_counts``: each cell's reads of each antibody tag;
- ``dna_variants``: the candidate variants, with ``id`` (such as
  ``chr1:115256513:G/A``), ``CHROM``, ``POS``, ``REF``, ``ALT`` and
  ``filtered``, 0 for a variant that passes the pipeline's default filters,
  and the layers ``NGT`` (0 wild type, 1 heterozygous, 2 homozygous
  alternative, 3 no call), ``AF`` (alternative reads over depth, in percent),
  ``DP`` (depth) and ``GQ`` (genotype quality).

``all_barcodes`` holds the reads of every barcode, cells or not, and is not
read. A file's cells are keyed by sample and barcode, ``<sample>__<barcode>``,
as a merged CyteArc store names its cells, so the cells of several samples in
one file stay apart even where their barcodes repeat.

The DAb-seq pipeline writes a ``.hdf5`` file of genotypes of cells by
variants: ``AD`` and ``RD``, the reads of the alternative and the reference,
``DP``, ``GQ`` and ``GT`` (0 wild type, 1 heterozygous, 2 homozygous
alternative, 3 no call, and -1 one reference allele beside one not called),
with the variants named
``<gene>:<chrom>:<pos>:<ref>/<alt>`` in ``VARIANTS``, the cells in
``CELL_BARCODES`` (such as ``AACAACCTAGTCTTCTCG-1-all``, the barcode, the
tube and the sample) and missing values written as -1. Its ``FLT3-ITD``
variants, which the pipeline adds from ITDseek, hold the depth in ``RD``.

These readers follow Mission Bio's documentation and real files, since
Mission Bio's mosaic is GPL-3.0 and its pipeline proprietary, and the DAb-seq
pipeline's documented layout.
"""

import hashlib
import os
import re
import sys
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ..utils.count_values import CountValueRange, dense_count_ranges
from ..utils.warnings import warn
from ..variants.counts import ALLELE_ROLES, Contig, EntryBatch, SiteTable
from ._assay_names import auto_name_feat_table
from ._text import as_text, require_unique_identifiers
from .cellranger import CrReader
from ..utils.arguments import quoted_value

__all__ = [
    "AMPLICON_FEATURE_TYPE",
    "CELL_ID_SEPARATOR",
    "PROTEIN_FEATURE_TYPE",
    "SkippedVariants",
    "TapestriReader",
    "TapestriSource",
    "dab_seq_source",
    "tapestri_cell_ids",
    "tapestri_source",
]

#: The text between a cell's sample and its barcode in the cell IDs, as a
#: merged store names its cells.
CELL_ID_SEPARATOR = "__"
#: The feature type of the amplicons, whose reads form the ``CNV`` assay.
AMPLICON_FEATURE_TYPE = "Amplicon"
#: The feature type of the antibody tags, whose reads form the ``ADT`` assay.
PROTEIN_FEATURE_TYPE = "Antibody Capture"
# The assays of a Tapestri file that a store holds, with their feature type.
_STORE_ASSAYS = (
    ("dna_read_counts", AMPLICON_FEATURE_TYPE),
    ("protein_read_counts", PROTEIN_FEATURE_TYPE),
)
_VARIANTS = "assays/dna_variants"
# The variants' metadata that the source record keeps: the panel, genome and
# pipeline, and the thresholds of the pipeline's genotype filters.
_RECORDED_METADATA = (
    "panel_name",
    "genome_version",
    "pipeline_version",
    "gq_cutoff",
    "dp_cutoff",
    "af_cutoff",
)
# The labels of the genotype codes -1 to 3, as Tapestri and DAb-seq write
# them, by the code plus one. DAb-seq's GT is the sum of a diploid call's two
# allele indices as scikit-allel reads them from the pipeline's VCF, 0 for the
# reference, 1 for the alternative and -1 for an allele not called, with a sum
# of -2, neither allele called, written as 3 (resources.py, vcf_to_tables); so
# its -1 is one reference allele beside one not called. Tapestri's NGT is 0
# to 3.
_CALL_LABELS = np.asarray(["-1", "0", "1", "2", "3"], dtype=object)
_NO_CALL = 3
_LOWEST_CALL = {"tapestri": 0, "dab_seq": -1}
# DAb-seq writes a missing depth, read count or quality as -1.
_MISSING = -1
# AF times DP over 100 is a whole number of reads up to the rounding of AF's
# dtype: a few of its epsilons of each read, and 1e-6 beside them.
_WHOLE_TOLERANCE = 1e-6
_WHOLE_EPSILONS = 16
# A REF or ALT that is a sequence of bases; an ALT may also be VCF's "*", a
# deletion that spans the position.
_BASES = re.compile(r"[ACGTN]+")
_ITD_PREFIX = "FLT3-ITD:"
# Bytes that describing a file holds for each candidate variant beside the
# copies of its text: its name, alleles and contig as Python text, the parts
# of its name, and its entries in the lookups of names and positions;
# DAb-seq's names split further.
_TAPESTRI_VARIANT_BYTES = 700
_DAB_SEQ_VARIANT_BYTES = 1_100
# And for each cell: its barcode, sample and ID as Python text and their
# entry in the lookup of repeated IDs.
_CELL_BYTES = 300
# The copies of each text byte that the description holds at once.
_DESCRIBED_TEXT_COPIES = 2
# The variable-length texts read at once to find the longest.
_WIDTH_CHUNK = 4_096
# A block of cells counts each cell's ID as this many entries, since an ID
# holds about as much as two entries of reads and calls.
_ID_ENTRIES = 2


def tapestri_cell_ids(
    samples: Sequence[str], barcodes: Sequence[str], path: str, *, start: int = 0
) -> np.ndarray:
    """The IDs of a Tapestri file's cells: ``<sample>__<barcode>``.

    Args:
        samples: Each cell's sample name.
        barcodes: Each cell's barcode.
        path: The file, as errors name it.
        start: The file's row of the first cell, counting from 0, which
            errors count from 1.

    Raises:
        ValueError: If a cell's sample or barcode is empty, which would join
            to an ID that lacks a part.
    """
    lacking = [
        position
        for position, (sample, barcode) in enumerate(
            zip(samples, barcodes, strict=True)
        )
        if not str(sample).strip() or not str(barcode).strip()
    ]
    if lacking:
        raise ValueError(
            f"{path} has cells without a sample name or a barcode, the first in row "
            f"{start + lacking[0] + 1}; each cell's ID joins both"
        )
    return np.asarray(
        [
            f"{sample}{CELL_ID_SEPARATOR}{barcode}"
            for sample, barcode in zip(samples, barcodes, strict=True)
        ],
        dtype=object,
    )


def _texts(values: Any) -> np.ndarray:
    return np.asarray(
        [as_text(value) for value in np.asarray(values).reshape(-1).tolist()],
        dtype=object,
    )


def _dataset(group: h5py.Group, name: str, path: str) -> h5py.Dataset:
    node = group.get(name)
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"{path} lacks {group.name.strip('/')}/{name}")
    return node


def _metadata_value(value: Any) -> Any:
    """A metadata value as JSON-ready text, numbers or lists of them."""
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        return item.decode("utf-8") if isinstance(item, bytes) else item
    return [_metadata_value(part) for part in array]


def _metadata(group: h5py.Group | None) -> dict[str, Any]:
    """The datasets of a ``metadata`` group, as JSON-ready values."""
    if not isinstance(group, h5py.Group):
        return {}
    return {
        name: _metadata_value(node[()])
        for name, node in group.items()
        if isinstance(node, h5py.Dataset)
    }


@dataclass(frozen=True, slots=True)
class _Cells:
    """A Tapestri assay's cells: their samples, barcodes and IDs."""

    samples: np.ndarray
    barcodes: np.ndarray
    ids: np.ndarray


def _assay_cells(handle: h5py.File, assay: str, path: str) -> _Cells:
    rows = handle[f"assays/{assay}"].get("ra")
    if not isinstance(rows, h5py.Group):
        raise ValueError(f"{path} lacks assays/{assay}/ra")
    barcodes = _texts(_dataset(rows, "barcode", path)[:])
    samples = _texts(_dataset(rows, "sample_name", path)[:])
    if len(samples) != len(barcodes):
        raise ValueError(
            f"assays/{assay}/ra of {path} names {len(barcodes)} barcodes but "
            f"{len(samples)} samples"
        )
    ids = tapestri_cell_ids(samples.tolist(), barcodes.tolist(), path)
    repeated = pd.Index(ids).duplicated()
    if repeated.any():
        raise ValueError(
            f"assays/{assay} of {path} lists the barcode {quoted_value(barcodes[repeated][0])} "
            f"of the sample {quoted_value(samples[repeated][0])} twice"
        )
    return _Cells(samples=samples, barcodes=barcodes, ids=ids)


def _layer(handle: h5py.File, assay: str, name: str, path: str) -> h5py.Dataset:
    layers = handle[f"assays/{assay}"].get("layers")
    if not isinstance(layers, h5py.Group):
        raise ValueError(f"{path} lacks assays/{assay}/layers")
    return _dataset(layers, name, path)


class TapestriReader(CrReader):
    """Read a Tapestri Pipeline 2.0 file of single-cell DNA and protein.

    The amplicons' reads (``dna_read_counts``) become the ``CNV`` assay, of
    feature type ``"Amplicon"``, and the antibody tags' reads
    (``protein_read_counts``) the ``ADT`` assay, of feature type
    ``"Antibody Capture"``; a file without proteins gives the ``CNV`` assay
    alone. Each feature's ``ca`` values other than its ``id``, such as an
    amplicon's ``CHROM``, ``start_pos`` and ``end_pos`` or an antibody's
    ``antibody_sequence``, become feature metadata of its assay. Each cell's
    ``sample_name`` and, when the file holds DNA variants, its ``filtered``
    flag (True for a cell that the pipeline's default filters removed)
    become cell metadata. Cells are named ``<sample>__<barcode>``. Pass the
    reader to :class:`~cytearc.TapestriToZarr`, which also records the file's
    metadata, such as its panel, genome and pipeline version. Variants are
    imported into a store with
    ``ds.variants.import_counts(path, format="tapestri")``. The reader warns
    about any other assay group of the file, which the store does not hold.

    Args:
        h5_fn: The Tapestri ``.h5`` file.

    Attributes:
        samples: The samples of the file's cells, in order of first
            appearance.
        metadata: The file's metadata: ``assays``, the scalar metadata of
            each assay group, such as its ``panel_name``,
            ``genome_version`` and ``pipeline_version``, and the file's own
            ``metadata`` group, such as ``sdk_version``.

    Raises:
        ValueError: If the file holds neither the amplicon nor the protein
            reads, lacks an assay's arrays, its assays list other cells, a
            sample's barcode repeats, an assay lists no features, a feature ID
            repeats or is blank, or a ``ca`` or ``filtered`` array does not
            align with its assay.
    """

    def __init__(self, h5_fn: str) -> None:
        self.path = os.fspath(h5_fn)
        self.h5obj: h5py.File = h5py.File(self.path, mode="r")
        try:
            self._load_layout()
            super().__init__(self._handle_version())
        except BaseException:
            self.h5obj.close()
            raise

    def _load_layout(self) -> None:
        handle = self.h5obj
        assays = handle.get("assays")
        present = (
            [
                (name, feature_type)
                for name, feature_type in _STORE_ASSAYS
                if name in assays
            ]
            if isinstance(assays, h5py.Group)
            else []
        )
        if not present:
            raise ValueError(
                f"{self.path} holds neither assays/dna_read_counts nor "
                "assays/protein_read_counts; it is not a Tapestri file of amplicon "
                "or protein reads"
            )
        assert isinstance(assays, h5py.Group)
        read = {name for name, _type in present} | {"dna_variants"}
        others = sorted(name for name in assays if name not in read)
        if others:
            warn(
                f"{self.path} holds {', '.join(f'assays/{name}' for name in others)}, "
                "which the store does not hold: it holds the amplicon and protein "
                "reads, and import_counts reads the DNA variants"
            )
        self._assays = present
        cells = _assay_cells(handle, present[0][0], self.path)
        for name, _feature_type in present[1:]:
            other = _assay_cells(handle, name, self.path)
            if not np.array_equal(other.ids, cells.ids):
                raise ValueError(
                    f"assays/{name} of {self.path} lists other cells than "
                    f"assays/{present[0][0]}, or in another order"
                )
        self._cells = cells
        self._layers = [
            _layer(handle, name, "read_counts", self.path) for name, _type in present
        ]
        feature_ids: list[str] = []
        feature_types: list[str] = []
        widths: list[int] = []
        for (name, feature_type), layer in zip(present, self._layers, strict=True):
            columns = handle[f"assays/{name}"].get("ca")
            if not isinstance(columns, h5py.Group):
                raise ValueError(f"{self.path} lacks assays/{name}/ca")
            ids = _texts(_dataset(columns, "id", self.path)[:])
            if not len(ids):
                raise ValueError(f"assays/{name} of {self.path} lists no features")
            require_unique_identifiers(ids, f"The feature IDs of assays/{name}")
            if layer.shape != (len(cells.ids), len(ids)):
                raise ValueError(
                    f"assays/{name}/layers/read_counts of {self.path} is "
                    f"{layer.shape[0]} by {layer.shape[1]}, but the assay lists "
                    f"{len(cells.ids)} cells and {len(ids)} features"
                )
            feature_ids.extend(ids.tolist())
            feature_types.extend([feature_type] * len(ids))
            widths.append(len(ids))
        self._featureIdList = feature_ids
        self._featureTypeList = feature_types
        self._widths = widths
        self._filtered: np.ndarray | None = None
        if "dna_variants" in assays:
            variants = _assay_cells(handle, "dna_variants", self.path)
            if not np.array_equal(variants.ids, cells.ids):
                raise ValueError(
                    f"assays/dna_variants of {self.path} lists other cells than "
                    f"assays/{present[0][0]}, or in another order"
                )
            flags = np.asarray(
                _dataset(handle[f"{_VARIANTS}/ra"], "filtered", self.path)[:]
            ).reshape(-1)
            if len(flags) != len(cells.ids) or not np.isin(flags, (0, 1)).all():
                raise ValueError(
                    f"assays/dna_variants/ra/filtered of {self.path} must hold a 0 "
                    "or 1 for each cell"
                )
            self._filtered = flags.astype(bool)
        self.samples = list(dict.fromkeys(cells.samples.tolist()))
        self.metadata = {
            "assays": {
                name: _metadata(node.get("metadata"))
                for name, node in assays.items()
                if isinstance(node, h5py.Group)
            },
            "file": _metadata(handle.get("metadata")),
        }

    def _handle_version(self) -> dict[str, str | None]:
        return {
            "feature_ids": "feature_ids",
            "feature_names": "feature_names",
            "feature_types": "feature_types",
            "cell_names": "cell_names",
        }

    def _read_dataset(self, key: str) -> list[str]:
        if key in ("feature_ids", "feature_names"):
            return list(self._featureIdList)
        if key == "feature_types":
            return list(self._featureTypeList)
        return [str(cell) for cell in self._cells.ids.tolist()]

    def _auto_named_feat_table(self, assay_feats: pd.DataFrame) -> pd.DataFrame:
        return auto_name_feat_table(
            assay_feats, {**self.autoNames, AMPLICON_FEATURE_TYPE: "CNV"}
        )

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        """The dtype in which ``consume`` yields the reads: the layers' common one."""
        return np.dtype(
            np.result_type(*(layer.dtype for layer in self._layers))
        ).newbyteorder("=")

    def _block(self, start: int, stop: int) -> np.ndarray:
        """The reads of cells ``[start, stop)``, cells by every feature."""
        return np.concatenate(
            [
                np.asarray(layer[start:stop], dtype=self.matrix_dtype)
                for layer in self._layers
            ],
            axis=1,
        )

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        """Return the range of the reads of each group of features.

        Raises:
            MemoryError: If one cell's reads do not fit in ``maxBytes``.
            ValueError: If a value is NaN or infinite.
        """
        return dense_count_ranges(
            self._block,
            self.nCells,
            self.nFeatures,
            maxBytes=maxBytes,
            groups=featureGroups,
        )

    def consume(
        self, batch_size: int, lines_in_mem: int | None = None
    ) -> Generator[coo_matrix, None, None]:
        """Yield the reads of ``batch_size`` cells at a time, cells by features.

        Args:
            batch_size: Number of cells per chunk.
            lines_in_mem: Unused; kept for CrReader API compatibility.
        """
        for start in range(0, self.nCells, batch_size):
            stop = min(self.nCells, start + batch_size)
            block = self._block(start, stop)
            rows, columns = np.nonzero(block)
            values = block[rows, columns]
            del block
            matrix = coo_matrix(
                (values, (rows, columns)), shape=(stop - start, self.nFeatures)
            )
            del rows, columns, values
            yield matrix

    def producer_staging_bytes(self, batch_size: int, lines_in_mem: int) -> int:
        """Count what one batch of cells holds as it is read, per value.

        A batch reads each assay's cells by features and joins them, two
        dense copies; finds the nonzero entries' rows and columns, 16 bytes
        each, and takes their values; and builds the sparse batch, whose
        32-bit indices are copied from the 64-bit ones. Every value counts
        the most of these at once: its dtype's bytes and 24.
        """
        rows = min(max(1, int(batch_size)), self.nCells)
        itemsize = self.matrix_dtype.itemsize
        return int(rows * self.nFeatures * (itemsize + 24))

    def _sparse_import_resident_bytes(self) -> int:
        """The cells' samples, barcodes and IDs, which the reader holds while
        it is open."""
        return int(
            sum(
                sum(map(sys.getsizeof, values.tolist())) + 8 * len(values)
                for values in (
                    self._cells.samples,
                    self._cells.barcodes,
                    self._cells.ids,
                )
            )
        )

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        """Each cell's sample and, with DNA variants, its ``filtered`` flag."""
        yield "sample_name", np.asarray(self._cells.samples, dtype=object)
        if self._filtered is not None:
            yield "filtered", self._filtered.copy()

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        """The feature types, then each assay's ``ca`` values beside the IDs.

        A column holds None for the features of the other assay, so it
        reaches only its own assay's feature metadata.
        """
        yield from super().get_feature_columns()
        offsets = np.r_[0, np.cumsum(self._widths)]
        # A name that both assays use is one column, each assay's values in
        # its own features.
        merged: dict[str, np.ndarray] = {}
        for index, (name, _feature_type) in enumerate(self._assays):
            columns = self.h5obj[f"assays/{name}/ca"]
            for key, node in columns.items():
                if key == "id" or not isinstance(node, h5py.Dataset):
                    continue
                values = np.asarray(node[:])
                if values.ndim != 1 or len(values) != self._widths[index]:
                    raise ValueError(
                        f"assays/{name}/ca/{key} of {self.path} has shape "
                        f"{values.shape}; expected ({self._widths[index]},)"
                    )
                column = merged.setdefault(
                    key, np.full(self.nFeatures, None, dtype=object)
                )
                column[offsets[index] : offsets[index + 1]] = (
                    _texts(values) if values.dtype.kind in "SUO" else values.tolist()
                )
        yield from merged.items()

    def close(self) -> None:
        """Close the file."""
        self.h5obj.close()


@dataclass(frozen=True, slots=True)
class _Columns:
    """The variant columns that an import reads, and their sites and alleles.

    Attributes:
        columns: Each chosen column's position in the file, ascending.
        column_site: Each chosen column's site.
        column_allele: Each chosen column's alternative allele.
        site_first: Each site's first chosen column, as a position in
            ``columns``.
        sites: The sites and alleles.
        site_reference: Each site's reference allele.
        reference_column: For DAb-seq, the chosen column whose ``RD`` counts
            each site's reference, -1 where none does; empty for Tapestri.
        labels: Each chosen column's variant as the file names it.
    """

    columns: np.ndarray
    column_site: np.ndarray
    column_allele: np.ndarray
    site_first: np.ndarray
    sites: SiteTable
    site_reference: np.ndarray
    reference_column: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class _Variants:
    """A file's variant columns, parsed from their names."""

    names: np.ndarray
    contig: np.ndarray
    position: np.ndarray
    reference: np.ndarray
    alternative: np.ndarray
    passing: np.ndarray


@dataclass(frozen=True, slots=True)
class SkippedVariants:
    """The chosen variants of a file that cannot be imported, by name.

    The names identify the variants across files of one panel, so an import
    of several files counts each skipped variant once.

    Attributes:
        names: Every skipped variant, in the file's order.
        empty_alternatives: Those of them with an empty ``ALT``; the others'
            alleles are not sequences of bases for another reason.
    """

    names: tuple[str, ...]
    empty_alternatives: frozenset[str]


def _chosen_columns(
    variants: _Variants,
    requested: Sequence[str],
    *,
    short_names: np.ndarray,
    path: str,
    what: str,
) -> tuple[np.ndarray, dict[str, Any], SkippedVariants]:
    """The columns to import: the passing ones and those ``requested`` names.

    A requested item names a variant as the file does or as
    ``chrom:pos:ref/alt``, or a position as ``chrom:pos``, which takes every
    variant there. Variants whose alleles are not sequences of bases, such
    as an empty ``ALT``, are skipped and counted, and refused when named
    exactly.

    Returns:
        The chosen columns, ascending; what the source record keeps of the
        choice: the variants added beside the passing ones are counted and
        named by the SHA-256 of their names, one per line, so an import is
        identified by what it reads rather than how ``requested`` names it;
        and the variants skipped, which the import counts in its
        diagnostics.

    Raises:
        ValueError: If a requested item names no variant of the file, a
            variant named exactly cannot be imported, or no variant is left.
    """
    n_columns = len(variants.names)
    by_name: dict[str, list[int]] = {}
    for column, (name, short) in enumerate(
        zip(variants.names.tolist(), short_names.tolist(), strict=True)
    ):
        by_name.setdefault(name, []).append(column)
        if short != name:
            by_name.setdefault(short, []).append(column)
    positions = [
        f"{contig}:{position}"
        for contig, position in zip(
            variants.contig.tolist(), variants.position.tolist(), strict=True
        )
    ]
    by_position: dict[str, list[int]] = {}
    for column, key in enumerate(positions):
        by_position.setdefault(key, []).append(column)
    chosen = np.asarray(variants.passing, dtype=bool).copy()
    exact = np.zeros(n_columns, dtype=bool)
    for item in requested:
        if item in by_name:
            chosen[by_name[item]] = True
            exact[by_name[item]] = True
        elif item in by_position:
            chosen[by_position[item]] = True
        else:
            raise ValueError(
                f"include_variants names {quoted_value(item)}, which is neither a variant nor a "
                f"position of {path}; name variants as {what} does, such as "
                f"{quoted_value(variants.names[0] if n_columns else 'chr1:115256513:G/A')}, or "
                "positions such as 'chr1:115256513'"
            )
    alternative = variants.alternative
    empty = alternative == ""
    readable = np.asarray(
        [
            bool(_BASES.fullmatch(reference))
            and (alt == "*" or bool(_BASES.fullmatch(alt)))
            for reference, alt in zip(
                variants.reference.tolist(), alternative.tolist(), strict=True
            )
        ],
        dtype=bool,
    )
    skipped = chosen & ~readable
    refused = skipped & exact
    if refused.any():
        name = variants.names[refused][0]
        reason = (
            "an empty ALT"
            if alternative[refused][0] == ""
            else "alleles that are not bases"
        )
        raise ValueError(
            f"The variant {quoted_value(name)} of {path} has {reason}, which no documentation "
            "explains, so it cannot be imported; leave it out of include_variants"
        )
    unreadable = SkippedVariants(
        names=tuple(str(name) for name in variants.names[skipped]),
        empty_alternatives=frozenset(
            str(name) for name in variants.names[skipped & empty]
        ),
    )
    columns = np.flatnonzero(chosen & readable)
    if not len(columns):
        raise ValueError(
            f"{path} has no variant to import: none passes the pipeline's filters "
            "and include_variants names none that can be read; name variants or "
            "positions with include_variants"
        )
    added = variants.names[columns[~variants.passing[columns]]].tolist()
    record = {
        "n_variants": n_columns,
        "n_passing_variants": int(np.count_nonzero(variants.passing)),
        "n_imported_variants": len(columns),
        "n_added_variants": len(added),
        "added_variants_sha256": hashlib.sha256(
            "\n".join(added).encode("utf-8")
        ).hexdigest(),
    }
    return columns, record, unreadable


def _sites(
    variants: _Variants, columns: np.ndarray, *, path: str, itd: np.ndarray | None
) -> _Columns:
    """Group the chosen columns into sites, one per contig and position.

    The variants of one position are the alternative alleles of one site.
    Where their references differ, as a SNV and a deletion of the same
    position do once split from one record, the longest reference is the
    site's and each shorter one is extended by its remaining bases, with
    its alternative, as VCF writes several alternatives of one record.
    DAb-seq's FLT3-ITD variants (``itd``), which carry ITDseek's depth rather
    than the caller's, are named sites of their own, named as the file names
    them.

    Raises:
        ValueError: If two references of one position differ beyond their
            shared bases, or two variants name the same allele of a site.
    """
    labels = variants.names[columns]
    # A named site's key has no contig and its name in place of a position.
    keys: list[tuple[str, int | str]] = [
        ("", str(label)) if itd is not None and itd[chosen] else (contig, position)
        for chosen, (label, contig, position) in enumerate(
            zip(
                labels.tolist(),
                variants.contig[columns].tolist(),
                variants.position[columns].tolist(),
                strict=True,
            )
        )
    ]
    site_of: dict[tuple[str, int | str], int] = {}
    members: list[list[int]] = []
    for chosen, key in enumerate(keys):
        site = site_of.setdefault(key, len(site_of))
        if site == len(members):
            members.append([])
        members[site].append(chosen)
    contig_names = list(dict.fromkeys(key[0] for key in site_of if key[0]))
    contig_of = {name: index for index, name in enumerate(contig_names)}
    allele_site: list[int] = []
    allele_text: list[str] = []
    allele_role: list[str] = []
    column_allele = np.zeros(len(columns), dtype=np.int64)
    site_reference = np.zeros(len(members), dtype=np.int64)
    reference_column = np.full(len(members), -1, dtype=np.int64)
    for site, chosen_columns in enumerate(members):
        references = [str(variants.reference[columns[c]]) for c in chosen_columns]
        longest = max(references, key=len)
        site_reference[site] = len(allele_text)
        allele_site.append(site)
        allele_text.append(longest)
        allele_role.append("reference")
        seen: dict[str, int] = {}
        for chosen, reference in zip(chosen_columns, references, strict=True):
            label = str(labels[chosen])
            if not longest.startswith(reference):
                other = labels[chosen_columns[references.index(longest)]]
                raise ValueError(
                    f"The variants {quoted_value(other)} and {quoted_value(label)} of {path} give one "
                    f"position the references {quoted_value(longest)} and {quoted_value(reference)}, which "
                    "differ beyond their shared bases, so they cannot be alleles of "
                    "one site"
                )
            alternative = str(variants.alternative[columns[chosen]])
            text = (
                alternative
                if alternative == "*"
                else alternative + longest[len(reference) :]
            )
            if text == longest:
                raise ValueError(
                    f"The variant {quoted_value(label)} of {path} has the reference as its "
                    "alternative"
                )
            if text in seen:
                raise ValueError(
                    f"The variants {quoted_value(labels[seen[text]])} and {quoted_value(label)} of {path} "
                    f"name the same allele {quoted_value(text)} of one site"
                )
            seen[text] = chosen
            column_allele[chosen] = len(allele_text)
            allele_site.append(site)
            allele_text.append(text)
            allele_role.append("alternative")
            if itd is not None and reference_column[site] < 0 and not itd[chosen]:
                reference_column[site] = chosen
    table = SiteTable(
        contigs=tuple(Contig(name=name) for name in contig_names),
        site_contig=np.asarray(
            [contig_of[key[0]] if key[0] else -1 for key in site_of], dtype=np.int32
        ),
        site_position=np.asarray(
            [key[1] if key[0] else -1 for key in site_of], dtype=np.int64
        ),
        site_name=np.asarray(
            ["" if key[0] else str(key[1]) for key in site_of], dtype=object
        ),
        site_channel=np.full(len(site_of), "", dtype=object),
        allele_site=np.asarray(allele_site, dtype=np.int32),
        allele=np.asarray(allele_text, dtype=object),
        allele_role=np.asarray(allele_role, dtype=object),
    )
    column_site = np.asarray([site_of[key] for key in keys], dtype=np.int64).reshape(-1)
    return _Columns(
        columns=columns,
        column_site=column_site,
        column_allele=column_allele,
        site_first=np.asarray([chosen[0] for chosen in members], dtype=np.int64),
        sites=table,
        site_reference=site_reference,
        reference_column=reference_column if itd is not None else np.empty(0, np.int64),
        labels=labels,
    )


@dataclass(frozen=True, slots=True)
class TapestriSource:
    """The variants of a Tapestri or DAb-seq file, read as a source of counts.

    Each position is a site whose alleles are its reference and the
    alternatives of its chosen variants; the depth is the file's ``DP``,
    which every variant of a position shares. Tapestri gives each variant's
    alternative reads as ``AF × DP / 100`` and no reference reads, and its
    ``NGT`` with ``GQ`` as calls; the frequencies of the variants of one
    position may add up to more than 100%, so their alleles overlap. DAb-seq
    gives ``AD`` and ``RD`` and its ``GT`` with ``GQ``. The cells' IDs are
    read with each block of cells, so the source holds none of them.
    ``skipped`` counts the chosen variants that cannot be imported, which the
    import records in its diagnostics.
    """

    path: str
    format: str
    files: tuple[tuple[str, str], ...]
    count_unit: str | None
    depth_definition: str
    count_fields: tuple[str, ...]
    call_fields: tuple[str, ...]
    depth_roles: frozenset[str] | None
    reference_from_depth: bool
    alleles_overlap: bool
    open_alleles: bool
    notes: dict[str, Any]
    skipped: SkippedVariants
    _columns: _Columns = field(repr=False)
    _n_cells: int = field(repr=False)
    _cell_arrays: tuple[str, ...] = field(repr=False)
    _layers: dict[str, str] = field(repr=False)

    def sites(self) -> SiteTable:
        """Each position of the chosen variants, with its alleles."""
        return self._columns.sites

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        """Each block of cells: its cells, then their depth, counts and calls.

        A block holds as many cells as keep its entries, cells by chosen
        variants, and its cells' IDs, counted as two entries each, within
        ``chunk_rows``.
        """
        chosen = self._columns
        block = max(1, int(chunk_rows) // (len(chosen.columns) + _ID_ENTRIES))
        with h5py.File(self.path, "r") as handle:
            layers = {name: handle[path] for name, path in self._layers.items()}
            for name, layer in layers.items():
                if layer.shape != (self._n_cells, self.notes["n_variants"]):
                    raise ValueError(
                        f"{self._layers[name]} of {self.path} is {layer.shape[0]} by "
                        f"{layer.shape[1]}, but the file lists {self._n_cells} cells "
                        f"and {self.notes['n_variants']} variants"
                    )
            for start in range(0, self._n_cells, block):
                stop = min(self._n_cells, start + block)
                ids = self._block_ids(handle, start, stop)
                yield EntryBatch(
                    kind="cells",
                    barcodes=ids,
                    barcode_codes=np.arange(len(ids), dtype=np.int64),
                )
                if self.format == "tapestri":
                    yield from self._tapestri_block(layers, start, stop, ids)
                else:
                    yield from self._dab_seq_block(layers, start, stop, ids)

    def _block_ids(self, handle: h5py.File, start: int, stop: int) -> np.ndarray:
        """The IDs of cells ``[start, stop)``: Tapestri's ``<sample>__<barcode>``,
        or DAb-seq's barcodes."""
        parts = [_texts(handle[name][start:stop]) for name in self._cell_arrays]
        if len(parts) == 1:
            return parts[0]
        samples, barcodes = parts
        return tapestri_cell_ids(
            samples.tolist(), barcodes.tolist(), self.path, start=start
        )

    def _read(self, layer: h5py.Dataset, start: int, stop: int) -> np.ndarray:
        return np.asarray(layer[start:stop, self._columns.columns])

    def _site_depth(self, depth: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Each cell's depth at each site, which its variants must agree on."""
        chosen = self._columns
        site_depth = depth[:, chosen.site_first]
        disagree = depth != site_depth[:, chosen.column_site]
        if disagree.any():
            row, column = (int(value) for value in np.argwhere(disagree)[0])
            first = int(chosen.site_first[chosen.column_site[column]])
            raise ValueError(
                f"The variants {quoted_value(chosen.labels[first])} and "
                f"{quoted_value(chosen.labels[column])} of {self.path} share a position but "
                f"give the cell {quoted_value(ids[row])} the DP {int(depth[row, first])} and "
                f"{int(depth[row, column])}, while the variants of one position "
                "are alleles of one site, whose depth they share"
            )
        return site_depth

    def _depth_batch(self, site_depth: np.ndarray, ids: np.ndarray) -> EntryBatch:
        rows, sites = np.nonzero(site_depth > 0)
        return EntryBatch(
            kind="depth",
            barcodes=ids,
            barcode_codes=rows.astype(np.int64),
            keys=sites.astype(np.int64),
            values=site_depth[rows, sites].astype(np.int64),
        )

    def _calls_batch(
        self,
        calls: np.ndarray,
        depth: np.ndarray,
        quality: np.ndarray,
        ids: np.ndarray,
        source: str,
    ) -> EntryBatch:
        """The calls of a block where the cell has reads at the position.

        A call without reads says nothing of the cell: Tapestri calls some
        such cells wild type at genotype quality 0, and DAb-seq calls FLT3-ITD
        wild type in every cell where ITDseek found none, read or not.
        """
        rows, columns = np.nonzero(depth > 0)
        return EntryBatch(
            kind="calls",
            barcodes=ids,
            barcode_codes=rows.astype(np.int64),
            keys=self._columns.column_allele[columns],
            call_source=source,
            calls=_CALL_LABELS[calls[rows, columns] + 1],
            call_quality=quality[rows, columns],
        )

    def _check_codes(self, calls: np.ndarray, ids: np.ndarray, layer: str) -> None:
        lowest = _LOWEST_CALL[self.format]
        bad = (calls < lowest) | (calls > _NO_CALL)
        if bad.any():
            row, column = (int(value) for value in np.argwhere(bad)[0])
            raise ValueError(
                f"{layer} of {self.path} gives the cell {quoted_value(ids[row])} at "
                f"{quoted_value(self._columns.labels[column])} the genotype "
                f"{int(calls[row, column])}; genotypes are {lowest} to {_NO_CALL}"
            )

    def _tapestri_block(
        self,
        layers: dict[str, h5py.Dataset],
        start: int,
        stop: int,
        ids: np.ndarray,
    ) -> Iterator[EntryBatch]:
        depth = self._read(layers["DP"], start, stop).astype(np.int64)
        if np.any(depth < 0):
            row, column = (int(value) for value in np.argwhere(depth < 0)[0])
            raise ValueError(
                f"{self._layers['DP']} of {self.path} gives the cell {quoted_value(ids[row])} "
                f"at {quoted_value(self._columns.labels[column])} the depth "
                f"{int(depth[row, column])}"
            )
        site_depth = self._site_depth(depth, ids)
        yield self._depth_batch(site_depth, ids)
        del site_depth
        epsilon = (
            float(np.finfo(layers["AF"].dtype).eps)
            if layers["AF"].dtype.kind == "f"
            else 0.0
        )
        frequency = self._read(layers["AF"], start, stop).astype(np.float64)
        outside = ~np.isfinite(frequency) | (frequency < 0) | (frequency > 100)
        if outside.any():
            row, column = (int(value) for value in np.argwhere(outside)[0])
            raise ValueError(
                f"{self._layers['AF']} of {self.path} gives the cell {quoted_value(ids[row])} "
                f"at {quoted_value(self._columns.labels[column])} the allele frequency "
                f"{quoted_value(float(frequency[row, column]))}; frequencies lie from 0 to 100%"
            )
        reads = frequency * depth
        reads /= 100
        whole = np.rint(reads)
        np.subtract(reads, whole, out=reads)
        np.abs(reads, out=reads)
        # A whole number of reads stays within the rounding of AF times depth.
        reads -= _WHOLE_TOLERANCE + _WHOLE_EPSILONS * epsilon * depth
        if np.any(reads > 0):
            row, column = (int(value) for value in np.argwhere(reads > 0)[0])
            raise ValueError(
                f"The allele frequency {quoted_value(float(frequency[row, column]))} and depth "
                f"{int(depth[row, column])} of the cell {quoted_value(ids[row])} at "
                f"{quoted_value(self._columns.labels[column])} in {self.path} do not give a "
                "whole number of reads as AF × DP / 100"
            )
        del reads, frequency
        rows, columns = np.nonzero(whole > 0)
        yield EntryBatch(
            kind="counts",
            barcodes=ids,
            barcode_codes=rows.astype(np.int64),
            keys=self._columns.column_allele[columns],
            values=whole[rows, columns].astype(np.int64),
        )
        del whole, rows, columns
        calls = self._read(layers["NGT"], start, stop).astype(np.int64)
        self._check_codes(calls, ids, self._layers["NGT"])
        quality = self._read(layers["GQ"], start, stop).astype(np.float64)
        yield self._calls_batch(calls, depth, quality, ids, "NGT")

    def _dab_seq_block(
        self,
        layers: dict[str, h5py.Dataset],
        start: int,
        stop: int,
        ids: np.ndarray,
    ) -> Iterator[EntryBatch]:
        chosen = self._columns
        depth = self._read(layers["DP"], start, stop).astype(np.int64)
        self._check_missing(depth, ids, "DP")
        site_depth = self._site_depth(depth, ids)
        yield self._depth_batch(site_depth, ids)
        del site_depth
        alternative = self._read(layers["AD"], start, stop).astype(np.int64)
        self._check_missing(alternative, ids, "AD")
        rows, columns = np.nonzero(alternative > 0)
        yield EntryBatch(
            kind="counts",
            barcodes=ids,
            barcode_codes=rows.astype(np.int64),
            keys=chosen.column_allele[columns],
            values=alternative[rows, columns],
        )
        del alternative, rows, columns
        reference = self._read(layers["RD"], start, stop).astype(np.int64)
        self._check_missing(reference, ids, "RD")
        sites = np.flatnonzero(chosen.reference_column >= 0)
        site_reference = reference[:, chosen.reference_column[sites]]
        # Every variant of a site counts the same reference reads, but
        # FLT3-ITD's sites, whose RD holds ITDseek's depth, have none.
        rank = np.full(chosen.sites.n_sites, -1, dtype=np.int64)
        rank[sites] = np.arange(len(sites))
        compared = np.flatnonzero(rank[chosen.column_site] >= 0)
        disagree = (
            reference[:, compared]
            != site_reference[:, rank[chosen.column_site[compared]]]
        )
        if disagree.any():
            row, position = (int(value) for value in np.argwhere(disagree)[0])
            column = int(compared[position])
            first = int(chosen.reference_column[chosen.column_site[column]])
            raise ValueError(
                f"The variants {quoted_value(chosen.labels[first])} and "
                f"{quoted_value(chosen.labels[column])} of {self.path} share a position but "
                f"give the cell {quoted_value(ids[row])} the reference reads "
                f"{int(reference[row, first])} and {int(reference[row, column])}, "
                "while the variants of one position count its reference reads once"
            )
        del reference
        rows, at = np.nonzero(site_reference > 0)
        yield EntryBatch(
            kind="counts",
            barcodes=ids,
            barcode_codes=rows.astype(np.int64),
            keys=chosen.site_reference[sites[at]],
            values=site_reference[rows, at],
        )
        del site_reference, rows, at
        calls = self._read(layers["GT"], start, stop).astype(np.int64)
        self._check_codes(calls, ids, self._layers["GT"])
        quality = self._read(layers["GQ"], start, stop).astype(np.float64)
        self._check_missing(quality, ids, "GQ")
        quality[quality == _MISSING] = np.nan
        yield self._calls_batch(calls, depth, quality, ids, "GT")

    def _check_missing(self, values: np.ndarray, ids: np.ndarray, name: str) -> None:
        """Refuse a negative value other than DAb-seq's -1 for a missing one."""
        bad = values < _MISSING
        if bad.any():
            row, column = (int(value) for value in np.argwhere(bad)[0])
            raise ValueError(
                f"{name} of {self.path} gives the cell {quoted_value(ids[row])} at "
                f"{quoted_value(self._columns.labels[column])} the value "
                f"{quoted_value(values[row, column].item())}; DAb-seq writes a missing value as -1"
            )


def _text_width(dataset: h5py.Dataset) -> int:
    """The bytes of a text dataset's longest value: its fixed width, or the
    longest of its variable-length values, read a chunk at a time."""
    dtype = dataset.dtype
    if dtype.kind == "S":
        return int(dtype.itemsize)
    longest = 0
    for start in range(0, int(dataset.shape[0]), _WIDTH_CHUNK):
        values = np.asarray(dataset[start : start + _WIDTH_CHUNK]).tolist()
        longest = max([longest, *map(len, values)])
    return longest


def _check_description(
    path: str,
    memory_bytes: int | None,
    *,
    n_variants: int,
    variant_bytes: int,
    n_cells: int,
    cell_bytes: int,
) -> None:
    """Refuse to describe a file whose variants and cells outgrow the budget.

    Raises:
        MemoryError: If reading the variants' names and alleles and the
            cells' IDs, and finding the chosen variants and repeated cells,
            holds more than ``memory_bytes``.
    """
    needed = n_variants * variant_bytes + n_cells * cell_bytes
    if memory_bytes is not None and needed > memory_bytes:
        raise MemoryError(
            f"Reading the {n_variants} variants and {n_cells} cells of {path} holds "
            f"about {needed} bytes, more than the memory budget of {memory_bytes} "
            "bytes. Raise mem_budget."
        )


def tapestri_source(
    path: str | os.PathLike[str],
    include_variants: Sequence[str] = (),
    memory_bytes: int | None = None,
) -> TapestriSource:
    """Describe the DNA variants of a Tapestri ``.h5`` file as a source of counts.

    The variants that pass the pipeline's default filters (``filtered`` 0)
    are read, with those that ``include_variants`` names. Before it reads them, the
    description checks what it holds against ``memory_bytes``. The notes
    keep the variants' panel, genome and pipeline version, and the
    thresholds of the pipeline's genotype filters, ``gq_cutoff``,
    ``dp_cutoff`` and ``af_cutoff``.

    Raises:
        ValueError: If the file lacks the DNA variants or their arrays, a
            variant's ``id`` disagrees with its ``CHROM``, ``POS``, ``REF``
            and ``ALT``, ``include_variants`` names an unknown or unreadable
            variant,
            a cell repeats, or no variant is left to import.
        MemoryError: If describing the variants and cells does not fit
            ``memory_bytes``.
    """
    text = os.fspath(path)
    requested = tuple(include_variants)
    with h5py.File(text, "r") as handle:
        group = handle.get(_VARIANTS)
        if not isinstance(group, h5py.Group):
            raise ValueError(
                f"{text} lacks {_VARIANTS}; it is not a Tapestri file of DNA "
                "variants (a DAb-seq file reads with format='dab_seq')"
            )
        columns_group = group.get("ca")
        if not isinstance(columns_group, h5py.Group):
            raise ValueError(f"{text} lacks {_VARIANTS}/ca")
        rows = group.get("ra")
        if not isinstance(rows, h5py.Group):
            raise ValueError(f"{text} lacks {_VARIANTS}/ra")
        texts = [
            _dataset(columns_group, name, text)
            for name in ("id", "CHROM", "REF", "ALT")
        ]
        cell_texts = [_dataset(rows, name, text) for name in ("barcode", "sample_name")]
        _check_description(
            text,
            memory_bytes,
            n_variants=int(texts[0].shape[0]),
            variant_bytes=_TAPESTRI_VARIANT_BYTES
            + _DESCRIBED_TEXT_COPIES * sum(map(_text_width, texts)),
            n_cells=int(cell_texts[0].shape[0]),
            cell_bytes=_CELL_BYTES
            + _DESCRIBED_TEXT_COPIES * sum(map(_text_width, cell_texts)),
        )
        cells = _assay_cells(handle, "dna_variants", text)
        ids, chrom, ref, alt = (_texts(dataset[:]) for dataset in texts)
        pos = np.asarray(_dataset(columns_group, "POS", text)[:]).reshape(-1)
        filtered = np.asarray(_dataset(columns_group, "filtered", text)[:]).reshape(-1)
        assays_metadata = _metadata(group.get("metadata"))
    lengths = {len(ids), len(chrom), len(pos), len(ref), len(alt), len(filtered)}
    if len(lengths) != 1:
        raise ValueError(f"The arrays of {_VARIANTS}/ca of {text} do not align")
    if not np.isin(filtered, (0, 1)).all():
        raise ValueError(f"{_VARIANTS}/ca/filtered of {text} must hold 0 or 1")
    contigs = []
    for index, (name, contig, position, reference, alternative) in enumerate(
        zip(
            ids.tolist(),
            chrom.tolist(),
            pos.tolist(),
            ref.tolist(),
            alt.tolist(),
            strict=True,
        )
    ):
        parts = name.split(":")
        if (
            len(parts) != 3
            or parts[0] not in (contig, f"chr{contig}")
            or parts[1] != str(position)
            or parts[2] != f"{reference}/{alternative}"
        ):
            raise ValueError(
                f"The variant {quoted_value(name)} at column {index} of {text} disagrees with its "
                f"CHROM {quoted_value(contig)}, POS {position}, REF {quoted_value(reference)} and ALT "
                f"{quoted_value(alternative)}"
            )
        contigs.append(parts[0])
    parsed = _Variants(
        names=ids,
        contig=np.asarray(contigs, dtype=object),
        position=pos.astype(np.int64),
        reference=np.asarray([value.upper() for value in ref.tolist()], dtype=object),
        alternative=np.asarray([value.upper() for value in alt.tolist()], dtype=object),
        passing=filtered == 0,
    )
    del chrom, ref, alt, contigs
    chosen, record, skipped = _chosen_columns(
        parsed, requested, short_names=ids, path=text, what="Tapestri"
    )
    columns = _sites(parsed, chosen, path=text, itd=None)
    notes: dict[str, Any] = {
        **record,
        "samples": list(dict.fromkeys(cells.samples.tolist())),
        **{
            name: assays_metadata[name]
            for name in _RECORDED_METADATA
            if name in assays_metadata
        },
    }
    return TapestriSource(
        path=text,
        format="tapestri",
        files=(("h5", text),),
        count_unit="reads",
        depth_definition="all",
        count_fields=(),
        call_fields=("quality",),
        depth_roles=frozenset(ALLELE_ROLES),
        reference_from_depth=False,
        alleles_overlap=True,
        open_alleles=False,
        notes=notes,
        skipped=skipped,
        _columns=columns,
        _n_cells=len(cells.ids),
        _cell_arrays=(f"{_VARIANTS}/ra/sample_name", f"{_VARIANTS}/ra/barcode"),
        _layers={
            name: f"{_VARIANTS}/layers/{name}" for name in ("DP", "AF", "NGT", "GQ")
        },
    )


def dab_seq_source(
    path: str | os.PathLike[str],
    include_variants: Sequence[str] = (),
    memory_bytes: int | None = None,
) -> TapestriSource:
    """Describe a DAb-seq ``.hdf5`` file of genotypes as a source of counts.

    DAb-seq flags no variant as failing, so every variant whose alleles are
    bases is read, and ``include_variants`` may name others only to check
    that the file holds them. FLT3-ITD variants are named sites of their own. Before
    it reads them, the description checks what it holds against
    ``memory_bytes``.

    Raises:
        ValueError: If the file lacks one of its arrays, a variant name is
            neither ``gene:chrom:pos:ref/alt`` nor ``chrom:pos:ref/alt``, a
            barcode repeats, ``include_variants`` names an unknown or
            unreadable variant, or no variant is left to import.
        MemoryError: If describing the variants and cells does not fit
            ``memory_bytes``.
    """
    text = os.fspath(path)
    requested = tuple(include_variants)
    with h5py.File(text, "r") as handle:
        names_dataset = _dataset(handle, "VARIANTS", text)
        barcode_dataset = _dataset(handle, "CELL_BARCODES", text)
        _check_description(
            text,
            memory_bytes,
            n_variants=int(names_dataset.shape[0]),
            variant_bytes=_DAB_SEQ_VARIANT_BYTES
            + 2 * _DESCRIBED_TEXT_COPIES * _text_width(names_dataset),
            n_cells=int(barcode_dataset.shape[0]),
            cell_bytes=_CELL_BYTES
            + _DESCRIBED_TEXT_COPIES * _text_width(barcode_dataset),
        )
        names = _texts(names_dataset[:])
        barcodes = _texts(barcode_dataset[:])
    repeated = pd.Index(barcodes).duplicated()
    if repeated.any():
        raise ValueError(
            f"CELL_BARCODES of {text} lists the barcode {quoted_value(barcodes[repeated][0])} twice"
        )
    n_cells = len(barcodes)
    del barcodes, repeated
    contigs, positions, references, alternatives, shorts = [], [], [], [], []
    for index, name in enumerate(names.tolist()):
        parts = name.split(":")
        # The pipeline's non-human mode names variants without a gene.
        if len(parts) == 3:
            parts = ["", *parts]
        alleles = parts[3].split("/") if len(parts) == 4 else []
        if len(alleles) != 2 or not parts[2].isdigit():
            raise ValueError(
                f"VARIANTS[{index}] of {text} is {quoted_value(name)}, neither "
                "<gene>:<chrom>:<pos>:<ref>/<alt> nor <chrom>:<pos>:<ref>/<alt>"
            )
        contigs.append(parts[1])
        positions.append(int(parts[2]))
        references.append(alleles[0].upper())
        alternatives.append(alleles[1].upper())
        shorts.append(":".join(parts[1:]))
    parsed = _Variants(
        names=names,
        contig=np.asarray(contigs, dtype=object),
        position=np.asarray(positions, dtype=np.int64),
        reference=np.asarray(references, dtype=object),
        alternative=np.asarray(alternatives, dtype=object),
        passing=np.ones(len(names), dtype=bool),
    )
    del contigs, positions, references, alternatives
    chosen, record, skipped = _chosen_columns(
        parsed,
        requested,
        short_names=np.asarray(shorts, dtype=object),
        path=text,
        what="DAb-seq",
    )
    itd = np.asarray(
        [str(names[column]).startswith(_ITD_PREFIX) for column in chosen], dtype=bool
    )
    columns = _sites(parsed, chosen, path=text, itd=itd)
    return TapestriSource(
        path=text,
        format="dab_seq",
        files=(("hdf5", text),),
        count_unit="reads",
        depth_definition="all",
        count_fields=(),
        call_fields=("quality",),
        depth_roles=frozenset(ALLELE_ROLES),
        reference_from_depth=False,
        alleles_overlap=False,
        open_alleles=False,
        notes=record,
        skipped=skipped,
        _columns=columns,
        _n_cells=n_cells,
        _cell_arrays=("CELL_BARCODES",),
        _layers={name: name for name in ("DP", "AD", "RD", "GQ", "GT")},
    )
