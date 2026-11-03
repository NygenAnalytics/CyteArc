"""Read the per-cell allele counts of VarTrix's coverage mode.

VarTrix assigns each read that covers a variant of a VCF to the reference or
the alternative haplotype, and in ``--scoring-method coverage`` writes two
Matrix Market files of variants by cells: the alternative reads (the
``--out-matrix``) and the reference reads (the ``--ref-matrix``). Rows follow
the VCF's records and columns the ``--cell-barcodes`` list. VarTrix names
neither matrix by itself, so a folder of its files holds the input VCF, the
barcode list, and the two matrices, the reference one with ``ref`` in its
name. Reads that VarTrix could not assign are left out, so its depth counts
the reference and alternative only. Records with several alternatives keep
empty rows and are skipped, and two records at one position, as a VCF with
split multi-allelic sites holds, are refused, since each counts the
reference reads apart.

Whether the counts are reads or UMIs depends on ``--umi``, which the files do
not say, so the import takes it as ``count_unit``.
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..variants.counts import Contig, EntryBatch, SiteTable
from ._variant_files import (
    MatrixHeader,
    matrix_entries,
    matrix_header,
    read_barcodes,
    read_vcf,
)
from ..utils.arguments import quoted_value

__all__ = ["VartrixSource", "vartrix_source"]


@dataclass(frozen=True, slots=True)
class VartrixSource:
    """A folder of VarTrix's coverage matrices, read as a source of counts."""

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
    _sites: SiteTable = field(repr=False)
    _record_sites: np.ndarray = field(repr=False)
    _barcodes: np.ndarray = field(repr=False)
    _alternative: MatrixHeader = field(repr=False)
    _reference: MatrixHeader = field(repr=False)

    def sites(self) -> SiteTable:
        """Each record with one alternative allele, with its two alleles."""
        return self._sites

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        """The listed cells, then the reference and alternative counts."""
        barcodes = self._barcodes
        yield EntryBatch(
            kind="cells",
            barcodes=barcodes,
            barcode_codes=np.arange(len(barcodes), dtype=np.int64),
        )
        # The site's alleles are its reference, then its alternative.
        for header, offset in ((self._reference, 0), (self._alternative, 1)):
            for rows, columns, values in matrix_entries(header, chunk_rows):
                sites = self._record_sites[rows]
                skipped = sites < 0
                if np.any(skipped & (values > 0)):
                    record = int(rows[skipped & (values > 0)][0])
                    raise ValueError(
                        f"{header.path} counts reads of record {record + 1}, which "
                        "has several alternative alleles; VarTrix leaves such rows "
                        "empty"
                    )
                kept = ~skipped
                yield EntryBatch(
                    kind="counts",
                    barcodes=barcodes,
                    barcode_codes=columns[kept],
                    keys=sites[kept] * 2 + offset,
                    values=values[kept],
                )


def _matrices(folder: str) -> tuple[str, str]:
    names = sorted(
        entry.name
        for entry in os.scandir(folder)
        if entry.is_file() and re.fullmatch(r".*\.mtx(\.gz)?", entry.name)
    )
    reference = [name for name in names if "ref" in name.lower()]
    alternative = [name for name in names if "ref" not in name.lower()]
    if len(names) != 2 or len(reference) != 1:
        raise ValueError(
            f"{folder} must hold VarTrix's two coverage matrices, the reference one "
            f"with 'ref' in its name, not {', '.join(names) or 'none'}"
        )
    return (
        os.path.join(folder, alternative[0]),
        os.path.join(folder, reference[0]),
    )


def _one_file(folder: str, pattern: str, what: str) -> str:
    names = sorted(
        entry.name
        for entry in os.scandir(folder)
        if entry.is_file() and re.fullmatch(pattern, entry.name)
    )
    if len(names) != 1:
        raise ValueError(
            f"{folder} must hold one {what}, not {', '.join(names) or 'none'}"
        )
    return os.path.join(folder, names[0])


def vartrix_source(path: str | os.PathLike[str]) -> VartrixSource:
    """Describe a folder of VarTrix's coverage output as a source of counts.

    Raises:
        ValueError: If the folder does not hold one VCF, one barcode list and
            the two matrices, the VCF lists two records with one alternative
            at a position, a matrix's shape differs from the records and
            cells, or a matrix holds counts that are not whole numbers, as
            the ``alt_frac`` mode writes.
    """
    folder = os.fspath(path)
    alternative_path, reference_path = _matrices(folder)
    vcf_path = _one_file(folder, r".*\.vcf(\.gz)?", "VCF of the variants")
    barcode_path = _one_file(
        folder,
        r".*\.(tsv|txt|csv)(\.gz)?",
        "barcode list (the --cell-barcodes file)",
    )
    vcf = read_vcf(vcf_path)
    barcodes = read_barcodes(barcode_path)
    single = np.asarray(
        [bool(alt) and "," not in alt and alt != "." for alt in vcf.alt.tolist()],
        dtype=bool,
    )
    seen: dict[tuple[str, int], int] = {}
    for record in np.flatnonzero(single).tolist():
        key = (str(vcf.chrom[record]), int(vcf.pos[record]))
        earlier = seen.setdefault(key, record)
        if earlier != record:
            raise ValueError(
                f"{vcf_path} lists two records at {key[0]}:{key[1]}, "
                f"{vcf.ref[earlier]}>{vcf.alt[earlier]} and "
                f"{vcf.ref[record]}>{vcf.alt[record]}, as a VCF with split "
                "multi-allelic sites does; VarTrix counts each record's reference "
                "reads apart, so they cannot be one site. Run VarTrix on a VCF "
                "with one record per position"
            )
    names = list(dict.fromkeys(vcf.chrom[single].tolist()))
    contig_of = {name: position for position, name in enumerate(names)}
    record_sites = np.full(len(vcf.pos), -1, dtype=np.int64)
    record_sites[single] = np.arange(int(single.sum()))
    n_sites = int(single.sum())
    references = vcf.ref[single]
    alternatives = vcf.alt[single]
    same = references == alternatives
    if same.any():
        raise ValueError(
            f"A record of {vcf_path} has the same reference and alternative allele, "
            f"{quoted_value(references[same][0])}"
        )
    sites = SiteTable(
        contigs=tuple(
            Contig(name=name, length=vcf.contig_lengths.get(name)) for name in names
        ),
        site_contig=np.asarray(
            [contig_of[name] for name in vcf.chrom[single].tolist()], dtype=np.int32
        ),
        site_position=vcf.pos[single],
        site_name=np.full(n_sites, "", dtype=object),
        site_channel=np.full(n_sites, "", dtype=object),
        allele_site=np.repeat(np.arange(n_sites, dtype=np.int32), 2),
        allele=np.ravel(np.column_stack([references, alternatives])).astype(object),
        allele_role=np.tile(
            np.asarray(["reference", "alternative"], dtype=object), n_sites
        ),
    )
    headers = []
    for matrix in (alternative_path, reference_path):
        header = matrix_header(matrix)
        if (header.n_rows, header.n_columns) != (len(vcf.pos), len(barcodes)):
            raise ValueError(
                f"{matrix} is a {header.n_rows} by {header.n_columns} matrix, but "
                f"{vcf_path} holds {len(vcf.pos)} records and {barcode_path} "
                f"{len(barcodes)} barcodes"
            )
        headers.append(header)
    return VartrixSource(
        path=folder,
        format="vartrix",
        files=(
            ("alternative", alternative_path),
            ("reference", reference_path),
            ("vcf", vcf_path),
            ("barcodes", barcode_path),
        ),
        count_unit=None,
        depth_definition="reference_and_alternative",
        count_fields=(),
        call_fields=(),
        depth_roles=None,
        reference_from_depth=False,
        alleles_overlap=False,
        open_alleles=False,
        notes={"n_skipped_records": int((~single).sum())},
        _sites=sites,
        _record_sites=record_sites,
        _barcodes=barcodes,
        _alternative=headers[0],
        _reference=headers[1],
    )
