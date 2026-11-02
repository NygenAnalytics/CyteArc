"""Read the per-cell allele counts of cellsnp-lite and MQuad.

cellsnp-lite counts, for each cell and each variant of its ``cellSNP.base.vcf``,
the reads or UMIs of the alternative allele (``cellSNP.tag.AD.mtx``), of the
reference and alternative together (``cellSNP.tag.DP.mtx``) and of the other
bases (``cellSNP.tag.OTH.mtx``), as Matrix Market files of variants by cells
whose columns follow ``cellSNP.samples.tsv``. Its depth at a site is DP plus
OTH, and the reference count is DP less AD. cellsnp-lite is bi-allelic, so
each record is one site with one alternative.

MQuad selects informative mitochondrial variants from cellsnp-lite's counts
and writes their AD and DP as ``passed_ad.mtx`` and ``passed_dp.mtx``, with
names such as ``2956C>A`` in ``passed_variant_names.txt``. It writes no cell
list, so the folder must also hold the ``cellSNP.samples.tsv`` of the
cellsnp-lite run it read, and it writes no contig, which its
``BIC_params.csv`` gives where present (``chrM_2956_C_A``). Its depth counts
the reference and alternative only.

Whether the counts are reads or UMIs depends on how cellsnp-lite ran, which
the files do not say, so the import takes it as ``count_unit``.
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..variants.counts import Contig, EntryBatch, SiteTable
from ._variant_files import (
    MatrixHeader,
    find_file,
    is_gzip,
    matrix_entries,
    matrix_header,
    open_text,
    read_barcodes,
    read_vcf,
)

__all__ = ["CellsnpSource", "cellsnp_source"]

#: The allele that holds the bases other than a record's reference and
#: alternative.
OTHER_ALLELE = "other"
_VARIANT_NAME = re.compile(r"^(?P<position>\d+)(?P<ref>[A-Za-z]+)>(?P<alt>[A-Za-z]+)$")
_BIC_NAME = re.compile(
    r"^(?P<contig>.+)_(?P<position>\d+)_(?P<ref>[A-Za-z]+)_(?P<alt>[A-Za-z]+)$"
)


@dataclass(frozen=True, slots=True)
class _Records:
    contigs: tuple[Contig, ...]
    contig: np.ndarray
    position: np.ndarray
    ref: np.ndarray
    alt: np.ndarray


def _check_records(records: _Records, path: str) -> None:
    for name, values in (("reference", records.ref), ("alternative", records.alt)):
        bad = [value for value in values.tolist() if not value or "," in value]
        if bad:
            raise ValueError(
                f"{path} holds the {name} allele {bad[0]!r}; each record needs one "
                "reference and one alternative allele"
            )
    same = records.ref == records.alt
    if same.any():
        position = int(np.flatnonzero(same)[0])
        raise ValueError(
            f"Record {position + 1} of {path} has the same reference and alternative "
            f"allele, {records.ref[position]!r}"
        )
    keys = pd.Series(
        list(zip(records.contig.tolist(), records.position.tolist(), strict=True))
    )
    repeated = keys.duplicated().to_numpy()
    if repeated.any():
        position = int(np.flatnonzero(repeated)[0])
        contig = records.contigs[int(records.contig[position])].name
        raise ValueError(
            f"{path} lists two variants at {contig + ':' if contig else ''}"
            f"{int(records.position[position])}; it counts one alternative per site"
        )


def _site_table(records: _Records, *, other: bool) -> SiteTable:
    n_sites = len(records.position)
    alleles: list[str] = []
    roles: list[str] = []
    allele_site: list[int] = []
    for site in range(n_sites):
        texts = [str(records.ref[site]), str(records.alt[site])]
        site_roles = ["reference", "alternative"]
        if other:
            texts.append(OTHER_ALLELE)
            site_roles.append("other")
        alleles.extend(texts)
        roles.extend(site_roles)
        allele_site.extend([site] * len(texts))
    return SiteTable(
        contigs=records.contigs,
        site_contig=records.contig,
        site_position=records.position,
        site_name=np.full(n_sites, "", dtype=object),
        site_channel=np.full(n_sites, "", dtype=object),
        allele_site=np.asarray(allele_site, dtype=np.int32),
        allele=np.asarray(alleles, dtype=object),
        allele_role=np.asarray(roles, dtype=object),
    )


@dataclass(frozen=True, slots=True)
class CellsnpSource:
    """A cellsnp-lite or MQuad folder, read as a source of allele counts."""

    path: str
    format: str
    files: tuple[tuple[str, str], ...]
    count_unit: str | None
    depth_definition: str
    count_fields: tuple[str, ...]
    depth_roles: frozenset[str] | None
    reference_from_depth: bool
    alleles_overlap: bool
    open_alleles: bool
    notes: dict[str, Any]
    _records: _Records = field(repr=False)
    _barcodes: np.ndarray = field(repr=False)
    _matrices: dict[str, MatrixHeader] = field(repr=False)

    def sites(self) -> SiteTable:
        """Each record's site, with its reference, its alternative and, where
        cellsnp-lite counted them, the other bases."""
        return _site_table(self._records, other="OTH" in self._matrices)

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        """The listed cells, then the alternative counts, the other bases'
        counts and the depth, in chunks."""
        barcodes = self._barcodes
        yield EntryBatch(
            kind="cells",
            barcodes=barcodes,
            barcode_codes=np.arange(len(barcodes), dtype=np.int64),
        )
        width = 3 if "OTH" in self._matrices else 2
        for tag, offset in (("AD", 1), ("OTH", 2)):
            if tag not in self._matrices:
                continue
            for rows, columns, values in matrix_entries(
                self._matrices[tag], chunk_rows
            ):
                yield EntryBatch(
                    kind="counts",
                    barcodes=barcodes,
                    barcode_codes=columns,
                    keys=rows * width + offset,
                    values=values,
                )
        for rows, columns, values in matrix_entries(self._matrices["DP"], chunk_rows):
            yield EntryBatch(
                kind="depth",
                barcodes=barcodes,
                barcode_codes=columns,
                keys=rows,
                values=values,
            )


def _check_shape(header: MatrixHeader, n_records: int, n_cells: int) -> None:
    if (header.n_rows, header.n_columns) != (n_records, n_cells):
        raise ValueError(
            f"{header.path} is a {header.n_rows} by {header.n_columns} matrix, but its "
            f"folder lists {n_records} variants and {n_cells} cells"
        )


def _vcf_records(path: str) -> _Records:
    vcf = read_vcf(path)
    names = list(dict.fromkeys(vcf.chrom.tolist()))
    contigs = tuple(
        Contig(name=name, length=vcf.contig_lengths.get(name)) for name in names
    )
    index = {name: position for position, name in enumerate(names)}
    return _Records(
        contigs=contigs,
        contig=np.asarray([index[name] for name in vcf.chrom.tolist()], dtype=np.int32),
        position=vcf.pos,
        ref=vcf.ref,
        alt=vcf.alt,
    )


def _mquad_records(names_path: str, bic_path: str | None) -> _Records:
    with open_text(names_path) as handle:
        names = [line.strip() for line in handle.read().splitlines() if line.strip()]
    parsed = []
    for name in names:
        match = _VARIANT_NAME.match(name)
        if match is None:
            raise ValueError(
                f"{names_path} names the variant {name!r}; MQuad names variants as "
                "position, reference, '>' and alternative, such as '2956C>A'"
            )
        parsed.append(
            (int(match["position"]), match["ref"].upper(), match["alt"].upper())
        )
    contig_name = ""
    if bic_path is not None:
        table = pd.read_csv(
            bic_path,
            usecols=["variant_name"],
            dtype=str,
            keep_default_na=False,
            compression="gzip" if is_gzip(bic_path) else None,
        )
        by_variant: dict[tuple[int, str, str], str] = {}
        for value in table["variant_name"].tolist():
            match = _BIC_NAME.match(value)
            if match is None:
                raise ValueError(
                    f"{bic_path} names the variant {value!r}; MQuad names them as "
                    "contig, position, reference and alternative joined by '_'"
                )
            key = (int(match["position"]), match["ref"].upper(), match["alt"].upper())
            by_variant[key] = match["contig"]
        missing = [
            name
            for name, key in zip(names, parsed, strict=True)
            if key not in by_variant
        ]
        if missing:
            raise ValueError(
                f"{bic_path} does not list the passed variant {missing[0]!r}"
            )
        contig_names = {by_variant[key] for key in parsed}
        if len(contig_names) > 1:
            raise ValueError(
                f"{bic_path} places the passed variants on several contigs, "
                f"{sorted(contig_names)}"
            )
        contig_name = contig_names.pop() if contig_names else ""
    return _Records(
        contigs=(Contig(name=contig_name),),
        contig=np.zeros(len(parsed), dtype=np.int32),
        position=np.asarray([key[0] for key in parsed], dtype=np.int64),
        ref=np.asarray([key[1] for key in parsed], dtype=object),
        alt=np.asarray([key[2] for key in parsed], dtype=object),
    )


def cellsnp_source(path: str | os.PathLike[str], *, format: str) -> CellsnpSource:
    """Describe a cellsnp-lite or MQuad folder as a source of allele counts.

    Args:
        path: The folder.
        format: ``"cellsnp_lite"`` or ``"mquad"``.

    Raises:
        FileNotFoundError: If a file of the format is missing, such as the
            cell list beside MQuad's matrices.
        ValueError: If a file is malformed, a matrix's shape differs from the
            variants and cells, or a record has no single alternative.
    """
    folder = os.fspath(path)
    files: list[tuple[str, str]] = []
    if format == "cellsnp_lite":
        vcf = find_file(
            folder, r"(.*[._])?base\.vcf(\.gz)?", "base VCF (cellSNP.base.vcf)"
        )
        samples = find_file(
            folder, r"(.*[._])?samples\.tsv(\.gz)?", "cell list (cellSNP.samples.tsv)"
        )
        tags = {
            tag: find_file(
                folder,
                rf"(.*\.)?tag\.{tag}\.mtx(\.gz)?",
                f"{tag} matrix (cellSNP.tag.{tag}.mtx)",
                required=tag != "OTH",
            )
            for tag in ("AD", "DP", "OTH")
        }
        assert vcf is not None and samples is not None
        records = _vcf_records(vcf)
        _check_records(records, vcf)
        files += [("vcf", vcf), ("samples", samples)]
    elif format == "mquad":
        names = find_file(
            folder,
            r"(.*_)?passed_variant_names\.txt(\.gz)?",
            "variant names (passed_variant_names.txt)",
        )
        samples = find_file(
            folder,
            r"(.*[._])?samples\.tsv(\.gz)?",
            "cell list; MQuad writes none, so copy the cellSNP.samples.tsv of the "
            "cellsnp-lite run it read into the folder",
        )
        bic = find_file(
            folder, r"(.*_)?BIC_params\.csv(\.gz)?", "BIC_params.csv", required=False
        )
        tags = {
            "AD": find_file(folder, r"(.*_)?passed_ad\.mtx(\.gz)?", "passed_ad.mtx"),
            "DP": find_file(folder, r"(.*_)?passed_dp\.mtx(\.gz)?", "passed_dp.mtx"),
        }
        assert names is not None and samples is not None
        records = _mquad_records(names, bic)
        _check_records(records, names)
        files += [("variant_names", names), ("samples", samples)]
        if bic is not None:
            files.append(("bic_params", bic))
    else:
        raise ValueError(f"format must be cellsnp_lite or mquad, not {format!r}")
    barcodes = read_barcodes(samples)
    matrices = {
        tag: matrix_header(file) for tag, file in tags.items() if file is not None
    }
    for tag, header in matrices.items():
        _check_shape(header, len(records.position), len(barcodes))
        files.append((tag, header.path))
    other = "OTH" in matrices
    return CellsnpSource(
        path=folder,
        format=format,
        files=tuple(files),
        count_unit=None,
        depth_definition="all" if other else "reference_and_alternative",
        count_fields=(),
        depth_roles=frozenset({"reference", "alternative"}),
        reference_from_depth=True,
        alleles_overlap=False,
        open_alleles=False,
        notes={},
        _records=records,
        _barcodes=barcodes,
        _matrices=matrices,
    )
