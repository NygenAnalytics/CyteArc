"""Gene annotations, transcription start sites, and region-to-gene windows."""

import gzip
import os
import re
from collections.abc import Iterable, Iterator
from typing import Literal, TextIO
from urllib.parse import unquote

import numpy as np
import pandas as pd
from numba import njit

from ...utils.arguments import integer_argument
from .sequence import parse_coordinate_ids

__all__ = [
    "link_regions_to_genes",
    "overlapping_intervals",
    "read_gene_annotation",
    "transcription_start_sites",
]

GENE_COLUMNS = ("chrom", "start", "end", "strand", "gene_id", "gene_name", "gene_type")

_GTF_ATTRIBUTE = re.compile(r'\s*([^\s";]+)\s+"?([^";]*)"?\s*')


def open_text(path: str | os.PathLike[str]) -> TextIO:
    """Open a text file, gzip-compressed (by its magic bytes, not its name) or not."""
    with open(path, "rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def shared_chromosomes(regions: np.ndarray, others: np.ndarray, what: str) -> list[str]:
    """Chromosomes named by both inputs; none at all means mismatched naming."""
    shared = sorted(set(regions.tolist()) & set(others.tolist()))
    if not shared and len(regions) and len(others):
        raise ValueError(
            f"No chromosome is named in both the regions (such as {regions[0]!r}) "
            f"and {what} (such as {others[0]!r}); use one naming, for example "
            "with or without a 'chr' prefix"
        )
    return shared


def _gtf_attributes(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in raw.split(";"):
        match = _GTF_ATTRIBUTE.fullmatch(item) if item.strip() else None
        if match is not None:
            values.setdefault(match.group(1), match.group(2))
    return values


def _gff3_attributes(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in raw.strip().split(";"):
        key, equals, value = item.partition("=")
        if equals:
            # GFF3 percent-encodes reserved characters in values.
            values.setdefault(key.strip(), unquote(value.strip()))
    return values


def _feature_rows(
    handle: TextIO, *, gff3: bool, features: frozenset[str]
) -> Iterator[tuple[str, int, int, str, str, str, str]]:
    for line_number, raw in enumerate(handle, start=1):
        # A GFF3 file may end with its sequences.
        if raw.startswith("##FASTA"):
            break
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.rstrip("\r\n").split("\t")
        if len(fields) != 9:
            raise ValueError(f"Annotation line {line_number} does not have 9 columns")
        if fields[2] not in features:
            continue
        attributes = _gff3_attributes(fields[8]) if gff3 else _gtf_attributes(fields[8])
        # Ensembl and NCBI prefix GFF3 gene IDs with "gene:" and "gene-".
        gene_id = attributes.get("gene_id") or (
            re.sub(r"^gene[:-]", "", attributes.get("ID", "")) if gff3 else ""
        )
        gene_name = attributes.get("Name" if gff3 else "gene_name") or (
            attributes.get("gene_name") or gene_id
        )
        gene_type = (
            attributes.get("biotype")
            or attributes.get("gene_type")
            or attributes.get("gene_biotype", "")
        )
        if not gene_id:
            raise ValueError(f"Annotation line {line_number} has no gene identifier")
        strand = fields[6]
        if strand not in {"+", "-"}:
            raise ValueError(f"Annotation line {line_number} has no +/- strand")
        # GTF and GFF3 are one-based and closed; convert to BED intervals.
        start = int(fields[3]) - 1
        end = int(fields[4])
        if start < 0 or end <= start:
            raise ValueError(f"Annotation line {line_number} has an invalid interval")
        yield fields[0], start, end, strand, gene_id, gene_name, gene_type


def read_gene_annotation(
    path: str | os.PathLike[str],
    *,
    feature: str | Iterable[str] = "gene",
    gene_types: set[str] | frozenset[str] | None = None,
) -> pd.DataFrame:
    """Read gene intervals and strands from a GTF, GFF3, or BED file.

    GTF and GFF3 files (optionally gzipped) contribute their ``feature``
    records, converted to zero-based half-open coordinates. BED files need
    at least six columns: chrom, start, end, gene name, score, and strand; a
    file of exactly seven columns gives the gene identifier in the seventh
    (in BED12, the seventh column is thickStart).

    Args:
        path: Annotation file. The format follows the extension: ``.gtf``,
            ``.gff3``/``.gff``, or ``.bed``, each optionally ``.gz``.
        feature: GTF/GFF3 feature type, or types, to read. Ensembl GFF3
            types non-coding genes ``ncRNA_gene``.
        gene_types: Keep only these gene types (GTF ``gene_type`` or
            ``gene_biotype``, GFF3 ``biotype``), such as
            ``{"protein_coding"}``. BED files carry no type, so they reject
            this filter.

    Returns:
        A DataFrame with ``chrom``, ``start``, ``end``, ``strand``,
        ``gene_id``, ``gene_name``, and ``gene_type`` columns.
    """
    name = os.fspath(path).lower().removesuffix(".gz")
    if name.endswith(".bed"):
        if gene_types is not None:
            raise ValueError("BED annotations carry no gene types to filter")
        with open_text(path) as handle:
            # Fields stay text: gene names such as "NA" are not missing values.
            lines = [
                raw.rstrip("\r\n").split("\t")
                for raw in handle
                if raw.strip() and not raw.startswith(("#", "track", "browser"))
            ]
        if not lines or min(len(fields) for fields in lines) < 6:
            raise ValueError("A BED gene annotation needs at least six columns")
        frame = pd.DataFrame(
            {
                "chrom": [fields[0] for fields in lines],
                "start": np.asarray([fields[1] for fields in lines], dtype=np.int64),
                "end": np.asarray([fields[2] for fields in lines], dtype=np.int64),
                "strand": [fields[5] for fields in lines],
                "gene_id": [
                    fields[6] if len(fields) == 7 else fields[3] for fields in lines
                ],
                "gene_name": [fields[3] for fields in lines],
                "gene_type": "",
            }
        )
        if not frame["strand"].isin(["+", "-"]).all():
            raise ValueError("BED column 6 must hold + or - strands")
        if (frame["start"] < 0).any() or (frame["end"] <= frame["start"]).any():
            raise ValueError("BED gene annotation holds an invalid interval")
    elif name.endswith((".gtf", ".gff3", ".gff")):
        features = frozenset([feature] if isinstance(feature, str) else feature)
        with open_text(path) as handle:
            rows = list(
                _feature_rows(handle, gff3=not name.endswith(".gtf"), features=features)
            )
        frame = pd.DataFrame.from_records(rows, columns=list(GENE_COLUMNS))
        if gene_types is not None:
            frame = frame.loc[frame["gene_type"].isin(set(gene_types))]
    else:
        raise ValueError("Gene annotations must be .gtf, .gff3, .gff, or .bed files")
    if frame.empty:
        raise ValueError("The gene annotation contains no genes")
    return frame.reset_index(drop=True).astype({"start": np.int64, "end": np.int64})


def transcription_start_sites(genes: pd.DataFrame) -> np.ndarray:
    """Return each gene's zero-based TSS: its start on ``+`` and ``end - 1`` on ``-``."""
    strand = genes["strand"].to_numpy()
    if not np.isin(strand, ["+", "-"]).all():
        raise ValueError("Gene strands must be + or -")
    starts = genes["start"].to_numpy(dtype=np.int64)
    ends = genes["end"].to_numpy(dtype=np.int64)
    sites: np.ndarray = np.where(strand == "+", starts, ends - 1).astype(np.int64)
    return sites


@njit(cache=True)
def _overlap_pairs(
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    max_region_length: int,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair windows with the regions they overlap.

    Regions are sorted by start. A region overlaps a window when it starts
    before the window ends and ends after the window starts.
    """
    total = 0
    lows = np.searchsorted(region_starts, window_starts - max_region_length)
    highs = np.searchsorted(region_starts, window_ends)
    for window in range(window_starts.size):
        for region in range(lows[window], highs[window]):
            if region_ends[region] > window_starts[window]:
                total += 1
    region_index = np.empty(total, dtype=np.int64)
    window_index = np.empty(total, dtype=np.int64)
    position = 0
    for window in range(window_starts.size):
        for region in range(lows[window], highs[window]):
            if region_ends[region] > window_starts[window]:
                region_index[position] = region
                window_index[position] = window
                position += 1
    return region_index, window_index


def overlapping_intervals(
    query: tuple[np.ndarray, np.ndarray, np.ndarray],
    subject: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair intervals of ``query`` and ``subject`` that share at least one base.

    Args:
        query, subject: ``(chroms, starts, ends)`` of zero-based half-open
            intervals.

    Returns:
        Query indices, subject indices, and the number of shared bases of each
        overlapping pair, sorted by query, then subject.
    """
    q_chroms, q_starts, q_ends = (np.asarray(values) for values in query)
    s_chroms, s_starts, s_ends = (np.asarray(values) for values in subject)
    q_starts = q_starts.astype(np.int64)
    q_ends = q_ends.astype(np.int64)
    s_starts = s_starts.astype(np.int64)
    s_ends = s_ends.astype(np.int64)
    if np.any(q_ends <= q_starts) or np.any(s_ends <= s_starts):
        raise ValueError("Every interval needs start < end")
    query_rows: list[np.ndarray] = []
    subject_rows: list[np.ndarray] = []
    for chrom in sorted(set(q_chroms.tolist()) & set(s_chroms.tolist())):
        subject_index = np.flatnonzero(s_chroms == chrom)
        order = subject_index[np.argsort(s_starts[subject_index], kind="stable")]
        query_index = np.flatnonzero(q_chroms == chrom)
        lengths = s_ends[order] - s_starts[order]
        found_subject, found_query = _overlap_pairs(
            s_starts[order],
            s_ends[order],
            int(lengths.max()),
            q_starts[query_index],
            q_ends[query_index],
        )
        query_rows.append(query_index[found_query])
        subject_rows.append(order[found_subject])
    if not query_rows:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    query_found = np.concatenate(query_rows)
    subject_found = np.concatenate(subject_rows)
    order = np.lexsort((subject_found, query_found))
    query_found = query_found[order]
    subject_found = subject_found[order]
    shared = np.minimum(q_ends[query_found], s_ends[subject_found]) - np.maximum(
        q_starts[query_found], s_starts[subject_found]
    )
    return query_found, subject_found, shared.astype(np.int64)


def link_regions_to_genes(
    regions: Iterable[str],
    genes: pd.DataFrame,
    *,
    upstream: int,
    downstream: int,
    anchor: Literal["tss", "gene_body"] = "tss",
) -> pd.DataFrame:
    """Link regions to genes whose window they overlap.

    With ``anchor="tss"`` each gene's window spans ``upstream`` bases before
    and ``downstream`` bases after its TSS in the direction of transcription,
    so the window of a minus-strand gene extends to higher coordinates
    upstream. With ``anchor="gene_body"`` the window is the gene body extended
    by ``upstream`` bases before its 5' end and ``downstream`` bases after its
    3' end, the SCENIC+ search space (150 kb each way by default there).

    Args:
        regions: Region ids ``chrom:start-end`` (zero-based, half-open), such
            as ATAC peak ids.
        genes: Annotation from ``read_gene_annotation``.
        upstream, downstream: Window extent in bases.
        anchor: ``"tss"`` or ``"gene_body"``.

    Returns:
        A DataFrame with one row per overlapping region and gene: ``region``
        (the region id), ``gene`` (the gene name), ``gene_id``, and
        ``distance``, the signed distance from the anchor to the nearest base
        of the region, negative upstream, positive downstream, and zero when
        the region overlaps the TSS (or the gene body). Rows follow the order
        of the regions, then of the gene ids, and are the candidates that
        ``region_gene_links`` scores.
    """
    if anchor not in ("tss", "gene_body"):
        raise ValueError("anchor must be 'tss' or 'gene_body'")
    upstream = integer_argument(upstream, "upstream", minimum=0)
    downstream = integer_argument(downstream, "downstream", minimum=0)
    region_names = np.asarray(list(regions), dtype=object).astype(str)
    chrom_values, start_values, end_values = parse_coordinate_ids(region_names)
    tss = transcription_start_sites(genes)
    plus = genes["strand"].to_numpy() == "+"
    if anchor == "tss":
        anchor_starts = tss
        anchor_ends = tss + 1
    else:
        anchor_starts = genes["start"].to_numpy(dtype=np.int64)
        anchor_ends = genes["end"].to_numpy(dtype=np.int64)
    window_starts = np.where(plus, anchor_starts - upstream, anchor_starts - downstream)
    window_ends = np.where(plus, anchor_ends + downstream, anchor_ends + upstream)
    gene_chroms = genes["chrom"].to_numpy(dtype=object)

    region_rows_found: list[np.ndarray] = []
    gene_rows: list[np.ndarray] = []
    for chrom in shared_chromosomes(chrom_values, gene_chroms, "the genes"):
        region_rows = np.flatnonzero(chrom_values == chrom)
        order = region_rows[np.argsort(start_values[region_rows], kind="stable")]
        windows = np.flatnonzero(gene_chroms == chrom)
        sorted_starts = start_values[order]
        sorted_ends = end_values[order]
        lengths = sorted_ends - sorted_starts
        region_index, window_index = _overlap_pairs(
            sorted_starts,
            sorted_ends,
            int(lengths.max()),
            window_starts[windows],
            window_ends[windows],
        )
        region_rows_found.append(order[region_index])
        gene_rows.append(windows[window_index])
    if region_rows_found:
        region_index = np.concatenate(region_rows_found)
        gene_index = np.concatenate(gene_rows)
    else:
        region_index = np.zeros(0, dtype=np.int64)
        gene_index = np.zeros(0, dtype=np.int64)

    first = anchor_starts[gene_index]
    last = anchor_ends[gene_index] - 1
    region_start = start_values[region_index]
    region_last = end_values[region_index] - 1
    # Offset of the region from the anchor in genome coordinates.
    offset = np.where(
        region_start > last,
        region_start - last,
        np.where(region_last < first, region_last - first, 0),
    )
    distance = np.where(plus[gene_index], offset, -offset)
    links = pd.DataFrame(
        {
            "region": region_index,
            "gene": genes["gene_name"].to_numpy(dtype=object)[gene_index],
            "gene_id": genes["gene_id"].to_numpy(dtype=object)[gene_index],
            "distance": distance.astype(np.int64),
        }
    )
    links = links.sort_values(["region", "gene_id"], kind="stable")
    links["region"] = region_names[links["region"].to_numpy()]
    return links.reset_index(drop=True)
