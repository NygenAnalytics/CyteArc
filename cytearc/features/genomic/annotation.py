"""Gene annotations, transcription start sites, and region-to-gene windows."""

import gzip
import os
import re
from collections.abc import Iterator
from typing import TextIO

from typing import Literal

import numpy as np
import pandas as pd
from numba import njit

__all__ = [
    "link_regions_to_genes",
    "overlapping_intervals",
    "read_gene_annotation",
    "transcription_start_sites",
]

GENE_COLUMNS = ("chrom", "start", "end", "strand", "gene_id", "gene_name", "gene_type")

_GTF_ATTRIBUTE = re.compile(r'\s*([^\s";]+)\s+"?([^";]*)"?\s*')


def _open_text(path: str | os.PathLike[str]) -> TextIO:
    if os.fspath(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


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
            values.setdefault(key.strip(), value.strip())
    return values


def _feature_rows(
    handle: TextIO, *, gff3: bool, feature: str
) -> Iterator[tuple[str, int, int, str, str, str, str]]:
    for line_number, raw in enumerate(handle, start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.rstrip("\r\n").split("\t")
        if len(fields) != 9:
            raise ValueError(f"Annotation line {line_number} does not have 9 columns")
        if fields[2] != feature:
            continue
        attributes = _gff3_attributes(fields[8]) if gff3 else _gtf_attributes(fields[8])
        if gff3:
            gene_id = attributes.get("gene_id") or attributes.get("ID", "")
            gene_id = (
                gene_id.split(":", 1)[1] if gene_id.startswith("gene:") else gene_id
            )
            gene_name = attributes.get("Name") or attributes.get("gene_name") or gene_id
            gene_type = attributes.get("biotype") or attributes.get("gene_type", "")
        else:
            gene_id = attributes.get("gene_id", "")
            gene_name = attributes.get("gene_name") or gene_id
            gene_type = attributes.get("gene_type") or attributes.get(
                "gene_biotype", ""
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
    feature: str = "gene",
    gene_types: set[str] | frozenset[str] | None = None,
) -> pd.DataFrame:
    """Read gene intervals and strands from a GTF, GFF3, or BED file.

    GTF and GFF3 files (optionally gzipped) contribute their ``feature``
    records, converted to zero-based half-open coordinates. BED files need
    at least six columns: chrom, start, end, gene name, score, and strand; a
    seventh column, if present, is read as the gene identifier.

    Args:
        path: Annotation file. The format follows the extension: ``.gtf``,
            ``.gff3``/``.gff``, or ``.bed``, each optionally ``.gz``.
        feature: GTF/GFF3 feature type to read.
        gene_types: Keep only these gene types (GTF ``gene_type`` or GFF3
            ``biotype``), such as ``{"protein_coding"}``. BED files carry no
            type, so they reject this filter.

    Returns:
        A DataFrame with ``chrom``, ``start``, ``end``, ``strand``,
        ``gene_id``, ``gene_name``, and ``gene_type`` columns.
    """
    name = os.fspath(path).lower().removesuffix(".gz")
    if name.endswith(".bed"):
        if gene_types is not None:
            raise ValueError("BED annotations carry no gene types to filter")
        table = pd.read_csv(path, sep="\t", header=None, comment="#")
        if table.shape[1] < 6:
            raise ValueError("A BED gene annotation needs at least six columns")
        frame = pd.DataFrame(
            {
                "chrom": table[0].astype(str),
                "start": table[1].astype(np.int64),
                "end": table[2].astype(np.int64),
                "strand": table[5].astype(str),
                "gene_id": (table[6] if table.shape[1] > 6 else table[3]).astype(str),
                "gene_name": table[3].astype(str),
                "gene_type": "",
            }
        )
        if not frame["strand"].isin(["+", "-"]).all():
            raise ValueError("BED column 6 must hold + or - strands")
        if (frame["start"] < 0).any() or (frame["end"] <= frame["start"]).any():
            raise ValueError("BED gene annotation holds an invalid interval")
    elif name.endswith((".gtf", ".gff3", ".gff")):
        with _open_text(path) as handle:
            rows = list(
                _feature_rows(handle, gff3=not name.endswith(".gtf"), feature=feature)
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
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
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
        chroms, starts, ends: Zero-based half-open regions.
        genes: Annotation from ``read_gene_annotation``.
        upstream, downstream: Window extent in bases.
        anchor: ``"tss"`` or ``"gene_body"``.

    Returns:
        A DataFrame with one row per overlapping region and gene:
        ``region`` (index into the inputs), ``gene_id``, ``gene_name``, and
        ``distance``, the signed distance from the anchor to the nearest base
        of the region, negative upstream, positive downstream, and zero when
        the region overlaps the TSS (or the gene body). Rows are sorted by
        region, then gene.
    """
    if anchor not in ("tss", "gene_body"):
        raise ValueError("anchor must be 'tss' or 'gene_body'")
    for label, value in (("upstream", upstream), ("downstream", downstream)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    chrom_values = np.asarray(chroms, dtype=object)
    start_values = np.asarray(starts, dtype=np.int64)
    end_values = np.asarray(ends, dtype=np.int64)
    if not (len(chrom_values) == len(start_values) == len(end_values)):
        raise ValueError("chroms, starts, and ends must be aligned")
    if np.any(end_values <= start_values):
        raise ValueError("Every region needs start < end")
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

    regions: list[np.ndarray] = []
    gene_rows: list[np.ndarray] = []
    for chrom in sorted(set(chrom_values.tolist()) & set(gene_chroms.tolist())):
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
        regions.append(order[region_index])
        gene_rows.append(windows[window_index])
    if regions:
        region_index = np.concatenate(regions)
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
            "gene_id": genes["gene_id"].to_numpy(dtype=object)[gene_index],
            "gene_name": genes["gene_name"].to_numpy(dtype=object)[gene_index],
            "distance": distance.astype(np.int64),
        }
    )
    return links.sort_values(["region", "gene_id"], kind="stable").reset_index(
        drop=True
    )
