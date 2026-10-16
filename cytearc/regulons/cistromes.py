"""Cistromes: the regions a transcription factor binds or can bind.

A cistrome comes either from motif matches, assigned to factors by a motif
annotation table as SCENIC+ does, or from an experiment that measured binding
directly, such as CUT&RUN, CUT&Tag, or ChIP-seq peaks of one factor.
"""

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..features.genomic.annotation import (
    open_text,
    overlapping_intervals,
    shared_chromosomes,
)
from ..features.genomic.sequence import parse_coordinate_ids

__all__ = [
    "binding_cistromes",
    "cistromes_from_motifs",
    "read_peak_file",
]

PEAK_COLUMNS = ("chrom", "start", "end", "name", "score", "summit")


def read_peak_file(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read peaks from a BED, narrowPeak, or SEACR BED file, optionally gzipped.

    Coordinates are kept as zero-based half-open BED intervals. Track and
    browser lines and ``#`` comments are skipped.

    Returns:
        A DataFrame with ``chrom``, ``start``, ``end``, ``name``, ``score``,
        and ``summit`` columns; ``score`` is the fifth column when present and
        ``summit`` the narrowPeak summit offset (-1 when absent).
    """
    rows = []
    with open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(f"Peak line {line_number} has fewer than 3 columns")
            try:
                start = int(fields[1])
                end = int(fields[2])
                score = (
                    float(fields[4])
                    if len(fields) > 4 and fields[4] not in {"", "."}
                    else np.nan
                )
                # Only narrowPeak's ten columns end with a summit offset.
                summit = int(fields[9]) if len(fields) == 10 else -1
            except ValueError as exc:
                raise ValueError(f"Peak line {line_number} is malformed") from exc
            if start < 0 or end <= start:
                raise ValueError(f"Peak line {line_number} has start >= end")
            name = fields[3] if len(fields) > 3 else f"{fields[0]}:{start}-{end}"
            rows.append((fields[0], start, end, name, score, summit))
    if not rows:
        raise ValueError(f"{os.fspath(path)} contains no peaks")
    return pd.DataFrame.from_records(rows, columns=list(PEAK_COLUMNS))


def cistromes_from_motifs(
    motif_network: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    categories: Sequence[str] = ("direct",),
) -> pd.DataFrame:
    """Assign motif-matched regions to the factors their motifs are annotated to.

    Each factor's cistrome is the union of the regions matched by every motif
    the annotation table assigns to it in ``categories``. SCENIC+ builds
    ``direct`` cistromes from direct annotations and ``extended`` cistromes
    from orthology annotations only; pass ``categories=("orthology",)`` for the
    latter.

    Args:
        motif_network: ``source`` (motif ID) and ``target`` (region) columns,
            as ``match_motifs`` returns, or the leading-edge regions of
            enriched motifs.
        annotations: Output of ``read_motif_annotations``.
        categories: Annotation categories that link a motif to a factor.

    Returns:
        A network with ``source`` (factor), ``target`` (region), and
        ``motifs`` (sorted, comma-separated motif IDs that support the link).
    """
    if not {"source", "target"} <= set(motif_network.columns):
        raise ValueError("motif_network needs source and target columns")
    if not {"motif", "TF", "category"} <= set(annotations.columns):
        raise ValueError("annotations must come from read_motif_annotations")
    chosen = annotations.loc[
        annotations["category"].isin(list(categories)), ["motif", "TF"]
    ].drop_duplicates()
    matches = motif_network.loc[:, ["source", "target"]].rename(
        columns={"source": "motif"}
    )
    joined = matches.merge(chosen, on="motif", how="inner")
    if joined.empty:
        return pd.DataFrame({"source": [], "target": [], "motifs": []})
    grouped = joined.groupby(["TF", "target"], sort=True)["motif"].agg(
        lambda motifs: ",".join(sorted(set(motifs)))
    )
    result = grouped.reset_index().rename(columns={"TF": "source", "motif": "motifs"})
    return result.loc[:, ["source", "target", "motifs"]]


def binding_cistromes(
    peaks: Mapping[str, pd.DataFrame | str | os.PathLike[str]],
    regions: Sequence[str] | np.ndarray,
    *,
    min_overlap_fraction: float = 0.0,
) -> pd.DataFrame:
    """Map each factor's binding peaks onto a set of regions.

    A region joins a factor's cistrome when one of the factor's peaks overlaps
    it. With ``min_overlap_fraction`` above zero, the shared bases must also
    exceed that fraction of the peak or of the region, the rule pycistarget
    uses to map regions onto its databases (``0.4``).

    Args:
        peaks: Factor name to binding peaks, as a ``read_peak_file`` table or
            a path such as a CUT&RUN narrowPeak file.
        regions: Region names ``chrom:start-end``, such as ATAC peak names.
        min_overlap_fraction: Required overlap fraction in ``[0, 1)``.

    Returns:
        A network with ``source`` (factor), ``target`` (region), and
        ``n_peaks`` (binding peaks overlapping the region).
    """
    if not 0 <= min_overlap_fraction < 1:
        raise ValueError("min_overlap_fraction must be in [0, 1)")
    region_names = np.asarray(regions, dtype=object)
    chroms, starts, ends = parse_coordinate_ids(region_names)
    frames = []
    for factor, source in peaks.items():
        table = source if isinstance(source, pd.DataFrame) else read_peak_file(source)
        if not {"chrom", "start", "end"} <= set(table.columns):
            raise ValueError(f"Peaks of {factor!r} need chrom, start, and end columns")
        peak_chroms = table["chrom"].astype(str).to_numpy(dtype=object)
        peak_starts = table["start"].to_numpy(dtype=np.int64)
        peak_ends = table["end"].to_numpy(dtype=np.int64)
        shared_chromosomes(chroms, peak_chroms, f"the peaks of {factor!r}")
        region_index, peak_index, shared = overlapping_intervals(
            (chroms, starts, ends), (peak_chroms, peak_starts, peak_ends)
        )
        if min_overlap_fraction > 0:
            keep = (
                shared / (ends[region_index] - starts[region_index])
                > min_overlap_fraction
            ) | (
                shared / (peak_ends[peak_index] - peak_starts[peak_index])
                > min_overlap_fraction
            )
            region_index = region_index[keep]
        if len(region_index) == 0:
            continue
        counts = np.bincount(region_index, minlength=len(region_names))
        hit = np.flatnonzero(counts)
        frames.append(
            pd.DataFrame(
                {
                    "source": str(factor),
                    "target": region_names[hit],
                    "n_peaks": counts[hit].astype(np.int64),
                }
            )
        )
    if not frames:
        return pd.DataFrame({"source": [], "target": [], "n_peaks": []})
    return pd.concat(frames, ignore_index=True)
