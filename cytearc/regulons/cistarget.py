"""cisTarget motif enrichment: recovery curves, AUC, NES, and leading edges.

The numerics follow ctxcore 0.2.0 (``ctxcore.recovery``), which pySCENIC and
pycistarget both use: ranks are zero-based, the AUC integrates the weighted
recovery curve below ``int(round(auc_threshold * total)) - 1``, the NES
standardizes AUCs over every motif of a database with the population standard
deviation, and a motif's leading edge holds the signature features ranked at
or before the point where its recovery curve rises furthest above the mean
plus two standard deviations of all motifs' curves.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from ..utils.arguments import integer_argument

__all__ = [
    "MotifEnrichment",
    "auc_rank_cutoff",
    "enrich_signature",
    "recovery_aucs",
    "recovery_curves",
]


def auc_rank_cutoff(auc_threshold: float, total: int) -> int:
    """Return ctxcore's AUC rank cutoff for a ranking of ``total`` features.

    Ranks strictly below the cutoff contribute to the AUC. Python's ``round``
    rounds halves to even, as ctxcore's does.
    """
    if not 0 < auc_threshold <= 1:
        raise ValueError("auc_threshold must be in (0, 1]")
    integer_argument(total, "total", minimum=2)
    cutoff = int(round(auc_threshold * total)) - 1
    if cutoff < 1:
        raise ValueError(
            f"auc_threshold {auc_threshold} leaves no ranks among {total} features"
        )
    return cutoff


@njit(cache=True, nogil=True)
def _aucs(rankings: np.ndarray, weights: np.ndarray, cutoff: int) -> np.ndarray:
    n_motifs, n_features = rankings.shape
    areas = np.zeros(n_motifs, dtype=np.float64)
    for motif in range(n_motifs):
        area = 0.0
        for feature in range(n_features):
            rank = rankings[motif, feature]
            if rank < cutoff:
                area += weights[feature] * (cutoff - rank)
        areas[motif] = area
    return areas


def recovery_aucs(
    rankings: np.ndarray,
    total: int,
    weights: np.ndarray,
    auc_threshold: float,
) -> np.ndarray:
    """Return each motif's AUC of the weighted recovery curve of a signature.

    Args:
        rankings: ``(motifs, signature features)`` zero-based ranks of the
            signature's features in each motif's ranking of ``total`` features.
        total: Number of ranked features in the database.
        weights: Signature feature weights.
        auc_threshold: Fraction of the ranking that the AUC covers.

    Returns:
        ``area / ((cutoff + 1) * sum(weights))`` per motif, where the area
        sums ``weight * (cutoff - rank)`` over features ranked below the cutoff,
        exactly ctxcore's ``weighted_auc1d``.
    """
    ranks = np.asarray(rankings)
    weight_values = np.asarray(weights, dtype=np.float64)
    if ranks.ndim != 2 or not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError("rankings must be a two-dimensional integer matrix")
    if weight_values.shape != (ranks.shape[1],):
        raise ValueError("weights must align with the ranking columns")
    total_weight = float(weight_values.sum())
    if not total_weight > 0:
        raise ValueError("Signature weights must sum to a positive value")
    if np.any(ranks < 0):
        raise ValueError("Ranks must be non-negative")
    cutoff = auc_rank_cutoff(auc_threshold, total)
    areas = _aucs(np.ascontiguousarray(ranks), weight_values, cutoff)
    return areas / ((cutoff + 1) * total_weight)


@njit(cache=True, nogil=True)
def _curves(rankings: np.ndarray, weights: np.ndarray, length: int) -> np.ndarray:
    n_motifs, n_features = rankings.shape
    curves = np.zeros((n_motifs, length), dtype=np.float64)
    for motif in range(n_motifs):
        for feature in range(n_features):
            rank = rankings[motif, feature]
            if rank < length:
                curves[motif, rank] += weights[feature]
        running = 0.0
        for position in range(length):
            running += curves[motif, position]
            curves[motif, position] = running
    return curves


def recovery_curves(
    rankings: np.ndarray,
    weights: np.ndarray,
    rank_threshold: int,
) -> np.ndarray:
    """Return the weighted recovery curve of each motif up to ``rank_threshold``.

    ``curve[motif, r]`` is the total weight of signature features ranked at or
    before ``r``, for ``r`` below ``rank_threshold``, ctxcore's ``rcc2d``.
    """
    ranks = np.asarray(rankings)
    weight_values = np.asarray(weights, dtype=np.float64)
    if ranks.ndim != 2 or not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError("rankings must be a two-dimensional integer matrix")
    if weight_values.shape != (ranks.shape[1],):
        raise ValueError("weights must align with the ranking columns")
    integer_argument(rank_threshold, "rank_threshold", minimum=1)
    if np.any(ranks < 0):
        raise ValueError("Ranks must be non-negative")
    return _curves(np.ascontiguousarray(ranks), weight_values, rank_threshold)


@njit(cache=True, nogil=True)
def _envelope(ranks: np.ndarray, weights: np.ndarray, length: int) -> np.ndarray:
    """Mean plus two population SDs of all motifs' recovery curves.

    Sums run over motifs in row order, then the deviations are summed again,
    the order of numpy's ``mean(axis=0)`` and ``std(axis=0)`` on the stacked
    curves, without holding every curve at once.
    """
    n_motifs, n_genes = ranks.shape
    curve = np.zeros(length, dtype=np.float64)
    total = np.zeros(length, dtype=np.float64)
    for motif in range(n_motifs):
        curve[:] = 0.0
        for gene in range(n_genes):
            rank = ranks[motif, gene]
            if rank < length:
                curve[rank] += weights[gene]
        running = 0.0
        for position in range(length):
            running += curve[position]
            total[position] += running
    mean = total / n_motifs
    squares = np.zeros(length, dtype=np.float64)
    for motif in range(n_motifs):
        curve[:] = 0.0
        for gene in range(n_genes):
            rank = ranks[motif, gene]
            if rank < length:
                curve[rank] += weights[gene]
        running = 0.0
        for position in range(length):
            running += curve[position]
            deviation = running - mean[position]
            squares[position] += deviation * deviation
    envelope: np.ndarray = mean + 2.0 * np.sqrt(squares / n_motifs)
    return envelope


@njit(cache=True, nogil=True)
def _rank_at_max(ranks: np.ndarray, weights: np.ndarray, envelope: np.ndarray) -> int:
    """First rank where one motif's recovery curve exceeds the envelope most."""
    length = envelope.size
    curve = np.zeros(length, dtype=np.float64)
    for gene in range(ranks.size):
        rank = ranks[gene]
        if rank < length:
            curve[rank] += weights[gene]
    running = 0.0
    best = 0
    best_value = -np.inf
    for position in range(length):
        running += curve[position]
        value = running - envelope[position]
        if value > best_value:
            best_value = value
            best = position
    return best


@dataclass(frozen=True, slots=True)
class MotifEnrichment:
    """Enriched motifs of one signature in one ranking database.

    Attributes:
        table: One row per enriched motif, sorted by decreasing NES, with
            ``motif``, ``AUC``, ``NES``, ``rank_at_max``, and ``n_targets``.
        leading_edges: For each motif in ``table``, the signature features at
            or before its rank at maximum, with their weights and ranks,
            sorted by rank.
        n_signature: Signature features found in the database.
    """

    table: pd.DataFrame
    leading_edges: dict[str, pd.DataFrame]
    n_signature: int


def enrich_signature(
    rankings: np.ndarray,
    motif_names: np.ndarray,
    feature_names: np.ndarray,
    total: int,
    *,
    weights: np.ndarray | None = None,
    rank_threshold: int,
    auc_threshold: float,
    nes_threshold: float,
) -> MotifEnrichment:
    """Find motifs whose rankings recover a signature better than the others.

    Args:
        rankings: ``(motifs, signature features)`` zero-based ranks, every
            motif of the database in rows.
        motif_names: Names of the rows.
        feature_names: Names of the signature features in the columns.
        total: Number of ranked features in the database.
        weights: Feature weights; all ones when omitted.
        rank_threshold: Length of the recovery curves used for leading edges.
        auc_threshold: Fraction of the ranking covered by the AUC.
        nes_threshold: Minimum NES of an enriched motif.

    Returns:
        The enriched motifs and their leading edges.
    """
    ranks = np.asarray(rankings)
    motifs = np.asarray(motif_names).astype(str)
    features = np.asarray(feature_names).astype(str)
    if ranks.ndim != 2 or ranks.shape != (len(motifs), len(features)):
        raise ValueError("rankings must be motifs by signature features")
    if len(motifs) < 2:
        raise ValueError("NES needs at least two motifs")
    weight_values = (
        np.ones(len(features), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    aucs = recovery_aucs(ranks, total, weight_values, auc_threshold)
    spread = aucs.std()
    if not spread > 0:
        nes = np.zeros_like(aucs)
    else:
        nes = (aucs - aucs.mean()) / spread
    enriched = np.flatnonzero(nes >= nes_threshold)
    columns = ["motif", "AUC", "NES", "rank_at_max", "n_targets"]
    if len(enriched) == 0:
        return MotifEnrichment(
            table=pd.DataFrame(columns=columns),
            leading_edges={},
            n_signature=len(features),
        )
    integer_argument(rank_threshold, "rank_threshold", minimum=1)
    # The envelope streams the curves: all of them at once would hold
    # motifs by rank_threshold values.
    contiguous = np.ascontiguousarray(ranks)
    envelope = _envelope(contiguous, weight_values, rank_threshold)
    rows = []
    edges: dict[str, pd.DataFrame] = {}
    for motif in enriched:
        rank_at_max = int(_rank_at_max(contiguous[motif], weight_values, envelope))
        motif_ranks = ranks[motif]
        keep = motif_ranks <= rank_at_max
        order = np.argsort(motif_ranks[keep], kind="stable")
        edge = pd.DataFrame(
            {
                "feature": features[keep][order],
                "weight": weight_values[keep][order],
                "rank": motif_ranks[keep][order].astype(np.int64),
            }
        )
        edges[motifs[motif]] = edge
        rows.append((motifs[motif], aucs[motif], nes[motif], rank_at_max, len(edge)))
    table = pd.DataFrame(rows, columns=columns).sort_values(
        ["NES", "motif"], ascending=[False, True], kind="mergesort"
    )
    return MotifEnrichment(
        table=table.reset_index(drop=True),
        leading_edges={name: edges[name] for name in table["motif"]},
        n_signature=len(features),
    )
