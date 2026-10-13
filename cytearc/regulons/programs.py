"""Co-regulated gene programs: genes that rise and fall together as a unit.

Programs are found from expression alone, so they need no chromatin data:
genes are standardized across samples (metacells, pseudobulk groups, or
smoothed cells), each gene is linked to the genes it correlates with most,
and the gene graph is partitioned with the Leiden algorithm. Motif enrichment
of each program's genes then nominates the transcription factors that may
drive it, and overlap tests relate programs to regulons.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from .cistarget import enrich_signature
from .databases import RankingDatabase

__all__ = [
    "GeneProgramParameters",
    "gene_programs",
    "program_regulators",
    "program_scores",
    "set_overlaps",
]


@dataclass(frozen=True, slots=True)
class GeneProgramParameters:
    """Settings of gene-program discovery.

    Attributes:
        nNeighbors: Most correlated genes linked to each gene.
        minCorrelation: Links need at least this Pearson correlation.
        resolution: Leiden resolution of the gene graph.
        minMembership: Genes correlating less than this with their program's
            mean profile are left unassigned, so chance groupings of
            uncorrelated genes do not form programs.
        minSize: Programs with fewer genes are left unassigned.
        seed: Leiden random seed.
    """

    nNeighbors: int = 15
    minCorrelation: float = 0.2
    resolution: float = 1.0
    minMembership: float = 0.3
    minSize: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        for label, value, low in (
            ("nNeighbors", self.nNeighbors, 1),
            ("minSize", self.minSize, 2),
            ("seed", self.seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{label} must be an integer of at least {low}")
        if not -1 <= self.minCorrelation < 1:
            raise ValueError("minCorrelation must lie in [-1, 1)")
        if not -1 <= self.minMembership < 1:
            raise ValueError("minMembership must lie in [-1, 1)")
        if not self.resolution > 0:
            raise ValueError("resolution must be positive")


def _standardize(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float64)
    centered = values - values.mean(axis=0)
    norms = np.sqrt((centered * centered).sum(axis=0))
    varying = norms > 0
    return centered[:, varying] / norms[varying], varying


def gene_programs(
    profiles: np.ndarray,
    gene_names: Sequence[str] | np.ndarray,
    *,
    parameters: GeneProgramParameters = GeneProgramParameters(),
    block_size: int = 2048,
) -> pd.DataFrame:
    """Partition genes into co-regulated programs.

    Args:
        profiles: Samples-by-genes expression, such as log-normalized
            metacell or pseudobulk means. Constant genes are skipped.
        gene_names: Names of the profile columns.
        parameters: Discovery settings.
        block_size: Genes whose correlations are computed at once.

    Returns:
        One row per assigned gene with ``program`` (``P1`` is the largest),
        ``gene``, and ``membership``, the gene's Pearson correlation with its
        program's mean standardized profile; rows sort by program, then
        decreasing membership.
    """
    from ..clustering import leiden_membership

    names = np.asarray(gene_names, dtype=object).astype(str)
    if np.asarray(profiles).ndim != 2 or np.asarray(profiles).shape[1] != len(names):
        raise ValueError("gene_names must align with the profile columns")
    if len(set(names)) != len(names):
        raise ValueError("gene_names must be unique")
    unit, varying = _standardize(profiles)
    names = names[varying]
    n_genes = len(names)
    if n_genes <= parameters.nNeighbors:
        raise ValueError("Programs need more varying genes than nNeighbors")
    k = parameters.nNeighbors
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        block = unit[:, start:stop].T @ unit
        block[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        nearest = np.argpartition(-block, k, axis=1)[:, :k]
        values = np.take_along_axis(block, nearest, axis=1)
        keep = values >= parameters.minCorrelation
        rows.append(np.repeat(np.arange(start, stop), k)[keep.reshape(-1)])
        columns.append(nearest.reshape(-1)[keep.reshape(-1)])
        weights.append(values.reshape(-1)[keep.reshape(-1)])
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n_genes, n_genes),
    ).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    labels = leiden_membership(graph, parameters.resolution, parameters.seed)
    candidates = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < parameters.minSize:
            continue
        score = unit[:, members].mean(axis=1)
        centered = score - score.mean()
        norm = np.sqrt((centered * centered).sum())
        if not norm > 0:
            continue
        membership = unit[:, members].T @ (centered / norm)
        coherent = membership >= parameters.minMembership
        if coherent.sum() >= parameters.minSize:
            candidates.append((members[coherent], membership[coherent]))
    candidates.sort(key=lambda item: (-len(item[0]), int(item[0][0])))
    frames = []
    for rank, (members, membership) in enumerate(candidates, start=1):
        frame = pd.DataFrame(
            {"program": f"P{rank}", "gene": names[members], "membership": membership}
        ).sort_values("membership", ascending=False, kind="mergesort")
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"program": [], "gene": [], "membership": []})
    return pd.concat(frames, ignore_index=True)


def program_scores(
    profiles: np.ndarray,
    gene_names: Sequence[str] | np.ndarray,
    programs: pd.DataFrame,
) -> pd.DataFrame:
    """Mean standardized profile of each program's genes in each sample.

    Args:
        profiles: Samples-by-genes values.
        gene_names: Names of the profile columns.
        programs: Output of ``gene_programs``.

    Returns:
        A samples-by-programs DataFrame.
    """
    names = np.asarray(gene_names, dtype=object).astype(str)
    values = np.asarray(profiles, dtype=np.float64)
    centered = values - values.mean(axis=0)
    spread = centered.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = centered / np.where(spread > 0, spread, np.nan)
    position = {name: index for index, name in enumerate(names)}
    result = {}
    for program, group in programs.groupby("program", sort=False):
        index = [position[gene] for gene in group["gene"] if gene in position]
        result[program] = np.nanmean(z[:, index], axis=1) if index else np.nan
    return pd.DataFrame(result)


def program_regulators(
    programs: pd.DataFrame,
    database: RankingDatabase,
    annotations: pd.DataFrame,
    *,
    rank_threshold: int = 5000,
    auc_threshold: float = 0.05,
    nes_threshold: float = 3.0,
    categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Nominate the transcription factors whose motifs explain each program.

    Each program's genes are tested with cisTarget against a gene-based
    ranking database, as pySCENIC tests modules, and enriched motifs are
    assigned to factors by the annotation table.

    Args:
        programs: Output of ``gene_programs`` (any ``program``/``gene`` table).
        database: Gene-based ranking database.
        annotations: Output of ``read_motif_annotations``.
        rank_threshold, auc_threshold, nes_threshold: cisTarget settings.
        categories: Annotation categories to accept; all when omitted.

    Returns:
        One row per program, enriched motif, and annotated factor: ``program``,
        ``TF``, ``motif``, ``NES``, ``AUC``, ``n_targets`` (leading-edge
        genes), and ``category``, sorted by program and decreasing NES.
    """
    table = annotations
    if categories is not None:
        table = table.loc[table["category"].isin(list(categories))]
    by_motif = table.groupby("motif")
    rows = []
    for program, group in programs.groupby("program", sort=False):
        genes, ranks = database.ranks(group["gene"])
        if len(genes) < 2:
            continue
        result = enrich_signature(
            ranks,
            database.rowNames,
            genes,
            database.total,
            rank_threshold=min(rank_threshold, database.total - 1),
            auc_threshold=auc_threshold,
            nes_threshold=nes_threshold,
        )
        for motif, auc, nes, _rank, n_targets in result.table.itertuples(index=False):
            if motif not in by_motif.groups:
                continue
            for tf, category in (
                by_motif.get_group(motif)[["TF", "category"]]
                .drop_duplicates()
                .itertuples(index=False)
            ):
                rows.append((program, tf, motif, nes, auc, n_targets, category))
    columns = ["program", "TF", "motif", "NES", "AUC", "n_targets", "category"]
    if not rows:
        return pd.DataFrame({column: [] for column in columns})
    frame = pd.DataFrame.from_records(rows, columns=columns)
    return frame.sort_values(
        ["program", "NES", "TF"], ascending=[True, False, True], kind="mergesort"
    ).reset_index(drop=True)


def set_overlaps(
    queries: pd.DataFrame,
    references: pd.DataFrame,
    universe: Sequence[str] | np.ndarray,
) -> pd.DataFrame:
    """Test every query set for overlap with every reference set.

    Args:
        queries: ``set`` and ``gene`` columns, such as programs renamed.
        references: ``set`` and ``gene`` columns, such as regulons.
        universe: Genes that could have been in any set.

    Returns:
        ``query``, ``reference``, ``overlap``, ``expected``, ``odds_ratio``,
        one-sided hypergeometric ``p_value``, and Benjamini-Hochberg ``q_value``
        across all tests.
    """
    from scipy.stats import hypergeom

    from ..features.statistical import adjust_pvalues

    genes = set(np.asarray(universe, dtype=object).astype(str))
    total = len(genes)
    query_sets = {
        str(name): set(group["gene"].astype(str)) & genes
        for name, group in queries.groupby("set")
    }
    reference_sets = {
        str(name): set(group["gene"].astype(str)) & genes
        for name, group in references.groupby("set")
    }
    rows = []
    for query, members in query_sets.items():
        for reference, targets in reference_sets.items():
            overlap = len(members & targets)
            expected = len(members) * len(targets) / total if total else np.nan
            p_value = float(
                hypergeom.sf(overlap - 1, total, len(targets), len(members))
            )
            a = overlap
            b = len(members) - overlap
            c = len(targets) - overlap
            d = total - len(members) - len(targets) + overlap
            odds = (a * d) / (b * c) if b * c > 0 else np.inf
            rows.append((query, reference, overlap, expected, odds, p_value))
    frame = pd.DataFrame.from_records(
        rows,
        columns=["query", "reference", "overlap", "expected", "odds_ratio", "p_value"],
    )
    frame["q_value"] = (
        adjust_pvalues(frame["p_value"].to_numpy(), method="fdr_bh")
        if len(frame)
        else []
    )
    return frame
