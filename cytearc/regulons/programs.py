"""Co-regulated gene programs: genes that rise and fall together as a unit.

Programs are found from expression alone, so they need no chromatin data:
genes are standardized across independent samples (metacells or pseudobulk
groups, each from its own cells), each gene is linked to the genes it
correlates with most, and the gene graph is partitioned with the Leiden
algorithm. Motif enrichment
of each program's genes then nominates the transcription factors that may
drive it, and overlap tests relate programs to regulons.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ..utils.arguments import integer_argument

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
        minMembership: Genes correlating less than this with the mean profile
            of their program's other genes are left unassigned.
        alpha: A screen against chance groupings: a gene's membership must
            also exceed the correlation that is significantly positive at this
            family-wise level over all genes (one-sided Fisher z, Bonferroni)
            for that many independent samples. It assumes independent samples
            and ignores that each program was chosen from the same
            correlations, so it is a filter, not a calibrated test of the
            programs.
        minSize: Programs with fewer genes are left unassigned.
        seed: Leiden random seed.
    """

    nNeighbors: int = 15
    minCorrelation: float = 0.2
    resolution: float = 1.0
    minMembership: float = 0.3
    alpha: float = 0.05
    minSize: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        integer_argument(self.nNeighbors, "nNeighbors", minimum=1)
        integer_argument(self.minSize, "minSize", minimum=2)
        integer_argument(self.seed, "seed", minimum=0)
        # Negative links would give the modularity graph negative weights.
        if not 0 <= self.minCorrelation < 1:
            raise ValueError("minCorrelation must lie in [0, 1)")
        if not -1 <= self.minMembership < 1:
            raise ValueError("minMembership must lie in [-1, 1)")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if not self.resolution > 0:
            raise ValueError("resolution must be positive")


def _standardize(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float64)
    centered = values - values.mean(axis=0)
    norms = np.sqrt((centered * centered).sum(axis=0))
    varying = norms > 0
    return centered[:, varying] / norms[varying], varying


def gene_programs(
    profiles: pd.DataFrame,
    *,
    parameters: GeneProgramParameters = GeneProgramParameters(),
    block_size: int = 2048,
) -> pd.DataFrame:
    """Partition genes into co-regulated programs.

    Args:
        profiles: Samples-by-genes expression named by gene, over independent
            samples such as the metacell or pseudobulk profiles of
            ``ds.regulons.profiles``. Smoothed or overlapping cells share their
            counts, which inflates every correlation and admits chance
            programs. Constant genes are skipped.
        parameters: Discovery settings.
        block_size: Genes whose correlations are computed at once.

    Returns:
        One row per assigned gene with ``source`` (the program, ``P1`` the
        largest), ``target`` (the gene), and ``membership``, the gene's
        Pearson correlation with the mean standardized profile of its
        program's other genes; rows sort by program, then decreasing
        membership.
    """
    from scipy.stats import norm

    from ..clustering import leiden_membership

    names = profiles.columns.to_numpy(dtype=object).astype(str)
    n_samples = len(profiles)
    if n_samples < 4:
        raise ValueError("Programs need at least four samples")
    if len(set(names)) != len(names):
        raise ValueError("Profile columns must name unique genes")
    unit, varying = _standardize(profiles.to_numpy(dtype=np.float64))
    names = names[varying]
    n_genes = len(names)
    if n_genes <= parameters.nNeighbors:
        raise ValueError("Programs need more varying genes than nNeighbors")
    # One-sided Fisher z critical correlation, Bonferroni-corrected over genes.
    z = norm.isf(parameters.alpha / n_genes)
    threshold = max(
        parameters.minMembership, float(np.tanh(z / np.sqrt(n_samples - 3)))
    )
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
        # Correlate each gene with the sum of the others: a mean that included
        # the gene would correlate with it by chance alone.
        total = unit[:, members].sum(axis=1)
        dots = unit[:, members].T @ total
        with np.errstate(divide="ignore", invalid="ignore"):
            membership = (dots - 1) / np.sqrt(total @ total - 2 * dots + 1)
        coherent = membership >= threshold
        if coherent.sum() >= parameters.minSize:
            candidates.append((members[coherent], membership[coherent]))
    candidates.sort(key=lambda item: (-len(item[0]), int(item[0][0])))
    frames = []
    for rank, (members, membership) in enumerate(candidates, start=1):
        frame = pd.DataFrame(
            {"source": f"P{rank}", "target": names[members], "membership": membership}
        ).sort_values("membership", ascending=False, kind="mergesort")
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"source": [], "target": [], "membership": []})
    return pd.concat(frames, ignore_index=True)


def program_scores(profiles: pd.DataFrame, programs: pd.DataFrame) -> pd.DataFrame:
    """Mean standardized profile of each program's genes in each sample.

    Args:
        profiles: Samples-by-genes values named by gene.
        programs: Output of ``gene_programs``.

    Returns:
        A samples-by-programs DataFrame indexed like ``profiles``.
    """
    names = profiles.columns.to_numpy(dtype=object).astype(str)
    values = profiles.to_numpy(dtype=np.float64)
    centered = values - values.mean(axis=0)
    spread = centered.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = centered / np.where(spread > 0, spread, np.nan)
    position = {name: index for index, name in enumerate(names)}
    result = {}
    for program, group in programs.groupby("source", sort=False):
        index = [position[gene] for gene in group["target"] if gene in position]
        result[program] = np.nanmean(z[:, index], axis=1) if index else np.nan
    return pd.DataFrame(result, index=profiles.index)


def program_regulators(
    programs: pd.DataFrame,
    database: RankingDatabase,
    annotations: pd.DataFrame,
    *,
    rank_threshold: int = 5000,
    auc_threshold: float = 0.05,
    nes_threshold: float = 3.0,
    categories: Sequence[str] | None = None,
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Nominate the transcription factors whose motifs explain each program.

    Each program's genes are tested with cisTarget against a gene-based
    ranking database, as pySCENIC tests modules, and enriched motifs are
    assigned to factors by the annotation table. A large program enriches
    many weak motifs, so pass ``profiles`` to check that each factor's own
    expression follows its program.

    Args:
        programs: Output of ``gene_programs`` (any ``source``/``target`` table).
        database: Gene-based ranking database.
        annotations: Output of ``read_motif_annotations``.
        rank_threshold, auc_threshold, nes_threshold: cisTarget settings.
        categories: Annotation categories to accept; all when omitted.
        profiles: Samples-by-genes expression named by gene, such as metacell
            profiles.

    Returns:
        One row per program, enriched motif, and annotated factor: ``program``,
        ``TF``, ``motif``, ``NES``, ``AUC``, ``n_targets`` (leading-edge
        genes), and ``category``, sorted by program and decreasing NES. With
        ``profiles``, ``correlation`` is the Pearson correlation of the
        factor's expression with the program's ``program_scores`` across
        samples, NaN for a factor without a varying profile.
    """
    table = annotations
    if categories is not None:
        table = table.loc[table["category"].isin(list(categories))]
    by_motif = table.groupby("motif")
    rows = []
    for program, group in programs.groupby("source", sort=False):
        genes, ranks = database.ranks(group["target"])
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
    frame = pd.DataFrame.from_records(rows, columns=columns)
    if profiles is not None:
        scores = program_scores(profiles, programs)
        with np.errstate(invalid="ignore", divide="ignore"):
            frame["correlation"] = [
                profiles[tf].corr(scores[program]) if tf in profiles.columns else np.nan
                for program, tf in zip(frame["program"], frame["TF"], strict=True)
            ]
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
        queries: ``source`` (set) and ``target`` (gene) columns, such as
            programs.
        references: ``source`` and ``target`` columns, such as regulons.
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
        str(name): set(group["target"].astype(str)) & genes
        for name, group in queries.groupby("source")
    }
    reference_sets = {
        str(name): set(group["target"].astype(str)) & genes
        for name, group in references.groupby("source")
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
            # As scipy's fisher_exact: undefined when a margin is empty.
            if 0 in (a + b, c + d, a + c, b + d):
                odds = np.nan
            else:
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
