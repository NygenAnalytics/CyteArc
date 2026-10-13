"""Enhancer-driven regulons (eRegulons) built the SCENIC+ way.

An eRegulon links a transcription factor to the regions of its cistrome and,
through region-to-gene links, to target genes. Region-to-gene links are split
by the sign of their correlation and thresholded several ways per gene; each
factor's cistrome selects the links of its regions; and a preranked GSEA of
the selected genes against the factor's co-expression targets (GRNBoost2
importances) keeps the leading edge. Surviving modules are merged per factor
and sign class, as SCENIC+ 1.0 ``build_grn`` with its pipeline settings does.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "EREGULON_COLUMNS",
    "EregulonParameters",
    "GseaResult",
    "SCENICPLUS_EREGULON_PARAMETERS",
    "binarize_region_to_gene",
    "build_eregulons",
    "gsea_prerank",
]

EREGULON_COLUMNS = (
    "eRegulon",
    "TF",
    "tf_to_gene_sign",
    "region_to_gene_sign",
    "region",
    "gene",
    "importance_R2G",
    "rho_R2G",
    "importance_TF2G",
    "rho_TF2G",
)


@dataclass(frozen=True, slots=True)
class EregulonParameters:
    """Settings of eRegulon construction.

    The defaults are those that the SCENIC+ 1.0 pipeline passes to
    ``build_grn`` (without BASC binarization, which CyteArc does not run).

    Attributes:
        rhoThreshold: Region-to-gene and factor-to-gene links with a
            correlation above this are positive, below its negative negative.
        quantiles: Per-gene importance quantiles that threshold region links.
        topRegionsPerGene: Each gene's most important region links.
        minTargetGenes: Smallest eRegulon, and smallest co-expression ranking.
        nPermutations: GSEA gene-set permutations.
        seed: GSEA random seed, the same for every module.
    """

    rhoThreshold: float = 0.05
    quantiles: tuple[float, ...] = (0.85, 0.90, 0.95)
    topRegionsPerGene: tuple[int, ...] = (5, 10, 15)
    minTargetGenes: int = 10
    nPermutations: int = 1000
    seed: int = 666

    def __post_init__(self) -> None:
        if not self.rhoThreshold >= 0:
            raise ValueError("rhoThreshold must be non-negative")
        if any(not 0 <= value <= 1 for value in self.quantiles):
            raise ValueError("quantiles must lie in [0, 1]")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.topRegionsPerGene
        ):
            raise ValueError("topRegionsPerGene must hold positive integers")
        for label, value in (
            ("minTargetGenes", self.minTargetGenes),
            ("nPermutations", self.nPermutations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")


#: SCENIC+ 1.0 pipeline settings.
SCENICPLUS_EREGULON_PARAMETERS = EregulonParameters()


@dataclass(frozen=True, slots=True)
class GseaResult:
    """One preranked GSEA result.

    Attributes:
        enrichmentScore: Running-sum maximum deviation from zero.
        nes: Enrichment score over the mean null score of the same sign.
        pValue: Fraction of same-signed null scores at least as extreme.
        leadingEdge: Gene-set members ranked at or before the running sum's
            extreme (after it, for a negative score).
    """

    enrichmentScore: float
    nes: float
    pValue: float
    leadingEdge: tuple[str, ...]


def gsea_prerank(
    ranked: pd.Series,
    gene_set: Sequence[str],
    *,
    n_permutations: int = 1000,
    seed: int = 666,
) -> GseaResult:
    """Run a weighted preranked GSEA with gene-set permutations, as gseapy 0.10.8 does.

    The running sum steps up by each member's absolute weight over the
    members' total and down by one over the non-members' count. Null scores
    come from shuffling member positions; the shuffles use numpy's legacy
    random state seeded the way gseapy's ``gsea_compute`` seeds one gene set,
    so results match gseapy for the same seed.

    Args:
        ranked: Weights indexed by gene, already in ranking order.
        gene_set: Genes to test; genes absent from ``ranked`` are ignored.
        n_permutations: Number of null permutations.
        seed: Random seed.

    Returns:
        The enrichment score, NES, nominal p-value, and leading edge.
    """
    genes = np.asarray(ranked.index, dtype=object)
    weights = np.abs(np.asarray(ranked.to_numpy(), dtype=np.float64))
    n_genes = len(genes)
    members = set(str(gene) for gene in gene_set)
    tags = np.fromiter(
        (str(gene) in members for gene in genes), dtype=np.int64, count=n_genes
    )
    hits = np.flatnonzero(tags)
    if len(hits) == 0 or len(hits) == n_genes or not weights[hits].sum() > 0:
        # gseapy divides by zero here; such a module has no score.
        return GseaResult(np.nan, np.nan, np.nan, ())
    # gseapy seeds numpy's global generator with ``seed`` and draws one
    # stream seed per gene set; a local generator gives the same draw.
    stream_seed = np.random.RandomState(seed).randint(np.iinfo(np.int32).max, size=1)[0]
    random_state = np.random.RandomState(stream_seed)
    tag_matrix = np.tile(tags, (n_permutations + 1, 1))
    for row in range(n_permutations):
        random_state.shuffle(tag_matrix[row])
    weight_matrix = np.tile(weights, (n_permutations + 1, 1))
    n_hits = tag_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_tag = 1.0 / np.sum(weight_matrix * tag_matrix, axis=1, keepdims=True)
        norm_miss = 1.0 / (n_genes - n_hits)
        running = np.cumsum(
            tag_matrix * weight_matrix * norm_tag - (1 - tag_matrix) * norm_miss,
            axis=1,
        )
    highest = running.max(axis=1)
    lowest = running.min(axis=1)
    scores = np.where(np.abs(highest) > np.abs(lowest), highest, lowest)
    observed = float(scores[-1])
    null = scores[:-1]
    observed_running = running[-1]
    same_sign = null[null >= 0] if observed >= 0 else null[null < 0]
    if len(same_sign) == 0:
        nes = p_value = np.nan
    elif observed >= 0:
        nes = observed / same_sign.mean()
        p_value = (null >= observed).sum() / len(same_sign)
    else:
        nes = -observed / same_sign.mean()
        p_value = (null < observed).sum() / len(same_sign)
    if observed > 0:
        cut = int(np.argmax(observed_running))
        edge = hits[hits <= cut]
    elif observed < 0:
        cut = int(np.argmin(observed_running))
        edge = hits[hits >= cut]
    else:
        edge = hits
    return GseaResult(
        enrichmentScore=observed,
        nes=float(nes),
        pValue=float(p_value),
        leadingEdge=tuple(str(gene) for gene in genes[edge]),
    )


def _bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, capped at 1 (SCENIC+ ``p_adjust_bh``)."""
    p = np.asarray(p_values, dtype=np.float64)
    if len(p) == 0:
        return p
    by_descend = p.argsort()[::-1]
    by_original = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    adjusted = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return adjusted[by_original]


def binarize_region_to_gene(
    links: pd.DataFrame,
    *,
    parameters: EregulonParameters = SCENICPLUS_EREGULON_PARAMETERS,
) -> list[tuple[str, str, pd.DataFrame]]:
    """Threshold region-to-gene links per gene, separately for each sign.

    Positive links (``rho`` above the threshold) and negative links (below
    its negative) are each thresholded per gene: by every importance
    quantile (strictly above) and by every top-n (at least the n-th largest
    importance, or all links of a gene with fewer). Weaker-correlated links
    are discarded.

    Args:
        links: ``region``, ``gene``, ``importance``, and ``rho`` columns.
        parameters: Thresholds; SCENIC+ pipeline settings when omitted.

    Returns:
        ``(sign, context, links)`` triples, sign ``"+"`` or ``"-"``.
    """
    required = {"region", "gene", "importance", "rho"}
    if not required <= set(links.columns):
        raise ValueError(f"links lacks columns {sorted(required - set(links.columns))}")
    result: list[tuple[str, str, pd.DataFrame]] = []
    rho = links["rho"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        subsets = (
            ("-", links.loc[rho < -parameters.rhoThreshold]),
            ("+", links.loc[rho > parameters.rhoThreshold]),
        )
    for sign, subset in subsets:
        if subset.empty:
            continue
        grouped = subset.groupby("gene", sort=False)["importance"]
        for quantile in parameters.quantiles:
            thresholds = grouped.transform(lambda x, q=quantile: np.quantile(x, q))
            passing = subset.loc[subset["importance"] > thresholds]
            result.append((sign, f"{quantile} quantile", passing))
        for n in parameters.topRegionsPerGene:
            thresholds = grouped.transform(
                lambda x, n=n: (
                    np.sort(x.to_numpy())[-n:].min() if len(x) >= n else x.min()
                )
            )
            passing = subset.loc[subset["importance"] >= thresholds]
            result.append((sign, f"Top {n} region-to-gene links per gene", passing))
    return result


def _rankings(
    tf_to_gene: pd.DataFrame, factors: set[str], parameters: EregulonParameters
) -> dict[tuple[str, str], pd.Series]:
    """Positive and negative co-expression rankings of each factor.

    Each factor gets a self-link with its top importance plus 1e-5 and
    correlation 1, as SCENIC+ adds before ranking.
    """
    links = tf_to_gene.loc[tf_to_gene["TF"].isin(factors)]
    tops = links.groupby("TF")["importance"].max()
    self_links = pd.DataFrame(
        {
            "TF": tops.index,
            "target": tops.index,
            "importance": tops.to_numpy() + 0.00001,
            "correlation": 1.0,
        }
    )
    links = pd.concat([links, self_links], ignore_index=True)
    rankings: dict[tuple[str, str], pd.Series] = {}
    for sign, mask in (
        ("+", links["correlation"] > parameters.rhoThreshold),
        ("-", links["correlation"] < -parameters.rhoThreshold),
    ):
        for tf, group in links.loc[mask].groupby("TF"):
            if len(group) >= parameters.minTargetGenes:
                ranking = group.set_index("target")["importance"].sort_values(
                    ascending=False, kind="mergesort"
                )
                rankings[(str(tf), sign)] = ranking
    return rankings


def build_eregulons(
    cistromes: pd.DataFrame,
    region_to_gene: pd.DataFrame,
    tf_to_gene: pd.DataFrame,
    *,
    label: str = "direct",
    parameters: EregulonParameters = SCENICPLUS_EREGULON_PARAMETERS,
    nthreads: int = 1,
) -> pd.DataFrame:
    """Build eRegulons from cistromes, region-to-gene links, and a co-expression network.

    Args:
        cistromes: ``source`` (factor) and ``target`` (region) columns, from
            ``cistromes_from_motifs`` or ``binding_cistromes``.
        region_to_gene: ``region``, ``gene``, ``importance``, and ``rho``
            columns, from ``region_gene_links``.
        tf_to_gene: ``TF``, ``target``, ``importance``, and ``correlation``
            columns, such as ``GrnResult.adjacencies``.
        label: Cistrome label in eRegulon names, such as ``direct``,
            ``extended``, or the experiment that measured binding.
        parameters: Construction settings; SCENIC+ pipeline settings when omitted.
        nthreads: Modules tested concurrently.

    Returns:
        One row per eRegulon region-to-gene link, with the factor-to-gene
        statistics of the target. Names read
        ``{TF}_{label}_{TF-to-gene sign}/{region-to-gene sign}``.
    """
    for frame, columns, name in (
        (cistromes, {"source", "target"}, "cistromes"),
        (tf_to_gene, {"TF", "target", "importance", "correlation"}, "tf_to_gene"),
    ):
        if not columns <= set(frame.columns):
            raise ValueError(
                f"{name} lacks columns {sorted(columns - set(frame.columns))}"
            )
    if isinstance(nthreads, bool) or not isinstance(nthreads, int) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer")
    factors = set(cistromes["source"].astype(str)) & set(tf_to_gene["TF"].astype(str))
    regions_of = {
        str(tf): set(group["target"].astype(str))
        for tf, group in cistromes.loc[cistromes["source"].isin(factors)].groupby(
            "source"
        )
    }
    rankings = _rankings(tf_to_gene, factors, parameters)
    modules: list[tuple[str, str, str, pd.DataFrame]] = []
    for r2g_sign, context, links in binarize_region_to_gene(
        region_to_gene, parameters=parameters
    ):
        if links.empty:
            continue
        for tf in sorted(factors):
            selected = links.loc[links["region"].astype(str).isin(regions_of[tf])]
            if not selected.empty:
                modules.append((tf, r2g_sign, context, selected))
    tasks = [
        (tf, r2g_sign, tf2g_sign, selected)
        for tf, r2g_sign, _context, selected in modules
        for tf2g_sign in ("+", "-")
        if (tf, tf2g_sign) in rankings
    ]

    def test(task: tuple[str, str, str, pd.DataFrame]) -> GseaResult | None:
        tf, _r2g, tf2g_sign, selected = task
        try:
            return gsea_prerank(
                rankings[(tf, tf2g_sign)],
                sorted(set(selected["gene"].astype(str))),
                n_permutations=parameters.nPermutations,
                seed=parameters.seed,
            )
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return None

    if nthreads == 1:
        results = [test(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            results = list(executor.map(test, tasks))
    tested = [
        (task, result)
        for task, result in zip(tasks, results, strict=True)
        if result is not None
        and np.isfinite(result.enrichmentScore)
        and np.isfinite(result.pValue)
    ]
    adjusted = _bh(np.asarray([result.pValue for _task, result in tested]))
    kept: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    link_values: dict[
        tuple[str, str, str], dict[tuple[str, str], tuple[float, float]]
    ] = {}
    for ((tf, r2g_sign, tf2g_sign, selected), result), adjusted_p in zip(
        tested, adjusted, strict=True
    ):
        if not (adjusted_p < 1 and result.nes > 0):
            continue
        edge = set(result.leadingEdge)
        in_edge = selected.loc[selected["gene"].astype(str).isin(edge)]
        if in_edge["gene"].nunique() < parameters.minTargetGenes:
            continue
        key = (tf, tf2g_sign, r2g_sign)
        pairs = kept.setdefault(key, set())
        values = link_values.setdefault(key, {})
        for region, gene, importance, rho in in_edge[
            ["region", "gene", "importance", "rho"]
        ].itertuples(index=False):
            pairs.add((str(region), str(gene)))
            values[(str(region), str(gene))] = (float(importance), float(rho))
    co_expression = tf_to_gene.set_index(["TF", "target"])[
        ["importance", "correlation"]
    ]
    rows = []
    for (tf, tf2g_sign, r2g_sign), pairs in sorted(kept.items()):
        name = f"{tf}_{label}_{tf2g_sign}/{r2g_sign}"
        for region, gene in sorted(pairs):
            importance, rho = link_values[(tf, tf2g_sign, r2g_sign)][(region, gene)]
            if (tf, gene) in co_expression.index:
                tf_importance, tf_rho = co_expression.loc[(tf, gene)].to_numpy()
            elif gene == tf:
                tf_importance, tf_rho = np.nan, 1.0
            else:
                tf_importance, tf_rho = np.nan, np.nan
            rows.append(
                (
                    name,
                    tf,
                    tf2g_sign,
                    r2g_sign,
                    region,
                    gene,
                    importance,
                    rho,
                    float(tf_importance),
                    float(tf_rho),
                )
            )
    return pd.DataFrame.from_records(rows, columns=list(EREGULON_COLUMNS))
