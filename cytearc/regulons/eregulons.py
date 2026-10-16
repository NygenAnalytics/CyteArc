"""Enhancer-driven regulons (eRegulons) built the SCENIC+ way.

An eRegulon links a transcription factor to the regions of its cistrome and,
through region-to-gene links, to target genes. Region-to-gene links are split
by the sign of their correlation and thresholded several ways per gene; each
factor's cistrome selects the links of its regions; and a preranked GSEA of
the selected genes against the factor's co-expression targets (GRNBoost2
importances) keeps the leading edge. Surviving modules form an eRegulon per
factor and sign class, as SCENIC+ 1.0 ``build_grn`` with its pipeline settings
does; each module's statistics stay with its links.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.arguments import integer_argument

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
    "source",
    "TF",
    "tf_to_gene_sign",
    "region_to_gene_sign",
    "context",
    "NES",
    "p_value",
    "adjusted_p",
    "region",
    "target",
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
        minTargetGenes: Smallest leading edge of a kept module, and smallest
            co-expression ranking.
        adjustedPThreshold: Kept modules have a Benjamini-Hochberg adjusted
            GSEA p-value below this; SCENIC+ keeps every tested module (1).
        nesThreshold: Kept modules have a GSEA NES above this.
        nPermutations: GSEA gene-set permutations.
        seed: GSEA random seed, the same for every module.
    """

    rhoThreshold: float = 0.05
    quantiles: tuple[float, ...] = (0.85, 0.90, 0.95)
    topRegionsPerGene: tuple[int, ...] = (5, 10, 15)
    minTargetGenes: int = 10
    adjustedPThreshold: float = 1.0
    nesThreshold: float = 0.0
    nPermutations: int = 1000
    seed: int = 666

    def __post_init__(self) -> None:
        if not self.rhoThreshold >= 0:
            raise ValueError("rhoThreshold must be non-negative")
        if any(not 0 <= value <= 1 for value in self.quantiles):
            raise ValueError("quantiles must lie in [0, 1]")
        for value in self.topRegionsPerGene:
            integer_argument(value, "topRegionsPerGene", minimum=1)
        integer_argument(self.minTargetGenes, "minTargetGenes", minimum=1)
        if not 0 < self.adjustedPThreshold <= 1:
            raise ValueError("adjustedPThreshold must lie in (0, 1]")
        if not np.isfinite(self.nesThreshold):
            raise ValueError("nesThreshold must be finite")
        integer_argument(self.nPermutations, "nPermutations", minimum=1)
        # numpy's legacy random state accepts 32-bit unsigned seeds only.
        integer_argument(self.seed, "seed", minimum=0, maximum=2**32 - 1)


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
        (str(gene) in members for gene in genes), dtype=bool, count=n_genes
    )
    hits = np.flatnonzero(tags)
    if len(hits) == 0 or len(hits) == n_genes or not weights[hits].sum() > 0:
        # gseapy divides by zero here; such a module has no score.
        return GseaResult(np.nan, np.nan, np.nan, ())
    norm_miss = 1.0 / (n_genes - len(hits))

    def extremes(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norm_tag = 1.0 / np.sum(weights * rows, axis=1, keepdims=True)
        running = np.cumsum(rows * weights * norm_tag - ~rows * norm_miss, axis=1)
        highest, lowest = running.max(axis=1), running.min(axis=1)
        return np.where(np.abs(highest) > np.abs(lowest), highest, lowest), running

    # gseapy seeds numpy's global generator with ``seed`` and draws one
    # stream seed per gene set; a local generator gives the same draw. The
    # permutations are shuffled in gseapy's order, a bounded block at a time.
    stream_seed = np.random.RandomState(seed).randint(np.iinfo(np.int32).max, size=1)[0]
    random_state = np.random.RandomState(stream_seed)
    null = np.empty(n_permutations)
    for start in range(0, n_permutations, 64):
        rows = np.tile(tags, (min(64, n_permutations - start), 1))
        for row in rows:
            random_state.shuffle(row)
        null[start : start + len(rows)] = extremes(rows)[0]
    scores, running = extremes(tags[None, :])
    observed = float(scores[0])
    observed_running = running[0]
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
) -> list[tuple[str, str, np.ndarray]]:
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
        ``(sign, context, rows)`` triples: sign ``"+"`` or ``"-"``, the
        thresholding rule, and the positions in ``links`` of the links that
        pass it.
    """
    required = {"region", "gene", "importance", "rho"}
    if not required <= set(links.columns):
        raise ValueError(f"links lacks columns {sorted(required - set(links.columns))}")
    if len(links) and links["importance"].isna().all():
        raise ValueError(
            "links have no importances; score them with region_gene_links(importance='gbm')"
        )
    importance = links["importance"].to_numpy(dtype=np.float64)
    rho = links["rho"].to_numpy(dtype=np.float64)
    genes = links["gene"].to_numpy()
    result: list[tuple[str, str, np.ndarray]] = []
    with np.errstate(invalid="ignore"):
        signs = (
            ("-", np.flatnonzero(rho < -parameters.rhoThreshold)),
            ("+", np.flatnonzero(rho > parameters.rhoThreshold)),
        )
    for sign, rows in signs:
        if rows.size == 0:
            continue
        values = importance[rows]
        grouped = pd.Series(values).groupby(genes[rows], sort=False)
        for quantile in parameters.quantiles:
            thresholds = grouped.transform(lambda x, q=quantile: np.quantile(x, q))
            result.append(
                (sign, f"{quantile} quantile", rows[values > thresholds.to_numpy()])
            )
        for n in parameters.topRegionsPerGene:
            thresholds = grouped.transform(
                lambda x, n=n: (
                    np.sort(x.to_numpy())[-n:].min() if len(x) >= n else x.min()
                )
            )
            result.append(
                (
                    sign,
                    f"Top {n} region-to-gene links per gene",
                    rows[values >= thresholds.to_numpy()],
                )
            )
    return result


def _with_self_links(tf_to_gene: pd.DataFrame, factors: set[str]) -> pd.DataFrame:
    """The factors' co-expression links, each factor targeting itself.

    The self-link has the factor's top importance plus 1e-5 and correlation
    1, as SCENIC+ adds before ranking; it replaces any given self-link.
    """
    links = tf_to_gene.loc[
        tf_to_gene["TF"].isin(factors) & (tf_to_gene["TF"] != tf_to_gene["target"]),
        ["TF", "target", "importance", "correlation"],
    ]
    tops = links.groupby("TF")["importance"].max()
    self_links = pd.DataFrame(
        {
            "TF": tops.index,
            "target": tops.index,
            "importance": tops.to_numpy() + 0.00001,
            "correlation": 1.0,
        }
    )
    return pd.concat([links, self_links], ignore_index=True)


def _rankings(
    links: pd.DataFrame, parameters: EregulonParameters
) -> dict[tuple[str, str], pd.Series]:
    """Positive and negative co-expression rankings of each factor."""
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

    Each factor's cistrome selects, from every binarization of the
    region-to-gene links, a module of links. A module is kept when the
    preranked GSEA of its genes against the factor's co-expression ranking
    has a Benjamini-Hochberg adjusted p-value below ``adjustedPThreshold``
    and an NES above ``nesThreshold``, and when the leading edge holds at
    least ``minTargetGenes`` genes. An eRegulon is the union of its kept
    modules' leading-edge links.

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
        nthreads: Factors tested concurrently.

    Returns:
        One row per leading-edge link of each kept module: ``source`` (the
        eRegulon, named ``{TF}_{label}_{TF-to-gene sign}/{region-to-gene
        sign}``), ``TF``, the two signs, the module's ``context`` (its
        binarization) with its ``NES``, ``p_value``, and ``adjusted_p``,
        ``region``, ``target`` (the gene), the link's ``importance_R2G`` and
        ``rho_R2G``, and the target's factor-to-gene ``importance_TF2G`` and
        ``rho_TF2G``. A link kept by several modules, or a gene linked by
        several regions, repeats: filter by the module statistics, then score
        the unique ``source`` and ``target`` pairs.
    """
    for frame, columns, name in (
        (cistromes, {"source", "target"}, "cistromes"),
        (tf_to_gene, {"TF", "target", "importance", "correlation"}, "tf_to_gene"),
    ):
        if not columns <= set(frame.columns):
            raise ValueError(
                f"{name} lacks columns {sorted(columns - set(frame.columns))}"
            )
    integer_argument(nthreads, "nthreads", minimum=1)
    if tf_to_gene.duplicated(["TF", "target"]).any():
        raise ValueError("tf_to_gene repeats a (TF, target) link")
    factors = set(cistromes["source"].astype(str)) & set(tf_to_gene["TF"].astype(str))
    co_expression = _with_self_links(tf_to_gene, factors)
    rankings = _rankings(co_expression, parameters)
    # A factor without a co-expression ranking cannot form a module.
    tested = sorted({tf for tf, _sign in rankings})
    cistrome_regions = {
        str(tf): set(group.astype(str))
        for tf, group in cistromes.groupby("source")["target"]
    }
    # Modules are row positions into the links, never copies of them.
    region_codes, regions = pd.factorize(region_to_gene["region"].astype(str))
    gene_codes, genes = pd.factorize(region_to_gene["gene"].astype(str))
    contexts = {
        (sign, context): rows
        for sign, context, rows in binarize_region_to_gene(
            region_to_gene, parameters=parameters
        )
    }

    def modules(tf: str) -> list[tuple[str, str, np.ndarray]]:
        """The factor's non-empty modules: the links of its cistrome's regions."""
        member = regions.isin(cistrome_regions[tf])
        selected = [
            (sign, context, rows[member[region_codes[rows]]])
            for (sign, context), rows in contexts.items()
        ]
        return [module for module in selected if module[2].size]

    def test(tf: str) -> list[tuple[str, str, str, GseaResult]]:
        results = []
        for r2g_sign, context, rows in modules(tf):
            gene_set = genes[np.unique(gene_codes[rows])]
            for tf2g_sign in ("+", "-"):
                if (tf, tf2g_sign) in rankings:
                    result = gsea_prerank(
                        rankings[(tf, tf2g_sign)],
                        gene_set,
                        n_permutations=parameters.nPermutations,
                        seed=parameters.seed,
                    )
                    results.append((tf2g_sign, r2g_sign, context, result))
        return results

    if nthreads == 1:
        per_factor = [test(tf) for tf in tested]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            per_factor = list(executor.map(test, tested))
    # A module without a score (NaN) is dropped before the correction.
    scored = [
        (tf, *outcome)
        for tf, outcomes in zip(tested, per_factor, strict=True)
        for outcome in outcomes
        if np.isfinite(outcome[-1].enrichmentScore) and np.isfinite(outcome[-1].pValue)
    ]
    adjusted = _bh(np.asarray([outcome[-1].pValue for outcome in scored]))
    passed: dict[str, list[tuple[str, str, str, GseaResult, float]]] = {}
    for (tf, tf2g_sign, r2g_sign, context, result), adjusted_p in zip(
        scored, adjusted, strict=True
    ):
        if (
            adjusted_p < parameters.adjustedPThreshold
            and result.nes > parameters.nesThreshold
        ):
            passed.setdefault(tf, []).append(
                (tf2g_sign, r2g_sign, context, result, float(adjusted_p))
            )
    kept_modules: list[tuple[tuple[Any, ...], np.ndarray]] = []
    for tf, kept in passed.items():
        module_rows = {(sign, context): rows for sign, context, rows in modules(tf)}
        for tf2g_sign, r2g_sign, context, result, adjusted_p in kept:
            rows = module_rows[(r2g_sign, context)]
            rows = rows[
                np.isin(gene_codes[rows], genes.get_indexer(result.leadingEdge))
            ]
            if np.unique(gene_codes[rows]).size < parameters.minTargetGenes:
                continue
            name = f"{tf}_{label}_{tf2g_sign}/{r2g_sign}"
            evidence = (name, tf, tf2g_sign, r2g_sign, context)
            kept_modules.append(
                (evidence + (result.nes, result.pValue, adjusted_p), rows)
            )
    # Each module's evidence repeats over its links, which stay column arrays.
    rows = np.concatenate([np.zeros(0, np.int64)] + [r for _, r in kept_modules])
    module_of_row = np.repeat(
        np.arange(len(kept_modules)), [len(r) for _, r in kept_modules]
    )
    eregulons = pd.DataFrame.from_records(
        [evidence for evidence, _ in kept_modules], columns=list(EREGULON_COLUMNS[:8])
    )
    eregulons = eregulons.iloc[module_of_row].reset_index(drop=True)
    eregulons = eregulons.assign(
        region=regions.to_numpy()[region_codes[rows]],
        target=genes.to_numpy()[gene_codes[rows]],
        importance_R2G=region_to_gene["importance"].to_numpy(dtype=np.float64)[rows],
        rho_R2G=region_to_gene["rho"].to_numpy(dtype=np.float64)[rows],
    )
    eregulons = eregulons.sort_values(
        ["source", "context", "region", "target"], kind="mergesort"
    )
    # Each target's co-expression statistics, the factor's self-link included.
    statistics = co_expression.set_axis(
        ["TF", "target", "importance_TF2G", "rho_TF2G"], axis=1
    ).astype({"TF": str, "target": str})
    return eregulons.merge(statistics, on=["TF", "target"], how="left")
