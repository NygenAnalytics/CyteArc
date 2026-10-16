from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cytearc.regulons.eregulons import (
    EREGULON_COLUMNS,
    EregulonParameters,
    binarize_region_to_gene,
    build_eregulons,
    gsea_prerank,
)


def _gseapy_0_10_8(gene_list, correl_vector, gene_set, nperm, seed):
    """gseapy 0.10.8 gsea_compute + enrichment_score for one gene set.

    Adapted from gseapy (https://github.com/zqfang/GSEApy), Copyright (c) 2017
    Zhuoqing Fang, BSD 3-Clause License.
    """
    np.random.seed(seed)
    stream = np.random.randint(np.iinfo(np.int32).max, size=1)[0]
    N = len(gene_list)
    # np.in1d in gseapy; np.isin is the same test in NumPy 2.
    tag_indicator = np.isin(gene_list, gene_set, assume_unique=True).astype(int)
    correl_vector = np.abs(correl_vector) ** 1
    hit_ind = np.flatnonzero(tag_indicator).tolist()
    tag_indicator = np.tile(tag_indicator, (nperm + 1, 1))
    correl_vector = np.tile(correl_vector, (nperm + 1, 1))
    rs = np.random.RandomState(stream)
    for i in range(nperm):
        rs.shuffle(tag_indicator[i])
    Nhint = tag_indicator.sum(axis=1, keepdims=True)
    sum_correl_tag = np.sum(correl_vector * tag_indicator, axis=1, keepdims=True)
    no_tag_indicator = 1 - tag_indicator
    RES = np.cumsum(
        tag_indicator * correl_vector * (1.0 / sum_correl_tag)
        - no_tag_indicator * (1.0 / (N - Nhint)),
        axis=1,
    )
    max_ES, min_ES = RES.max(axis=1), RES.min(axis=1)
    es_vec = np.where(np.abs(max_ES) > np.abs(min_ES), max_ES, min_ES)
    es, esnull, RES = es_vec[-1], es_vec[:-1], RES[-1, :]
    nes = (
        es / esnull[esnull >= 0].mean() if es >= 0 else -es / esnull[esnull < 0].mean()
    )
    pval = (
        (esnull >= es).sum() / (esnull >= 0).sum()
        if es >= 0
        else (esnull < es).sum() / (esnull < 0).sum()
    )
    if es > 0:
        idx = RES.argmax()
        ldg = [x for x in hit_ind if x <= idx]
    else:
        idx = RES.argmin()
        ldg = [x for x in hit_ind if x >= idx]
    return es, nes, pval, [gene_list[i] for i in ldg]


@pytest.mark.parametrize("seed", [666, 555, 1])
def test_gsea_prerank_matches_gseapy(seed):
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i}" for i in range(300)], dtype=object)
    weights = np.sort(rng.exponential(1.0, size=300))[::-1]
    gene_set = list(genes[rng.choice(60, size=15, replace=False)]) + ["ABSENT"]
    ranked = pd.Series(weights, index=genes)

    result = gsea_prerank(ranked, gene_set, n_permutations=200, seed=seed)

    es, nes, pval, edge = _gseapy_0_10_8(genes, weights, gene_set, 200, seed)
    assert result.enrichmentScore == pytest.approx(es, abs=1e-12)
    assert result.nes == pytest.approx(nes, rel=1e-12)
    assert result.pValue == pytest.approx(pval)
    assert list(result.leadingEdge) == edge
    assert result.nes > 1 and set(result.leadingEdge) <= set(gene_set)


def test_gsea_prerank_without_hits_has_no_score():
    ranked = pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"])

    result = gsea_prerank(ranked, ["z"], n_permutations=10)

    assert np.isnan(result.enrichmentScore) and result.leadingEdge == ()


def test_gsea_prerank_with_a_missing_weight_has_no_score():
    # A NaN weight makes every running sum NaN, observed and null alike.
    ranked = pd.Series([3.0, np.nan, 1.0], index=["a", "b", "c"])

    result = gsea_prerank(ranked, ["a"], n_permutations=10)

    assert np.isnan([result.enrichmentScore, result.nes, result.pValue]).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rhoThreshold": -0.1}, "rhoThreshold"),
        ({"quantiles": (0.5, 1.5)}, "quantiles"),
        ({"seed": -1}, "seed"),
        ({"adjustedPThreshold": 0.0}, "adjustedPThreshold"),
        ({"adjustedPThreshold": 1.5}, "adjustedPThreshold"),
        ({"nesThreshold": np.inf}, "nesThreshold"),
    ],
)
def test_eregulon_parameters_reject_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EregulonParameters(**kwargs)


def test_binarization_follows_scenicplus_rules():
    rng = np.random.default_rng(0)
    links = pd.DataFrame(
        {
            "region": [f"r{i}" for i in range(60)],
            "gene": np.repeat(["A", "B", "C"], 20),
            "importance": rng.uniform(0, 1, size=60),
            "rho": rng.uniform(-0.3, 0.3, size=60),
        }
    )
    parameters = EregulonParameters(quantiles=(0.85,), topRegionsPerGene=(5,))

    sets = binarize_region_to_gene(links, parameters=parameters)

    assert [(sign, context) for sign, context, _ in sets] == [
        ("-", "0.85 quantile"),
        ("-", "Top 5 region-to-gene links per gene"),
        ("+", "0.85 quantile"),
        ("+", "Top 5 region-to-gene links per gene"),
    ]
    for sign, context, rows in sets:
        selected = links.iloc[rows]
        subset = links[(links["rho"] > 0.05) if sign == "+" else (links["rho"] < -0.05)]
        for gene, group in subset.groupby("gene"):
            values = group["importance"].to_numpy()
            if "quantile" in context:
                expected = set(group["region"][values > np.quantile(values, 0.85)])
            else:
                cutoff = (
                    np.sort(values)[-5:].min() if len(values) >= 5 else values.min()
                )
                expected = set(group["region"][values >= cutoff])
            assert set(selected.loc[selected["gene"] == gene, "region"]) == expected
    with pytest.raises(ValueError, match="lacks columns"):
        binarize_region_to_gene(links.drop(columns="rho"))


def test_build_eregulons_keeps_co_expressed_targets_of_the_cistrome():
    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(60)]
    # SPI1's top co-expression targets are G0..G19; CEBPA's are G40..G59.
    tf_to_gene = pd.concat(
        [
            pd.DataFrame(
                {
                    "TF": tf,
                    "target": genes,
                    "importance": np.concatenate(
                        [np.linspace(50, 30, 20), rng.uniform(0, 5, 40)]
                    )
                    if tf == "SPI1"
                    else np.concatenate(
                        [rng.uniform(0, 5, 40), np.linspace(50, 30, 20)]
                    ),
                    "correlation": 0.4,
                }
            )
            for tf in ("SPI1", "CEBPA")
        ],
        ignore_index=True,
    )
    # Each gene has three regions; region 0 of each gene is strongly linked.
    rows = []
    for gene in genes:
        for k in range(3):
            rows.append((f"{gene}_r{k}", gene, 10.0 if k == 0 else 1.0, 0.3))
    rows_columns = ["region", "gene", "importance", "rho"]
    region_to_gene = pd.DataFrame(rows, columns=rows_columns)
    # SPI1 binds the strong regions of G0..G24; CEBPA binds those of G0..G24 too.
    cistromes = pd.DataFrame(
        {
            "source": ["SPI1"] * 25 + ["CEBPA"] * 25,
            "target": [f"G{i}_r0" for i in range(25)] * 2,
        }
    )
    parameters = EregulonParameters(nPermutations=200, minTargetGenes=10)

    eregulons = build_eregulons(
        cistromes, region_to_gene, tf_to_gene, parameters=parameters, nthreads=2
    )

    names = set(eregulons["source"])
    assert names == {"SPI1_direct_+/+"}
    spi1 = eregulons[eregulons["source"] == "SPI1_direct_+/+"]
    assert set(spi1["target"]) <= {f"G{i}" for i in range(25)}
    assert len(set(spi1["target"]) & {f"G{i}" for i in range(20)}) >= 10
    assert (spi1["importance_TF2G"] >= 30).all()
    # Every link carries the statistics of the module that kept it.
    modules = spi1.drop_duplicates("context")
    assert set(modules["context"]) <= {
        f"{q} quantile" for q in parameters.quantiles
    } | {f"Top {n} region-to-gene links per gene" for n in parameters.topRegionsPerGene}
    assert (modules["NES"] > 0).all()
    assert (modules["adjusted_p"] >= modules["p_value"]).all()
    assert (spi1.groupby("context")["adjusted_p"].nunique() == 1).all()
    # No module passes an NES above the largest; each module holds at most
    # the 25 genes the cistrome links to, so no leading edge has 30 targets.
    for strict in (
        replace(parameters, nesThreshold=modules["NES"].max()),
        replace(parameters, minTargetGenes=30),
    ):
        assert build_eregulons(
            cistromes, region_to_gene, tf_to_gene, parameters=strict
        ).empty

    # The factor's own gene carries its injected self-link, as SCENIC+ reports.
    with_self = build_eregulons(
        pd.concat([cistromes, pd.DataFrame({"source": ["SPI1"], "target": ["s_r0"]})]),
        pd.concat(
            [
                region_to_gene,
                pd.DataFrame([("s_r0", "SPI1", 10.0, 0.3)], columns=rows_columns),
            ]
        ),
        tf_to_gene,
        parameters=parameters,
    )
    self_row = with_self.loc[with_self["target"] == "SPI1"]
    assert not self_row.empty
    assert self_row["importance_TF2G"].to_numpy() == pytest.approx(50 + 1e-5)
    assert (self_row["rho_TF2G"] == 1.0).all()

    # Invalid inputs raise rather than return an empty table.
    with pytest.raises(ValueError, match="cistromes lacks"):
        build_eregulons(cistromes[["source"]], region_to_gene, tf_to_gene)
    with pytest.raises(ValueError, match="repeats"):
        build_eregulons(
            cistromes, region_to_gene, pd.concat([tf_to_gene, tf_to_gene.iloc[:1]])
        )
    with pytest.raises(ValueError, match="importances"):
        build_eregulons(cistromes, region_to_gene.assign(importance=np.nan), tf_to_gene)


def test_eregulons_of_random_links_fail_a_significance_threshold():
    rng = np.random.default_rng(5)
    genes = [f"G{i}" for i in range(80)]
    factors = [f"TF{i}" for i in range(4)]
    # Positive correlations, but random importances and memberships.
    tf_to_gene = pd.DataFrame(
        [(tf, gene, rng.gamma(1.0), 0.3) for tf in factors for gene in genes],
        columns=["TF", "target", "importance", "correlation"],
    )
    region_to_gene = pd.DataFrame(
        [(f"r{i}", rng.choice(genes), rng.gamma(1.0), 0.3) for i in range(400)],
        columns=["region", "gene", "importance", "rho"],
    ).drop_duplicates(["region", "gene"])
    cistromes = pd.DataFrame(
        [(tf, f"r{i}") for tf in factors for i in rng.choice(400, 200, replace=False)],
        columns=["source", "target"],
    )
    parameters = EregulonParameters(nPermutations=200, minTargetGenes=5)

    permissive = build_eregulons(
        cistromes, region_to_gene, tf_to_gene, parameters=parameters
    )
    strict = build_eregulons(
        cistromes,
        region_to_gene,
        tf_to_gene,
        parameters=replace(parameters, adjustedPThreshold=0.05),
    )

    # SCENIC+'s defaults keep modules whose adjusted p-values show no evidence;
    # the reported statistics let a stricter threshold remove them.
    assert not permissive.empty
    assert (permissive["adjusted_p"] > 0.05).all()
    assert strict.empty


def test_build_eregulons_without_a_ranked_factor_is_empty():
    # SPI1's one region link passes only the top-n thresholds, and its two
    # co-expression targets, the self-link included, are too few to rank.
    cistromes = pd.DataFrame({"source": ["SPI1"], "target": ["r1"]})
    region_to_gene = pd.DataFrame(
        {"region": ["r1"], "gene": ["G1"], "importance": [1.0], "rho": [0.3]}
    )
    tf_to_gene = pd.DataFrame(
        {"TF": ["SPI1"], "target": ["G1"], "importance": [1.0], "correlation": [0.4]}
    )

    eregulons = build_eregulons(cistromes, region_to_gene, tf_to_gene)

    assert eregulons.empty and list(eregulons.columns) == list(EREGULON_COLUMNS)
