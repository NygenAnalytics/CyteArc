import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cytearc.regulons.databases import RankingDatabase  # noqa: E402
from cytearc.regulons.modules import (  # noqa: E402
    ModuleParameters,
    modules_from_adjacencies,
)
from cytearc.regulons.pruning import (  # noqa: E402
    PruningParameters,
    prune_modules,
    regulons_from_motifs,
)


def _adjacencies() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for tf in ("TFA", "TFB", "TFC"):
        for gene in range(40):
            rows.append(
                (
                    tf,
                    f"G{gene}",
                    float(rng.uniform(0.1, 10.0)),
                    float(rng.uniform(-0.5, 0.5)),
                )
            )
    frame = pd.DataFrame(rows, columns=["TF", "target", "importance", "correlation"])
    return frame.sort_values(
        "importance", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def _pyscenic_modules(
    adjacencies,
    thresholds=(0.75, 0.90),
    top_targets=(50,),
    top_regulators=(5, 10, 50),
    min_genes=20,
):
    """pySCENIC 0.12.1 ``modules_from_adjacencies`` rules, written with pandas."""
    links = adjacencies.copy()
    links["regulation"] = (links["correlation"] > 0.03).astype(int) - (
        links["correlation"] < -0.03
    ).astype(int)
    cutoffs = list(links["importance"].quantile(list(thresholds)))
    activating = links[links["regulation"] > 0]
    modules = {}

    def add(context, frame):
        for tf, group in frame.groupby("TF"):
            genes = dict(zip(group["target"], group["importance"]))
            genes[tf] = 1.0
            if len(genes) >= min_genes:
                modules[(tf, context)] = genes

    for fraction, cutoff in zip(thresholds, cutoffs):
        add(f"weight>{fraction * 100}%", activating[activating["importance"] > cutoff])

    def largest(frame, by, n):
        groups = [group.nlargest(n, "importance") for _, group in frame.groupby(by)]
        return pd.concat(groups) if groups else frame.iloc[:0]

    for n in top_targets:
        add(f"top{n}", largest(activating, "TF", n))
    for n in top_regulators:
        add(f"top{n}perTarget", largest(activating, "target", n))
    return modules


def test_modules_follow_pyscenic_rules():
    adjacencies = _adjacencies()

    modules = modules_from_adjacencies(adjacencies)

    expected = _pyscenic_modules(adjacencies)
    observed = {
        (tf, context): dict(zip(group["target"], group["weight"]))
        for (tf, context), group in modules.groupby(["TF", "context"])
    }
    assert observed == expected
    assert set(modules["direction"]) == {"activating"}
    # Threshold modules here hold fewer than 20 genes, so only the top families remain.
    assert set(modules["context"]) == {
        "top50",
        "top5perTarget",
        "top10perTarget",
        "top50perTarget",
    }
    relaxed = modules_from_adjacencies(
        adjacencies, parameters=ModuleParameters(minGenes=2)
    )
    assert {"weight>75.0%", "weight>90.0%"} <= set(relaxed["context"])


def test_modules_can_include_repressing_links_and_respect_min_genes():
    adjacencies = _adjacencies()

    both = modules_from_adjacencies(
        adjacencies, parameters=ModuleParameters(keepOnlyActivating=False, minGenes=5)
    )

    assert set(both["direction"]) == {"activating", "repressing"}
    sizes = both.groupby("module").size()
    assert sizes.min() >= 5
    with pytest.raises(ValueError, match="repeats"):
        modules_from_adjacencies(pd.concat([adjacencies, adjacencies.iloc[:1]]))


def _database(path, rankings, genes, motifs):
    columns = {
        gene: rankings[:, index].astype(np.int32) for index, gene in enumerate(genes)
    }
    columns["motifs"] = motifs
    pl.DataFrame(columns).write_ipc(path, compression="zstd")
    return RankingDatabase(path)


def _reference_prune(
    module_genes,
    module_weights,
    tf,
    rankings,
    genes,
    motifs,
    total,
    annotated,
    rank_threshold,
    auc_threshold,
    nes_threshold,
):
    """pySCENIC pruning with each leading edge taken from its own motif."""
    present = [gene for gene in sorted(module_genes) if gene in genes]
    if (len(module_genes) - len(present)) / len(module_genes) >= 0.2:
        return {}
    columns = [genes.index(gene) for gene in present]
    ranks = rankings[:, columns]
    k = int(round(auc_threshold * total))
    aucs = np.array([np.maximum(0, k - 1 - row).sum() for row in ranks]) / (
        k * len(present)
    )
    nes = (aucs - aucs.mean()) / aucs.std()
    curves = np.array(
        [
            np.cumsum(np.bincount(row, minlength=total + 1)[:rank_threshold])
            for row in ranks
        ]
    ).astype(float)
    envelope = curves.mean(axis=0) + 2.0 * curves.std(axis=0)
    result = {}
    for index in np.flatnonzero(nes >= nes_threshold):
        if (tf, motifs[index]) not in annotated:
            continue
        rank_at_max = int(np.argmax(curves[index] - envelope))
        order = np.argsort(ranks[index])
        result[motifs[index]] = [
            (present[i], module_weights[present[i]])
            for i in order
            if ranks[index, i] <= rank_at_max
        ]
    return result


def test_prune_modules_matches_reference(tmp_path):
    rng = np.random.default_rng(5)
    total = 600
    genes = [f"G{index}" for index in range(total)]
    motifs = [f"m{index}" for index in range(80)]
    rankings = np.vstack([rng.permutation(total) for _ in motifs])
    module_genes = [f"G{index}" for index in range(30)] + ["NOTINDB"]
    # Plant enrichment for two annotated motifs and one unannotated motif:
    # module genes G0..G29 take ranks among the top 40.
    for motif in (3, 7, 11):
        top = rng.permutation(40)[:30]
        rest = rng.permutation(np.setdiff1d(np.arange(total), top))
        rankings[motif] = np.concatenate([top, rest])
    database = _database(tmp_path / "db.feather", rankings, genes, motifs)
    weights = {gene: float(rng.uniform(1, 5)) for gene in module_genes}
    weights["G0"] = 1.0
    modules = pd.DataFrame(
        {
            "module": "G0|activating|top50",
            "TF": "G0",
            "direction": "activating",
            "context": "top50",
            "target": module_genes,
            "weight": [weights[gene] for gene in module_genes],
        }
    )
    annotations = pd.DataFrame(
        {
            "motif": ["m3", "m7", "m7"],
            "TF": ["G0", "G0", "G0"],
            "motif_similarity_qvalue": [0.0, 0.0005, 0.0],
            "orthologous_identity": [1.0, 1.0, 1.0],
            "description": [
                "gene is directly annotated",
                "similar",
                "gene is directly annotated",
            ],
        }
    )
    parameters = PruningParameters(
        rankThreshold=200, aucThreshold=0.05, nesThreshold=3.0
    )

    result = prune_modules(
        modules, [database], annotations, parameters=parameters, nthreads=2
    )

    expected = _reference_prune(
        module_genes,
        weights,
        "G0",
        rankings,
        genes,
        motifs,
        total,
        {("G0", "m3"), ("G0", "m7")},
        200,
        0.05,
        3.0,
    )
    assert set(result["motif"]) == set(expected) == {"m3", "m7"}
    for _, row in result.iterrows():
        assert (
            list(zip(row["targets"], row["target_weights"])) == expected[row["motif"]]
        )
    # The best annotation per (TF, motif) has the smallest q-value.
    m7 = result.set_index("motif").loc["m7"]
    assert m7["motif_similarity_qvalue"] == 0.0
    assert m7["annotation"] == "gene is directly annotated"

    regulons = regulons_from_motifs(result)
    assert set(regulons["regulon"]) == {"G0(+)"}
    merged = {}
    for edge in expected.values():
        for gene, weight in edge:
            merged[gene] = max(weight, merged.get(gene, 0.0))
    assert dict(zip(regulons["target"], regulons["weight"])) == merged


def test_prune_modules_skips_modules_missing_from_the_database(tmp_path):
    rng = np.random.default_rng(2)
    total = 600
    genes = [f"G{index}" for index in range(total)]
    rankings = np.vstack([rng.permutation(total) for _ in range(80)])
    # m3 ranks G0..G23 among its top 30, so it is enriched for the module.
    top = rng.permutation(30)[:24]
    rest = rng.permutation(np.setdiff1d(np.arange(total), top))
    rankings[3] = np.concatenate([top, rest])
    database = _database(
        tmp_path / "db.feather", rankings, genes, [f"m{index}" for index in range(80)]
    )
    annotations = pd.DataFrame(
        {
            "motif": ["m3"],
            "TF": ["G0"],
            "motif_similarity_qvalue": [0.0],
            "orthologous_identity": [1.0],
            "description": ["d"],
        }
    )

    def prune(n_missing):
        targets = genes[:24] + [f"X{index}" for index in range(n_missing)]
        modules = pd.DataFrame(
            {
                "module": "M",
                "TF": "G0",
                "direction": "activating",
                "context": "top50",
                "target": targets,
                "weight": 1.0,
            }
        )
        return prune_modules(
            modules,
            [database],
            annotations,
            parameters=PruningParameters(rankThreshold=100),
        )

    # The motif is found while under a fifth of the module is missing, and
    # the module is skipped from exactly a fifth, as pySCENIC does.
    assert prune(5)["motif"].tolist() == ["m3"]
    assert prune(6).empty
