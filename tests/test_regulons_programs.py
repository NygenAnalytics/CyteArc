import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import fisher_exact
from sklearn.metrics import adjusted_rand_score

from cytearc.regulons.programs import (
    GeneProgramParameters,
    gene_programs,
    program_regulators,
    program_scores,
    set_overlaps,
)
from cytearc.regulons.specificity import (
    connection_specificity,
    regulon_modules,
    regulon_specificity,
)


def _planted(seed=0, n_samples=200, per_program=25):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_samples, 3))
    blocks = []
    truth = []
    for program in range(3):
        loadings = rng.uniform(0.7, 1.3, size=per_program)
        blocks.append(
            factors[:, [program]] * loadings
            + rng.normal(0, 0.5, (n_samples, per_program))
        )
        truth += [program] * per_program
    noise = rng.normal(size=(n_samples, 30))
    values = np.hstack(blocks + [noise])
    names = [f"G{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, columns=names), names, np.asarray(truth), factors


def test_gene_programs_recover_planted_programs():
    profiles, names, truth, factors = _planted()

    programs = gene_programs(
        profiles, parameters=GeneProgramParameters(nNeighbors=10, minSize=10)
    )

    assigned = programs.set_index("target")["source"]
    planted = [name for name in names[: len(truth)]]
    found = assigned.reindex(planted).fillna("none").to_numpy()
    assert adjusted_rand_score(truth, found) > 0.9
    # Uncorrelated genes stay unassigned.
    assert not set(programs["target"]) & set(names[len(truth) :])
    assert programs["source"].nunique() == 3
    assert programs["membership"].min() > 0.5
    scores = program_scores(profiles, programs)
    assert scores.index.equals(profiles.index)
    best = scores.corrwith(pd.Series(factors[:, 0])).abs().max()
    assert best > 0.95


def test_gene_programs_find_none_in_noise_at_metacell_scale():
    # Few samples make chance correlations strong; nearest-neighbour graphs of
    # noise genes must still not pass as programs.
    for samples in (30, 50, 120):
        profiles = np.random.default_rng(samples).standard_normal((samples, 1500))
        programs = gene_programs(
            pd.DataFrame(profiles, columns=[f"G{i}" for i in range(1500)])
        )
        assert programs.empty


@pytest.mark.parametrize(
    ("profiles", "message"),
    [
        (pd.DataFrame(np.ones((5, 2)), columns=["a", "a"]), "unique"),
        (pd.DataFrame(np.ones((3, 2)), columns=["a", "b"]), "four samples"),
        # Constant genes are skipped, which leaves none to link.
        (pd.DataFrame(np.ones((5, 2)), columns=["a", "b"]), "varying genes"),
    ],
)
def test_gene_programs_validates_inputs(profiles, message):
    with pytest.raises(ValueError, match=message):
        gene_programs(profiles)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nNeighbors": 0}, "nNeighbors"),
        ({"minCorrelation": 1.0}, "minCorrelation"),
        ({"minMembership": 1.0}, "minMembership"),
        ({"alpha": 0.0}, "alpha"),
        ({"resolution": 0.0}, "resolution"),
    ],
)
def test_gene_program_parameters_reject_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GeneProgramParameters(**kwargs)


def test_program_regulators_nominate_annotated_factors(tmp_path):
    pl = pytest.importorskip("polars")
    from cytearc.regulons.databases import RankingDatabase

    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(500)]
    rankings = np.vstack([rng.permutation(500) for _ in range(40)])
    program_genes = [f"G{i}" for i in range(20)]
    top = rng.permutation(30)[:20]
    rankings[5] = np.concatenate(
        [top, rng.permutation(np.setdiff1d(np.arange(500), top))]
    )
    columns = {
        gene: rankings[:, index].astype(np.int32) for index, gene in enumerate(genes)
    }
    columns["motifs"] = [f"m{i}" for i in range(40)]
    path = tmp_path / "db.genes_vs_motifs.rankings.feather"
    pl.DataFrame(columns).write_ipc(path, compression="zstd")
    annotations = pd.DataFrame(
        {
            "motif": ["m5", "m5", "m9"],
            "TF": ["SPI1", "SPIB", "PAX5"],
            "category": ["direct", "similarity", "direct"],
        }
    )
    # P2 has a single ranked gene, too few to test.
    programs = pd.DataFrame(
        {"source": ["P1"] * 20 + ["P2"], "target": program_genes + ["G300"]}
    )
    # SPI1's expression follows the program; SPIB is not profiled.
    signal = rng.normal(size=12)
    profiles = pd.DataFrame(
        {gene: signal + rng.normal(0, 0.1, 12) for gene in program_genes}
        | {"SPI1": signal}
    )

    regulators = program_regulators(
        programs, RankingDatabase(path), annotations, rank_threshold=200
    )
    checked = program_regulators(
        programs,
        RankingDatabase(path),
        annotations,
        rank_threshold=200,
        profiles=profiles,
    ).set_index("TF")
    direct = program_regulators(
        programs,
        RankingDatabase(path),
        annotations,
        rank_threshold=200,
        categories=["direct"],
    )
    orthology = program_regulators(
        programs,
        RankingDatabase(path),
        annotations,
        rank_threshold=200,
        categories=["orthology"],
    )

    assert regulators[["TF", "motif", "category"]].values.tolist() == [
        ["SPI1", "m5", "direct"],
        ["SPIB", "m5", "similarity"],
    ]
    assert direct["TF"].tolist() == ["SPI1"]
    # m5 is enriched, but no orthology annotation names a factor for it.
    assert orthology.empty
    assert (regulators["NES"] >= 3).all()
    assert "correlation" not in regulators
    assert checked.loc["SPI1", "correlation"] > 0.9
    assert np.isnan(checked.loc["SPIB", "correlation"])


def test_set_overlaps_match_fisher_exact():
    universe = [f"G{i}" for i in range(100)]
    queries = pd.DataFrame({"source": ["P1"] * 20, "target": universe[:20]})
    references = pd.DataFrame(
        {
            "source": ["SPI1(+)"] * 15 + ["PAX5(+)"] * 10,
            "target": universe[10:25] + universe[50:60],
        }
    )

    overlaps = set_overlaps(queries, references, universe).set_index("reference")

    for reference, members in (
        ("SPI1(+)", set(universe[10:25])),
        ("PAX5(+)", set(universe[50:60])),
    ):
        a = len(set(universe[:20]) & members)
        table = [[a, 20 - a], [len(members) - a, 100 - 20 - len(members) + a]]
        assert overlaps.loc[reference, "overlap"] == a
        assert overlaps.loc[reference, "p_value"] == pytest.approx(
            fisher_exact(table, alternative="greater").pvalue
        )
        assert overlaps.loc[reference, "odds_ratio"] == pytest.approx(
            fisher_exact(table).statistic
        )
    assert overlaps["q_value"].notna().all()
    # A reference covering the universe has an empty margin: no odds ratio.
    whole = pd.DataFrame({"source": "all", "target": universe})
    assert np.isnan(set_overlaps(queries, whole, universe)["odds_ratio"].iloc[0])


def test_regulon_specificity_matches_pyscenic():
    rng = np.random.default_rng(2)
    activity = pd.DataFrame(
        rng.uniform(0, 1, size=(30, 3)), columns=["A(+)", "B(+)", "C(+)"]
    )
    labels = np.repeat(["T", "B", "NK"], 10)
    activity.iloc[:10, 0] += 2.0

    rss = regulon_specificity(activity, labels)

    assert rss.index.tolist() == ["T", "B", "NK"]
    for group in ("T", "B", "NK"):
        for regulon in activity.columns:
            values = activity[regulon].to_numpy()
            onehot = (labels == group).astype(int)
            expected = 1.0 - jensenshannon(values / values.sum(), onehot / onehot.sum())
            assert rss.loc[group, regulon] == pytest.approx(expected)
    assert rss["A(+)"].idxmax() == "T"


@pytest.mark.parametrize(
    ("values", "labels", "message"),
    [
        ([[1.0], [2.0]], ["T"], "align"),
        ([[1.0], [-2.0]], ["T", "B"], "non-negative"),
    ],
)
def test_regulon_specificity_rejects_invalid_inputs(values, labels, message):
    with pytest.raises(ValueError, match=message):
        regulon_specificity(pd.DataFrame(values), labels)


def test_connection_specificity_and_modules():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(300, 2))
    activity = pd.DataFrame(
        np.column_stack(
            [base[:, 0] + rng.normal(0, 0.3, 300) for _ in range(4)]
            + [base[:, 1] + rng.normal(0, 0.3, 300) for _ in range(4)]
        ),
        columns=[f"R{i}" for i in range(8)],
    )

    csi = connection_specificity(activity)

    correlation = np.corrcoef(activity.to_numpy().T)
    for a in range(8):
        for b in range(8):
            r = correlation[a, b] - 0.05
            expected = np.mean(
                [(correlation[a, k] < r) and (correlation[b, k] < r) for k in range(8)]
            )
            assert csi.iloc[a, b] == pytest.approx(expected)
    modules = regulon_modules(csi, 2)
    assert modules.iloc[:4].nunique() == 1 and modules.iloc[4:].nunique() == 1
    assert modules.iloc[0] != modules.iloc[4]
    # Ward linkage needs two regulons; a lone regulon forms module 1.
    lone = regulon_modules(csi.iloc[:1, :1], 2)
    assert lone.tolist() == [1] and lone.name == modules.name == "module"
