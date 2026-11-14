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
from tests.regulon_helpers import csi_by_definition, rss_by_definition


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
    # Few samples make chance correlations strong; nearest-neighbor graphs of
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


def _passes(values: np.ndarray, rows: int):
    """Return a callable that makes one pass over ``values`` in blocks of ``rows`` rows."""
    return lambda: (
        values[start : start + rows] for start in range(0, len(values), rows)
    )


@pytest.mark.parametrize(
    ("n_cells", "n_regulons", "n_groups"),
    [(200, 6, 4), (1500, 25, 40), (60, 3, 60)],
)
def test_specificity_scores_sum_any_blocks_of_cells_into_the_pyscenic_scores(
    n_cells, n_regulons, n_groups
):
    from cytearc.regulons.specificity import specificity_scores

    rng = np.random.default_rng(n_cells)
    values = rng.uniform(0, 0.4, size=(n_cells, n_regulons)).astype(np.float32)
    # A regulon without activity, one active in three cells, and a group of
    # one cell.
    values[:, :2] = 0.0
    values[rng.choice(n_cells, 3, replace=False), 1] = 0.7
    labels = np.asarray([f"g{code}" for code in rng.integers(0, n_groups, n_cells)])
    labels[0] = "lone"
    expected = rss_by_definition(values, labels)
    codes = pd.Index(expected.index).get_indexer(labels)

    for rows in (1, 7, 64, n_cells):
        scores = specificity_scores(
            _passes(values, rows), codes, len(expected), n_regulons
        )
        np.testing.assert_allclose(scores, expected.to_numpy(), rtol=0, atol=1e-12)
    assert np.isnan(scores[:, 0]).all()
    assert not np.isnan(scores[:, 1:]).any()


def test_regulon_specificity_gives_the_scores_derived_by_hand():
    activity = pd.DataFrame({"X": [2.0, 2.0, 0.0, 0.0], "Y": [1.0, 1.0, 1.0, 1.0]})

    rss = regulon_specificity(activity, ["a", "a", "b", "b"])

    # X is active only, and evenly, in group a, and none of its activity is
    # in b's cells, so the divergence from b is ln 2.
    assert rss.loc["a", "X"] == 1.0
    assert rss.loc["b", "X"] == pytest.approx(1 - np.sqrt(np.log(2)), abs=1e-15)
    # Even activity puts 1/4 on each cell against 1/2 on each of the group's
    # two, a divergence of 3/4 ln(4/3).
    even = 1 - np.sqrt(0.75 * np.log(4 / 3))
    assert rss["Y"].tolist() == pytest.approx([even, even], abs=1e-15)


@pytest.mark.parametrize("size", [3, 7, 29])
@pytest.mark.parametrize("value", [0.1, 1 / 3, 0.987, 1e-5])
def test_a_regulon_active_only_and_evenly_in_a_group_scores_one(size, value):
    # The activity of each cell over the total and the group's 1 / size can
    # differ in their last bits; none of the regulon's activity lies outside
    # the group, so its divergence holds no rounding of the other groups.
    values = np.zeros((size + 5, 1), dtype=np.float32)
    values[:size, 0] = value
    labels = ["in"] * size + ["out"] * 5

    rss = regulon_specificity(pd.DataFrame(values), labels)

    assert rss.loc["in", 0] == 1.0
    assert rss.loc["out", 0] == pytest.approx(1 - np.sqrt(np.log(2)), abs=1e-12)


def test_in_memory_specificity_reads_its_table_a_block_of_rows_at_a_time(
    monkeypatch,
):
    from cytearc.regulons import specificity

    rng = np.random.default_rng(5)
    values = rng.integers(0, 9, size=(101, 5))
    labels = np.asarray([None, "B", "C"] * 33 + ["B", "C"], dtype=object)
    labels[50] = np.nan
    # Blocks of three rows of five regulons, beside the default of one.
    monkeypatch.setattr(specificity, "_TABLE_BLOCK_BYTES", 8 * 5 * 3)
    reads: list[int] = []
    add_totals = specificity._add_totals

    def counted(block, totals):
        reads.append(len(block))
        return add_totals(block, totals)

    monkeypatch.setattr(specificity, "_add_totals", counted)
    activity = pd.DataFrame(values, columns=list("VWXYZ"))

    rss = regulon_specificity(activity, labels)
    csi = connection_specificity(activity)

    assert reads == [3] * 33 + [2]
    # A label that matches no cell, as NaN does, scores NaN, and its cell
    # counts in each regulon's total only, as pySCENIC's scores count it.
    expected = rss_by_definition(values, labels)
    assert rss.index[1:3].tolist() == ["B", "C"] and rss.index[[0, 3]].isna().all()
    assert np.isnan(rss.iloc[3]).all() and not np.isnan(rss.iloc[0]).any()
    np.testing.assert_allclose(rss.to_numpy(), expected.to_numpy(), atol=1e-12)
    np.testing.assert_array_equal(csi.to_numpy(), csi_by_definition(values))


def test_the_specificity_sums_let_each_block_go_before_the_next_is_read():
    import weakref

    from cytearc.regulons.specificity import activity_correlation, specificity_scores

    still_held: list[int] = []

    def blocks(values: np.ndarray):
        # A stream that notes each block that is still alive when the next is
        # asked for, which a stream of large blocks would hold twice over.
        previous = None
        for start in range(0, len(values), 4):
            if previous is not None and previous() is not None:
                still_held.append(start)
            block = np.array(values[start : start + 4])
            previous = weakref.ref(block)
            yield block
            del block

    values = np.random.default_rng(6).uniform(size=(30, 5))
    codes = np.arange(30) % 3

    specificity_scores(lambda: blocks(values), codes, 3, 5)
    activity_correlation(blocks(values), 5)

    assert still_held == []


@pytest.mark.parametrize("rows", [1, 5, 97, 5000])
def test_activity_correlation_merges_blocks_without_cancelling_large_sums(rows):
    from cytearc.regulons.specificity import activity_correlation

    rng = np.random.default_rng(3)
    # Activity far from 0 beside its spread, where a correlation from sums of
    # squares and products loses all but about two of its digits.
    values = 1e4 + rng.normal(scale=1e-2, size=(5000, 6))
    values[:, 2] = values[:, 0] + rng.normal(scale=1e-3, size=5000)
    values[:, 5] = 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        reference = np.corrcoef(values.T)

    correlation = activity_correlation(
        (values[start : start + rows] for start in range(0, len(values), rows)), 6
    )

    np.testing.assert_allclose(
        correlation[:5, :5], reference[:5, :5], rtol=0, atol=1e-9
    )
    # A regulon of constant activity correlates 0 with every regulon.
    assert not correlation[5].any() and not correlation[:, 5].any()


@pytest.mark.parametrize("margin", [0.0, 0.05, 0.3])
def test_connection_specificity_counts_the_regulons_below_each_pair(margin):
    rng = np.random.default_rng(4)
    base = rng.normal(size=(400, 3))
    values = np.column_stack(
        [base[:, k % 3] + rng.normal(0, 0.5 + k / 10, 400) for k in range(12)]
    )
    values[:, 11] = 1.0
    activity = pd.DataFrame(values, columns=[f"R{k}" for k in range(12)])

    csi = connection_specificity(activity, margin=margin)

    np.testing.assert_array_equal(csi.to_numpy(), csi_by_definition(values, margin))
    assert csi.index.tolist() == csi.columns.tolist() == list(activity.columns)


@pytest.mark.parametrize("value", [-0.5, np.nan, np.inf])
def test_specificity_scores_refuse_negative_or_non_finite_activity(value):
    from cytearc.regulons.specificity import specificity_scores

    values = np.asarray([[1.0, 2.0], [3.0, value]])

    with pytest.raises(ValueError, match="AUCs must be finite and non-negative"):
        specificity_scores(_passes(values, 1), np.asarray([0, 1]), 2, 2, name="AUCs")


def test_specificity_scores_check_their_codes_and_blocks():
    from cytearc.regulons.specificity import specificity_scores

    values = np.ones((3, 2))
    with pytest.raises(ValueError, match="codes must name groups from 0 to 1, or -1"):
        specificity_scores(_passes(values, 2), np.asarray([0, 2, 1]), 2, 2)
    with pytest.raises(ValueError, match="codes must name groups"):
        specificity_scores(_passes(values, 2), np.asarray([0, -2, 1]), 2, 2)
    with pytest.raises(ValueError, match="must hold 3 columns"):
        specificity_scores(_passes(values, 2), np.asarray([0, 1, 1]), 2, 3)
    with pytest.raises(ValueError, match="activity holds 3 rows for 4 cells"):
        specificity_scores(_passes(values, 2), np.asarray([0, 1, 1, 0]), 2, 2)
    # A second pass must read the rows of the first.
    passes = iter([values, np.ones((4, 2))])
    with pytest.raises(ValueError, match="more rows than its 3 cells"):
        specificity_scores(lambda: [next(passes)], np.asarray([0, 1, 1]), 2, 2)
    passes = iter([values, np.ones((2, 2))])
    with pytest.raises(ValueError, match="activity holds 2 rows for 3 cells"):
        specificity_scores(lambda: [next(passes)], np.asarray([0, 1, 1]), 2, 2)


def test_connection_specificity_checks_its_margin_and_shapes():
    from cytearc.regulons.specificity import (
        activity_correlation,
        connection_specificity_from_correlation,
    )

    activity = pd.DataFrame(np.eye(3))
    with pytest.raises(ValueError, match="margin must be finite"):
        connection_specificity(activity, margin=np.nan)
    with pytest.raises(TypeError, match="margin must be a real number"):
        connection_specificity(activity, margin=True)
    with pytest.raises(ValueError, match="cells-by-regulons"):
        connection_specificity(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="square"):
        connection_specificity_from_correlation(np.zeros((2, 3)), 0.05)
    with pytest.raises(ValueError, match="must hold 2 columns"):
        activity_correlation([np.zeros((2, 3))], 2)
