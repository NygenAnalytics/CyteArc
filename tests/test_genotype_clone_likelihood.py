"""Genotype clones called by the likelihood of SCClone's mixture model.

The kernels are checked against direct counterparts written from the model's
definition, each call's probability under each genotype; the search's bound
against the log-likelihood that EM then finds in every round; and the calls
against synthetic truths whose clones and dropout and error rates are drawn
first, which the code under test never computes.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import adjusted_rand_score

from cytearc.utils.memory_budget import MemoryLedger
from cytearc.variants import genotype_mixture as mixture_module
from cytearc.variants.genotype_clones import (
    _removal_deltas,
    genotype_clone_rules,
    likelihood_clones,
)
from cytearc.variants.genotype_mixture import (
    N_CHUNKS,
    GenotypeMixture,
    MixtureFit,
    SiteRates,
    bit_pairs,
    call_patterns,
    hard_call_sums,
    pack_calls,
    pack_mask,
    spanning_tree,
)
from tests.clone_helpers import every_cell, write_cells_store
from tests.variant_truth import (
    DOUBLET_CELL,
    DropoutTruth,
    dropout_truth,
    write_call_tables,
)


def _packed(calls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pack_mask(calls == 1), pack_mask(calls == 0)


def _call(truth_calls: np.ndarray, *, nthreads: int = 2, **rules) -> object:
    mutant, wild = _packed(truth_calls)
    return likelihood_clones(
        mutant,
        wild,
        truth_calls.shape[1],
        dropout_probability=rules.get("dropout_probability"),
        error_probability=rules.get("error_probability"),
        min_size=rules.get("min_size", 2),
        min_posterior=rules.get("min_posterior", 0.6),
        ledger=MemoryLedger(None, "Calling clones"),
        nthreads=nthreads,
    )


def _model_loglik(
    calls: np.ndarray, genotype: np.ndarray, dropout: np.ndarray, error: np.ndarray
) -> np.ndarray:
    """Each cell's log-likelihood of its calls under one genotype, call by call."""
    carried = genotype[None, :]
    mutant = calls == 1
    wild = calls == 0
    probability = np.where(
        carried,
        np.where(mutant, 1 - dropout, np.where(wild, dropout, 1.0)),
        np.where(mutant, error, np.where(wild, 1 - error, 1.0)),
    )
    return np.log(probability).sum(axis=1)


def _random_calls(rng: np.random.Generator, n_cells: int, n_sites: int) -> np.ndarray:
    return rng.choice(
        np.array([-1, 0, 1], dtype=np.int8), size=(n_cells, n_sites), p=[0.2, 0.5, 0.3]
    )


def test_word_bit_counts_match_python_integers() -> None:
    rng = np.random.default_rng(1)
    words = rng.integers(0, 2**63, size=500, dtype=np.uint64) * np.uint64(
        2
    ) + rng.integers(0, 2, size=500, dtype=np.uint64)
    words[:4] = [1, 2**63, 2**64 - 1, 6]
    for word in words.tolist():
        value = np.uint64(word)
        assert int(mixture_module._popcount(value)) == word.bit_count()
        if word:
            assert (
                int(mixture_module._lowest_bit(value))
                == (word & -word).bit_length() - 1
            )


@pytest.mark.parametrize("n_sites", [1, 40, 64, 65, 130])
def test_packed_calls_hold_one_bit_per_known_call_at_a_chosen_site(n_sites) -> None:
    rng = np.random.default_rng(n_sites)
    n_cells, n_stored = 30, n_sites + 5
    # Call codes 0 to 5 at stored sites, five of which are not chosen.
    cells = np.repeat(np.arange(n_cells), n_stored)
    stored = np.tile(np.arange(n_stored), n_cells)
    codes = rng.integers(0, 6, size=len(cells)).astype(np.int8)
    chosen = np.sort(rng.permutation(n_stored)[:n_sites])
    rank = np.full(n_stored, -1, dtype=np.int64)
    rank[chosen] = np.arange(n_sites)
    is_mutant = np.zeros(6, dtype=bool)
    is_mutant[[1, 2, 3]] = True
    words = mixture_module.n_words(n_sites)
    mutant_bits = np.zeros((n_cells, words), dtype=np.uint64)
    wild_bits = np.zeros((n_cells, words), dtype=np.uint64)
    pack_calls(
        cells, stored.astype(np.int32), codes, rank, is_mutant, mutant_bits, wild_bits
    )
    dense = np.full((n_cells, n_stored), -1)
    dense[cells, stored] = np.where(
        np.isin(codes, [1, 2, 3]), 1, np.where(codes == 0, 0, -1)
    )
    expected = dense[:, chosen]
    np.testing.assert_array_equal(mutant_bits, pack_mask(expected == 1))
    np.testing.assert_array_equal(wild_bits, pack_mask(expected == 0))


@pytest.mark.parametrize("n_sites", [7, 64, 150])
def test_the_tree_e_step_equals_each_genotype_summed_over_every_call(n_sites) -> None:
    rng = np.random.default_rng(n_sites)
    calls = _random_calls(rng, 300, n_sites)
    # Genotypes near one another and far apart, so the spanning tree has
    # long and short edges.
    base = rng.random((4, n_sites)) < 0.3
    genotypes = np.vstack([base, base ^ (rng.random((4, n_sites)) < 0.05)])
    dropout = rng.uniform(0.05, 0.5, n_sites)
    error = rng.uniform(0.001, 0.05, n_sites)
    mutant, wild = _packed(calls)
    mixture = GenotypeMixture(
        mutant,
        wild,
        n_sites,
        dropout_probability=None,
        error_probability=None,
        ledger=MemoryLedger(None, "Calling clones"),
    )
    relative = np.empty((300, len(genotypes) + 1))
    mixture.relative(genotypes, SiteRates(dropout, error, 0.2, 0.01), relative)
    wild_type = _model_loglik(calls, np.zeros(n_sites, dtype=bool), dropout, error)
    expected = np.column_stack(
        [np.zeros(300)]
        + [
            _model_loglik(calls, genotype, dropout, error) - wild_type
            for genotype in genotypes
        ]
    )
    np.testing.assert_allclose(relative, expected, rtol=1e-10, atol=1e-9)


def test_the_m_step_sums_equal_the_posterior_weighted_calls() -> None:
    rng = np.random.default_rng(3)
    n_cells, n_sites, n_components = 500, 130, 6
    calls = _random_calls(rng, n_cells, n_sites)
    genotypes = rng.random((n_components - 1, n_sites)) < 0.3
    posteriors = rng.dirichlet(np.ones(n_components), size=n_cells)
    mutant_bits, wild_bits = _packed(calls)
    gbits = pack_mask(np.vstack([np.zeros((1, n_sites), dtype=bool), genotypes]))
    words = mutant_bits.shape[1]
    mutant = np.empty((n_components, n_sites))
    wild = np.empty((n_components, n_sites))
    mixture_module._call_sums(
        mutant_bits,
        wild_bits,
        gbits,
        posteriors,
        n_sites,
        np.empty((N_CHUNKS, n_components, n_components)),
        np.empty((3, N_CHUNKS, n_components, n_sites)),
        np.empty((N_CHUNKS, 3, words), dtype=np.uint64),
        mutant,
        wild,
    )
    np.testing.assert_allclose(
        mutant, posteriors.T @ (calls == 1), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(wild, posteriors.T @ (calls == 0), rtol=1e-9, atol=1e-9)


def test_hard_call_sums_count_the_calls_of_each_component_cells() -> None:
    rng = np.random.default_rng(4)
    calls = _random_calls(rng, 400, 70)
    hard = rng.integers(0, 5, size=400)
    mutant_bits, wild_bits = _packed(calls)
    mutant = np.empty((5, 70))
    wild = np.empty((5, 70))
    hard_call_sums(
        mutant_bits, wild_bits, hard, 70, np.empty((2, N_CHUNKS, 5, 70)), mutant, wild
    )
    for component in range(5):
        rows = calls[hard == component]
        np.testing.assert_array_equal(mutant[component], (rows == 1).sum(axis=0))
        np.testing.assert_array_equal(wild[component], (rows == 0).sum(axis=0))


def test_removal_scores_equal_the_log_likelihood_without_each_component() -> None:
    truth = dropout_truth(n_cells=1_500, n_sites=30, n_clones=6, seed=7)
    mutant, wild = _packed(truth.calls)
    mixture = GenotypeMixture(
        mutant,
        wild,
        30,
        dropout_probability=None,
        error_probability=None,
        ledger=MemoryLedger(None, "Calling clones"),
    )
    start = MixtureFit(
        truth.genotypes.copy(), np.log(np.full(7, 1 / 7)), mixture.initial_rates()
    )
    fit = mixture.em(start, max_iter=30)
    deltas = _removal_deltas(fit, 1_500)
    totals = fit.relative + fit.log_weights
    full = np.logaddexp.reduce(totals, axis=1)
    for component in range(1, 7):
        kept = np.delete(totals, component, axis=1)
        renormalised = np.logaddexp.reduce(np.delete(fit.log_weights, component))
        expected = float(
            (np.logaddexp.reduce(kept, axis=1) - renormalised - full).sum()
        )
        assert deltas[component] == pytest.approx(expected, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("n_local", [3, 10, 12, 40, 70])
def test_call_patterns_group_equal_rows_and_sum_their_weights(n_local) -> None:
    rng = np.random.default_rng(n_local)
    calls = rng.choice(
        np.array([-1, 0, 1], dtype=np.int8), size=(800, n_local), p=[0.1, 0.6, 0.3]
    )
    # Few distinct rows, so patterns repeat.
    calls = calls[rng.integers(0, 40, size=800)]
    weights = rng.random(800)
    weights[:5] = 0.0
    local_mutant, local_wild = _packed(calls)
    pattern_mutant, pattern_wild, summed, inverse = call_patterns(
        local_mutant, local_wild, weights, n_local
    )
    # Each row maps to the pattern of its own calls.
    np.testing.assert_array_equal(pattern_mutant[inverse], local_mutant)
    np.testing.assert_array_equal(pattern_wild[inverse], local_wild)
    expected: dict[bytes, float] = {}
    for row, weight in zip(calls, weights, strict=True):
        key = row.tobytes()
        expected[key] = expected.get(key, 0.0) + weight
    assert len(summed) == len(expected)
    for position in range(len(summed)):
        row = np.flatnonzero(inverse == position)[0]
        assert summed[position] == pytest.approx(
            expected[calls[row].tobytes()], rel=1e-12
        )


def test_the_spanning_tree_is_as_short_as_a_minimum_spanning_tree() -> None:
    rng = np.random.default_rng(9)
    genotypes = rng.random((12, 90)) < 0.2
    nodes = np.vstack([np.zeros((1, 90), dtype=bool), genotypes])
    total, parent, order = spanning_tree(pack_mask(nodes))
    distance = (nodes[:, None, :] != nodes[None, :, :]).sum(axis=2)
    assert total == int(minimum_spanning_tree(distance).sum())
    # Each node joins after its parent, and the tree's edges sum to its length.
    joined = np.empty(13, dtype=np.int64)
    joined[order] = np.arange(13)
    assert order[0] == 0 and all(
        joined[parent[node]] < joined[node] for node in range(1, 13)
    )
    assert sum(distance[node, parent[node]] for node in range(1, 13)) == total


def test_bit_pairs_list_each_mutant_call_and_each_unread_site() -> None:
    rng = np.random.default_rng(10)
    calls = _random_calls(rng, 50, 70)
    mutant, wild = _packed(calls)
    rows = np.array([1, 4, 9, 30])
    ledger = MemoryLedger(None, "Calling clones")
    for unknown, value in ((False, 1), (True, -1)):
        cells, sites, held = bit_pairs(ledger, mutant, wild, rows, 70, unknown=unknown)
        expected_cells, expected_sites = np.nonzero(calls[rows] == value)
        np.testing.assert_array_equal(cells, rows[expected_cells])
        np.testing.assert_array_equal(sites, expected_sites)
        assert held == 16 * len(cells)


@pytest.mark.parametrize(
    "settings",
    [
        {"min_own": 2, "dropout": 0.2, "seed": 2},
        {"min_own": 1, "dropout": 0.2, "seed": 5},
        {"min_own": 2, "dropout": 0.3, "seed": 3, "doublet_share": 0.04},
    ],
    ids=["two own sites", "one own site", "doublets"],
)
def test_the_first_e_step_reaches_the_joint_bound_in_every_round(settings) -> None:
    # A split is accepted by a lower bound on its gain without EM over every
    # cell; EM's first E-step after it computes the log-likelihood exactly,
    # which the bound must not pass.
    truth = dropout_truth(n_cells=2_000, n_sites=40, n_clones=8, **settings)
    called = _call(truth.calls)
    assert len(called.bound_checks) >= 2
    for actual, bound in called.bound_checks:
        assert bound > 0
        assert actual >= bound - 1e-7 * max(1.0, abs(bound)), (actual, bound)


def _singlet_ari(truth: DropoutTruth, codes: np.ndarray) -> float:
    singlet = truth.clones != DOUBLET_CELL
    reference = np.where(truth.clones >= 0, truth.clones, -1)[singlet]
    return float(adjusted_rand_score(reference, codes[singlet]))


@pytest.mark.parametrize(
    ("dropout", "seed", "least_ari"),
    [(0.1, 1, 0.85), (0.2, 2, 0.8), (0.2, 8, 0.8)],
)
def test_the_likelihood_recovers_the_clones_and_rates_of_a_dropout_truth(
    dropout, seed, least_ari
) -> None:
    # Eight clones, each two sites or more from its parent, among 2,000
    # cells with 10% wild-type cells, 5% of calls unknown and each site's
    # dropout and error drawn around the stated rates.
    truth = dropout_truth(
        n_cells=2_000, n_sites=40, n_clones=8, dropout=dropout, seed=seed
    )
    called = _call(truth.calls)
    assert len(called.sizes) == 8
    assert _singlet_ari(truth, called.codes) >= least_ari
    realized_dropout, realized_error = truth.realized_rates()
    assert called.carried_wild / called.carried_calls == pytest.approx(
        realized_dropout, abs=0.03
    )
    assert called.absent_mutant / called.absent_calls == pytest.approx(
        realized_error, abs=0.002
    )
    # Every clone holds at least min_size cells, numbered by size.
    assert (called.sizes >= 2).all()
    assert (np.diff(called.sizes) <= 0).all()
    np.testing.assert_array_equal(
        np.bincount(called.codes[called.codes >= 0], minlength=len(called.sizes)),
        called.sizes,
    )
    # A cell with a clone has its posterior of at least min_posterior; a
    # cell without one has none, and only cells with a mutant call are
    # ambiguous.
    clustered = called.codes >= 0
    assert (called.probability[clustered] >= 0.6).all()
    assert np.isnan(called.probability[~clustered]).all()
    has_mutant = (truth.calls == 1).any(axis=1)
    assert not (called.ambiguous & ~has_mutant).any()
    assert not (called.ambiguous & clustered).any()
    assert not (clustered & ~has_mutant).any()


def test_the_exact_rule_fragments_the_same_truth() -> None:
    # The exact rule links cells that agree wherever both were called, so
    # one dropout splits a cell from its clone: hundreds of clones, against
    # the likelihood's eight.
    from cytearc.clones.calling import cell_sets, set_clones

    truth = dropout_truth(n_cells=2_000, n_sites=40, n_clones=8, dropout=0.2, seed=2)
    rows, sites = np.nonzero(truth.calls == 1)
    unknown_rows, unknown_sites = np.nonzero(truth.calls == -1)
    mutant_cells = np.zeros(2_000, dtype=bool)
    mutant_cells[rows] = True
    kept = mutant_cells[unknown_rows]
    exact = set_clones(
        cell_sets(rows, sites, 2_000),
        unknown=cell_sets(unknown_rows[kept], unknown_sites[kept], 2_000),
        method="jaccard",
        similarity=1.0,
        inclusive=True,
        ambiguous=True,
    )
    assert exact.n_clones > 100
    assert _singlet_ari(truth, exact.codes) < 0.3
    assert _singlet_ari(truth, _call(truth.calls).codes) > 0.8


def test_held_rates_are_used_at_every_site_and_estimated_ones_are_not() -> None:
    truth = dropout_truth(n_cells=1_500, n_sites=30, n_clones=6, dropout=0.2, seed=11)
    held = _call(truth.calls, dropout_probability=0.25)
    np.testing.assert_array_equal(held.dropout, np.full(30, 0.25))
    assert len(np.unique(held.error)) > 1
    both = _call(truth.calls, dropout_probability=0.25, error_probability=0.02)
    np.testing.assert_array_equal(both.error, np.full(30, 0.02))
    estimated = _call(truth.calls)
    assert len(np.unique(estimated.dropout)) > 1


def test_results_do_not_change_with_the_threads() -> None:
    truth = dropout_truth(
        n_cells=3_000, n_sites=70, n_clones=10, dropout=0.2, doublet_share=0.04, seed=12
    )
    one = _call(truth.calls, nthreads=1)
    four = _call(truth.calls, nthreads=4)
    np.testing.assert_array_equal(one.codes, four.codes)
    np.testing.assert_array_equal(one.probability, four.probability)
    np.testing.assert_array_equal(one.ambiguous, four.ambiguous)
    np.testing.assert_array_equal(one.genotypes, four.genotypes)
    np.testing.assert_array_equal(one.dropout, four.dropout)
    np.testing.assert_array_equal(one.error, four.error)
    assert one.loglik == four.loglik
    assert one.bound_checks == four.bound_checks


def test_cells_without_a_mutant_call_and_a_truth_without_clones_take_none() -> None:
    # Every cell reads wild type, bar a few errors: no clone is called.
    rng = np.random.default_rng(13)
    calls = np.where(rng.random((400, 20)) < 0.01, 1, 0).astype(np.int8)
    called = _call(calls)
    assert len(called.sizes) == 0
    assert (called.codes == -1).all()
    assert np.isnan(called.probability).all()


@pytest.mark.parametrize(
    ("arguments", "error", "words"),
    [
        ({"method": "nearest"}, ValueError, "method must be one of likelihood, exact"),
        ({"method": 1}, ValueError, "method must be one of"),
        (
            {"method": "exact", "dropout_probability": 0.1, "min_size": 2},
            ValueError,
            "method='exact' uses no dropout_probability, min_size",
        ),
        (
            {"method": "likelihood", "dropout_probability": 1.0},
            ValueError,
            "up to but not including 1",
        ),
        (
            {"method": "likelihood", "error_probability": -0.1},
            ValueError,
            "from 0 up to",
        ),
        (
            {"method": "likelihood", "error_probability": float("nan")},
            ValueError,
            "from 0 up to",
        ),
        (
            {"method": "likelihood", "dropout_probability": True},
            TypeError,
            "must be a number",
        ),
        (
            {"method": "likelihood", "dropout_probability": "0.1"},
            TypeError,
            "must be a number",
        ),
        (
            {
                "method": "likelihood",
                "dropout_probability": 0.6,
                "error_probability": 0.4,
            },
            ValueError,
            "must sum to less than 1",
        ),
        (
            {"method": "likelihood", "min_size": 0},
            ValueError,
            "min_size must be at least 1",
        ),
        (
            {"method": "likelihood", "min_size": 2.0},
            TypeError,
            "min_size must be an integer",
        ),
        (
            {"method": "likelihood", "min_posterior": 1.5},
            ValueError,
            "min_posterior must lie from 0 to 1",
        ),
        (
            {"method": "likelihood", "min_posterior": False},
            TypeError,
            "min_posterior must be a number",
        ),
    ],
)
def test_genotype_clone_rules_refuse_what_the_method_cannot_use(
    arguments, error, words
) -> None:
    arguments = dict(arguments)
    method = arguments.pop("method")
    with pytest.raises(error, match=words):
        genotype_clone_rules(method, **arguments)


def test_genotype_clone_rules_resolve_the_likelihood_defaults() -> None:
    rules = genotype_clone_rules("likelihood")
    assert (rules.min_size, rules.min_posterior) == (2, 0.6)
    assert (rules.dropout_probability, rules.error_probability) == (None, None)
    rules = genotype_clone_rules(
        "likelihood", dropout_probability=0, min_size=np.int64(3)
    )
    assert (rules.dropout_probability, rules.min_size) == (0.0, 3)
    assert genotype_clone_rules("exact").min_size is None


def _store_of_truth(folder: Path, calls: np.ndarray, donors: np.ndarray | None = None):
    ids = [f"cell{cell:05d}-1" for cell in range(len(calls))]
    store = write_cells_store(folder / "store.zarr", ids)
    if donors is not None:
        store.cells.insert("donor", donors.astype(object))
    paths = write_call_tables(calls, ids, folder / "got")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(paths, cell_selection=every_cell(store))
    genotypes = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    return store, ids, genotypes


def test_likelihood_clones_save_probabilities_rates_and_diagnostics(tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group

    truth = dropout_truth(n_cells=1_200, n_sites=24, n_clones=6, dropout=0.15, seed=14)
    store, ids, genotypes = _store_of_truth(tmp_path, truth.calls)
    clones = store.variants.genotype_clones(genotypes)
    loaded = store.clones.load_clones(clones)
    cells = loaded.cells.loc[ids]
    # The saved clones are those of the packed calls, cell for cell.
    direct = _call(truth.calls)
    saved_codes = np.asarray(artifact_group(store.zw, clones)["codes"][:])
    np.testing.assert_array_equal(saved_codes >= 0, direct.codes >= 0)
    assert adjusted_rand_score(direct.codes, saved_codes) == 1.0
    np.testing.assert_array_equal(cells["ambiguous"].to_numpy(), direct.ambiguous)
    np.testing.assert_allclose(
        cells["clone_probability"].to_numpy(), direct.probability, equal_nan=True
    )
    assert loaded.clones["n_cells"].tolist() == direct.sizes.tolist()
    record = store.artifacts.inspect(clones)
    assert record.parameters["method"] == "likelihood"
    assert (record.parameters["min_size"], record.parameters["min_posterior"]) == (
        2,
        0.6,
    )
    assert record.parameters["dropout_prior_calls"] == 20.0
    assert "dropout_probability" not in record.parameters
    assert "similarity" not in record.parameters
    diagnostics = record.diagnostics
    assert diagnostics["dropoutProbability"] == pytest.approx(
        direct.carried_wild / direct.carried_calls
    )
    assert diagnostics["errorProbability"] == pytest.approx(
        direct.absent_mutant / direct.absent_calls
    )
    assert diagnostics["cellsInSmallClones"] == direct.cells_in_small_clones
    assert diagnostics["ambiguousCells"] == int(direct.ambiguous.sum())
    assert diagnostics["mutantCells"] == int((truth.calls == 1).any(axis=1).sum())
    assert diagnostics["sites"] == 24
    group = store.artifacts.load(clones)
    assert group["site_labels"][:].tolist() == [f"T{site + 1}" for site in range(24)]
    np.testing.assert_array_equal(
        group["dropout_probability"][:], direct.dropout[None, :]
    )
    np.testing.assert_array_equal(group["error_probability"][:], direct.error[None, :])
    assert dict(group.attrs)["n_clones"] == len(direct.sizes)
    assert "n_nodes" not in dict(group.attrs)
    # A held rate is recorded, saved at every site and is the diagnostic.
    held = store.variants.genotype_clones(genotypes, dropout_probability=0.2)
    assert store.artifacts.inspect(held).parameters["dropout_probability"] == 0.2
    assert store.artifacts.inspect(held).diagnostics["dropoutProbability"] == 0.2
    np.testing.assert_array_equal(
        store.artifacts.load(held)["dropout_probability"][:], np.full((1, 24), 0.2)
    )
    # The exact rule records None for the likelihood's diagnostics.
    exact = store.artifacts.inspect(
        store.variants.genotype_clones(genotypes, method="exact")
    )
    assert exact.diagnostics["dropoutProbability"] is None
    assert exact.diagnostics["cellsInSmallClones"] is None


def test_within_groups_are_fitted_apart_with_rates_of_their_own(tmp_path) -> None:
    # Two donors of one panel, the second read with three times the
    # first's dropout: each is fitted alone, so each keeps its own rates and
    # clones, numbered by size over both.
    first = dropout_truth(n_cells=800, n_sites=20, n_clones=4, dropout=0.08, seed=15)
    second = dropout_truth(n_cells=600, n_sites=20, n_clones=4, dropout=0.25, seed=16)
    calls = np.vstack([first.calls, second.calls])
    donors = np.array(["d1"] * 800 + ["d2"] * 600)
    store, ids, genotypes = _store_of_truth(tmp_path, calls, donors)
    clones = store.variants.genotype_clones(genotypes, within="donor")
    cells = store.clones.load_clones(clones).cells.loc[ids]
    alone = [_call(first.calls), _call(second.calls)]
    labels = cells["clone"].astype(object).where(cells["clone"].notna(), "").to_numpy()
    # Each donor's cells are partitioned as its fit alone partitions them,
    # and no clone holds cells of both.
    assert adjusted_rand_score(alone[0].codes, labels[:800]) == 1.0
    assert adjusted_rand_score(alone[1].codes, labels[800:]) == 1.0
    assert not (set(labels[:800]) - {""}) & (set(labels[800:]) - {""})
    saved = store.artifacts.load(clones)["dropout_probability"][:]
    assert saved.shape == (2, 20)
    np.testing.assert_array_equal(saved[0], alone[0].dropout)
    np.testing.assert_array_equal(saved[1], alone[1].dropout)
    assert saved[1].mean() > 2 * saved[0].mean()
    pooled = store.artifacts.inspect(clones).diagnostics["dropoutProbability"]
    expected = (alone[0].carried_wild + alone[1].carried_wild) / (
        alone[0].carried_calls + alone[1].carried_calls
    )
    assert pooled == pytest.approx(expected)
