"""Fate coupling and fate bias against CoSpar 0.5.0 on a synthetic experiment.

``cospar_0_5_0_golden.json`` holds the experiment, 990 cells at day 2 and day
4 with 120 clones of known fate profiles, and what CoSpar computed from it on
Modal with the script that pull request 04 keeps beside its scratch work:
``fate_coupling`` of the day-4 cells by four methods, the permuted couplings
of ``pvalue_for_fate_coupling``, ``fate_map`` on the clonal transition map,
and ``clonal_fate_bias`` per fate with statsmodels' Benjamini-Hochberg
adjustment. CoSpar never runs in the tests.
"""

import json
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cytearc.clones import COUPLING_METHODS
from tests.clone_helpers import open_clone_store
from tests.storage_helpers import write_count_store

GOLDEN = json.loads(
    (Path(__file__).parent / "cospar_0_5_0_golden.json").read_text(encoding="utf-8")
)
CELLS = {name: np.asarray(values) for name, values in GOLDEN["cells"].items()}
FATES = GOLDEN["coupling"]["fates"]
# CoSpar adds this to each fate's mean cells per clone in its "Weinreb"
# normalized covariance.
MEAN_PSEUDOCOUNT = 1e-4


@pytest.fixture(scope="module")
def experiment(tmp_path_factory) -> SimpleNamespace:
    path = tmp_path_factory.mktemp("cospar_golden") / "store.zarr"
    n_cells = len(CELLS["ids"])
    counts = np.random.default_rng(0).poisson(2.0, size=(n_cells, 4)) + 1
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=CELLS["ids"])
    store = open_clone_store(path)
    store.cells.insert("cell_type", CELLS["cell_type"].astype(object))
    store.cells.insert("clone", CELLS["clone"].astype(object))
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    store.cells.insert("early", CELLS["time"] == "2")
    store.cells.insert("late", CELLS["time"] == "4")
    return SimpleNamespace(
        store=store,
        clones=store.clones.from_column(
            "clone", cell_selection=store.snapshot_cell_selection("every_cell")
        ),
        early=store.snapshot_cell_selection("early"),
        late=store.snapshot_cell_selection("late"),
    )


def _matrix(pairs, column: str) -> np.ndarray:
    """A pairs table as a matrix in CoSpar's order of the fates."""
    at = {name: position for position, name in enumerate(FATES)}
    found = np.full((len(FATES), len(FATES)), np.nan)
    for first, second, value in pairs[["group_1", "group_2", column]].itertuples(
        index=False
    ):
        found[at[first], at[second]] = found[at[second], at[first]] = value
    return found


def _mean_cells_per_clone() -> np.ndarray:
    """Each fate's late cells with a clone over the clones with a late cell."""
    late = (CELLS["time"] == "4") & (CELLS["clone"] != "")
    n_clones = len(set(CELLS["clone"][late].tolist()))
    return (
        np.asarray(
            [np.count_nonzero(late & (CELLS["cell_type"] == fate)) for fate in FATES]
        )
        / n_clones
    )


@pytest.mark.parametrize("method", COUPLING_METHODS)
def test_coupling_equals_cospars_fate_coupling(experiment, method) -> None:
    result = experiment.store.clones.coupling(
        experiment.clones,
        group_by="cell_type",
        cell_selection=experiment.late,
        method=method,
        permutations=0,
    )
    assert sorted(result.group_sizes["group"]) == FATES
    found = _matrix(result.pairs, "coupling")
    expected = np.asarray(GOLDEN["coupling"][method])
    off_diagonal = ~np.eye(len(FATES), dtype=bool)
    if method == "normalized_covariance":
        # Without CoSpar's pseudocount, the means of the two fates divide the
        # covariance exactly.
        means = _mean_cells_per_clone()
        np.testing.assert_allclose(means, GOLDEN["coupling"]["cells_per_clone_mean"])
        shifted = means + MEAN_PSEUDOCOUNT
        expected = expected * np.outer(shifted, shifted) / np.outer(means, means)
    # CoSpar's other pseudocounts, 1e-10 beside each sum it divides by, move
    # its values by less than 1e-8.
    np.testing.assert_allclose(
        found[off_diagonal], expected[off_diagonal], rtol=1e-8, atol=1e-12
    )


@pytest.mark.parametrize("method", ["abundance_cosine", "share_cosine"])
def test_the_null_agrees_with_cospars_permuted_couplings(experiment, method) -> None:
    null = GOLDEN["null"]
    assert null["fates"] == FATES
    result = experiment.store.clones.coupling(
        experiment.clones,
        group_by="cell_type",
        cell_selection=experiment.late,
        method=method,
        permutations=null["permutations"],
    )
    mean, spread = _matrix(result.pairs, "null_mean"), _matrix(result.pairs, "null_sd")
    expected_mean = np.asarray(null[method]["mean"])
    expected_spread = np.asarray(null[method]["sd"])
    off_diagonal = ~np.eye(len(FATES), dtype=bool)
    # Two estimates of one null from 2,000 permutations each: their means
    # differ by sqrt(sd1^2 + sd2^2) / sqrt(2000) at one standard error, and
    # their standard deviations by about 2.2% at one.
    error = np.sqrt((spread**2 + expected_spread**2) / null["permutations"])
    assert (
        np.abs(mean - expected_mean)[off_diagonal] < 4.5 * error[off_diagonal]
    ).all()
    np.testing.assert_allclose(
        spread[off_diagonal], expected_spread[off_diagonal], rtol=0.12
    )
    # Pairs that CoSpar's null places apart stay apart in CyteArc's.
    assert sqrt(null["permutations"]) > 40


def test_fate_fractions_equal_cospars_clonal_fate_map(experiment) -> None:
    bias = experiment.store.clones.load_fate_bias(
        experiment.store.clones.fate_bias(
            experiment.clones,
            group_by="cell_type",
            early=experiment.early,
            late=experiment.late,
        )
    )
    mapped = GOLDEN["fate_map"]
    assert mapped["fates"] == FATES
    cells = bias.cells.loc[mapped["early_cells"]]
    found = cells[FATES].to_numpy(dtype=np.float64)
    expected = np.asarray(mapped["fractions"])
    # CoSpar gives an early cell whose clone has no late cell 0 in every
    # fate, where CyteArc gives it no fractions.
    without = cells["late_cells"].to_numpy() == 0
    assert 0 < without.sum() < len(without)
    assert np.isnan(found[without]).all() and (expected[without] == 0).all()
    # CoSpar divides by the clone's late cells plus 1e-10.
    np.testing.assert_allclose(found[~without], expected[~without], rtol=1e-9)


def test_clone_tests_equal_cospars_clonal_fate_bias(experiment) -> None:
    bias = experiment.store.clones.load_fate_bias(
        experiment.store.clones.fate_bias(
            experiment.clones,
            group_by="cell_type",
            early=experiment.early,
            late=experiment.late,
        )
    )
    table = bias.clone_fates
    for fate, tests in GOLDEN["clonal_fate_bias"].items():
        rows = table[table["fate"] == fate].set_index("clone").loc[tests["clones"]]
        assert rows["late_cells"].tolist() == tests["clone_size"]
        np.testing.assert_allclose(rows["fraction"], tests["fraction"], rtol=1e-12)
        # CoSpar's alternative="less" is the upper tail, P[X >= x].
        np.testing.assert_allclose(rows["p_value"], tests["p_value"], rtol=1e-12)
        np.testing.assert_allclose(rows["q_value"], tests["bh_q_value"], rtol=1e-12)
    # Every clone with a late cell is tested in every fate.
    assert len(table) == len(GOLDEN["clonal_fate_bias"][FATES[0]]["clones"]) * len(
        FATES
    )
