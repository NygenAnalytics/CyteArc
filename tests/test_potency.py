"""The potency score: signaling entropy rate and CCAT on a protein network.

Domain tests check ``cytearc.trajectory.potency`` against values derived by
hand, an explicit stationary distribution and entropy rate, NumPy's
correlation, a dense solve of the smoothing and invariants that a wrong
normalization or bookkeeping would break. Operation tests run
``ds.scores.potency`` on the 1K PBMC fixture with a network of its own genes,
recorded in place of the STRING network, and check it against counts read
past the operation's stream.
"""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from cytearc import datasets
from cytearc.datastore._operations import trajectory as trajectory_operations
from cytearc.datastore.datastore import DataStore
from cytearc.metadata.selection import cell_selection_indices
from cytearc.readers import H5adReader
from cytearc.storage.artifacts import ArtifactRef, new_artifact_id
from cytearc.storage.virtual_arrays import read_selection
from cytearc.trajectory import potency
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import H5adToZarr
from scripts.build_string_networks import write_npz
from tests.network_helpers import NETWORK_SKIP, hub_or_skip

LOG2_1_1 = math.log2(1.1)
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2


def path_network(n: int) -> potency.MatchedNetwork:
    """A path of ``n`` genes whose features are their positions."""
    network = potency.ProteinNetwork(
        ensembl_ids=np.array([f"ENSG{i:011d}" for i in range(n)]),
        symbols=np.array([f"G{i}" for i in range(n)]),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)]),
        metadata={},
    )
    return potency.match_network(network, network.ensembl_ids, "ensembl")


def random_network(
    rng: np.random.Generator, n: int, extra: int
) -> potency.MatchedNetwork:
    """A connected network of ``n`` genes: a random tree and ``extra`` more edges."""
    parents = np.array([rng.integers(0, child) for child in range(1, n)])
    pairs = {(int(p), int(c)) for p, c in zip(parents, range(1, n), strict=True)}
    while len(pairs) < n - 1 + extra:
        a, b = sorted(int(v) for v in rng.choice(n, 2, replace=False))
        pairs.add((a, b))
    edges = np.array(sorted(pairs)).T
    network = potency.ProteinNetwork(
        ensembl_ids=np.array([f"ENSG{i:011d}" for i in range(n)]),
        symbols=np.array([f"G{i}" for i in range(n)]),
        edges=edges,
        metadata={},
    )
    return potency.match_network(network, network.ensembl_ids, "ensembl")


def log_values(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """x = log2(10,000 counts / total + 1.1), written out apart from the kernel."""
    return np.log2(10_000.0 * counts / totals[:, None] + 1.1)


def explicit_entropy_rate(adjacency: np.ndarray, x: np.ndarray) -> float:
    """The walk's entropy rate from its transition matrix and stationary vector."""
    walk = adjacency * x[None, :]
    walk /= walk.sum(axis=1, keepdims=True)
    values, vectors = np.linalg.eig(walk.T)
    stationary = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    stationary /= stationary.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(walk > 0, walk * np.log(walk), 0.0)
    return float(-(stationary[:, None] * terms).sum())


# ---------------------------------------------------------------------------
# The measures


def test_one_detected_gene_on_a_path_of_three_genes_is_worked_by_hand():
    network = path_network(3)
    # 10,000 * 77 / 100,000 + 1.1 = 8.8 = 8 * 1.1, so x0 = 3 + b with b = log2(1.1).
    block = np.array([[77, 0, 0]])
    entropy, correlation, detected = potency.network_scores(
        block, np.array([100_000.0]), network
    )
    b = LOG2_1_1
    x0 = 3 + b
    # (Ax) = (b, x0 + b, b) and the flux x_i (Ax)_i = (x0 b, b (x0 + b), b^2).
    flux = np.array([x0 * b, b * (x0 + b), b * b])
    logs = np.array([math.log(b / x0), math.log((x0 + b) / b), 0.0])
    rate = float((flux * logs).sum() / flux.sum())
    # The path's largest eigenvalue is sqrt(2).
    assert entropy[0] == pytest.approx(rate / math.log(math.sqrt(2)), rel=1e-12)
    # x = (x0, b, b) against degrees (1, 2, 1): both centred, x0 - mean = 2,
    # the others -1, and the degrees (-1/3, 2/3, -1/3).
    xc = np.array([2.0, -1.0, -1.0])
    kc = np.array([-1.0, 2.0, -1.0]) / 3
    assert correlation[0] == pytest.approx(
        (xc @ kc) / math.sqrt((xc @ xc) * (kc @ kc)), rel=1e-12
    )
    assert detected.tolist() == [1]


def test_uniform_expression_on_a_path_of_four_genes_is_worked_by_hand():
    network = path_network(4)
    assert math.exp(network.log_lambda) == pytest.approx(GOLDEN_RATIO, rel=1e-12)
    entropy, correlation, detected = potency.network_scores(
        np.zeros((1, 4), dtype=np.uint16), np.array([1_000.0]), network
    )
    # Uniform x walks the simple random walk: pi_i = k_i / 2m = (1, 2, 2, 1) / 6
    # and the rate is sum_i pi_i log k_i = (2/3) log 2, over log of the golden ratio.
    assert entropy[0] == pytest.approx((2 / 3) * math.log(2) / math.log(GOLDEN_RATIO))
    assert math.isnan(correlation[0])
    assert detected.tolist() == [0]


@pytest.mark.parametrize("seed", range(5))
def test_the_rate_is_the_explicit_entropy_rate_over_the_largest_rate(seed):
    rng = np.random.default_rng(seed)
    network = random_network(rng, int(rng.integers(6, 25)), int(rng.integers(0, 15)))
    n = network.genes_used
    counts = rng.integers(0, 40, size=(4, n)) * (rng.random((4, n)) < 0.6)
    totals = counts.sum(axis=1) + rng.integers(1, 500, size=4)
    entropy, correlation, _ = potency.network_scores(
        counts, totals.astype(float), network
    )

    dense = network.adjacency.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    largest = eigenvalues[-1]
    # The walk p_ij = A_ij v_j / (lambda v_i) on the Perron vector v has the
    # largest entropy rate, log lambda.
    perron = np.abs(eigenvectors[:, -1])
    assert explicit_entropy_rate(dense, perron) == pytest.approx(math.log(largest))
    x = log_values(counts, totals.astype(float))
    for row in range(4):
        assert entropy[row] == pytest.approx(
            explicit_entropy_rate(dense, x[row]) / math.log(largest), rel=1e-10
        )
        assert correlation[row] == pytest.approx(
            np.corrcoef(x[row], dense.sum(axis=1))[0, 1], rel=1e-10
        )
        assert 0 < entropy[row] <= 1 + 1e-12


def test_the_kernel_matches_a_dense_reference_on_sparse_counts():
    rng = np.random.default_rng(7)
    network = random_network(rng, 300, 900)
    counts = rng.poisson(0.4, size=(50, 300)).astype(np.uint16)
    counts[3] = 0
    counts[3, 5] = 9
    totals = counts.sum(axis=1).astype(float) + 1_000.0
    entropy, correlation, detected = potency.network_scores(counts, totals, network)

    x = log_values(counts.astype(float), totals)
    dense = network.adjacency.toarray()
    ax = x @ dense
    flux = x * ax
    expected_entropy = (flux * np.log(ax / x)).sum(axis=1) / flux.sum(axis=1)
    expected_entropy /= math.log(np.linalg.eigvalsh(dense)[-1])
    degree = dense.sum(axis=1)
    expected_correlation = [np.corrcoef(row, degree)[0, 1] for row in x]
    np.testing.assert_allclose(entropy, expected_entropy, rtol=1e-12)
    np.testing.assert_allclose(correlation, expected_correlation, rtol=1e-10)
    np.testing.assert_array_equal(detected, (counts > 0).sum(axis=1))


def test_scores_are_invariant_to_gene_order_cell_order_and_depth():
    rng = np.random.default_rng(3)
    network = random_network(rng, 40, 60)
    counts = rng.integers(0, 30, size=(6, 40))
    totals = counts.sum(axis=1).astype(float) + 200.0
    entropy, correlation, _ = potency.network_scores(counts, totals, network)

    # Renumbering the network's genes, with the counts' columns, changes nothing.
    order = rng.permutation(40)
    position = np.empty(40, dtype=np.int64)
    position[order] = np.arange(40)
    adjacency = network.adjacency.tocoo()
    renumbered = potency.ProteinNetwork(
        ensembl_ids=np.array([f"ENSG{i:011d}" for i in range(40)])[order],
        symbols=np.array([f"G{i}" for i in range(40)])[order],
        edges=np.array(
            sorted(
                {
                    tuple(sorted((int(position[a]), int(position[b]))))
                    for a, b in zip(adjacency.row, adjacency.col, strict=True)
                }
            )
        ).T,
        metadata={},
    )
    matched = potency.match_network(
        renumbered, np.array([f"ENSG{i:011d}" for i in range(40)]), "ensembl"
    )
    shuffled = potency.network_scores(counts[:, matched.features], totals, matched)
    np.testing.assert_allclose(shuffled[0], entropy, rtol=1e-12)
    np.testing.assert_allclose(shuffled[1], correlation, rtol=1e-12)

    # Reordering cells reorders their scores.
    cells = rng.permutation(6)
    reordered = potency.network_scores(counts[cells], totals[cells], network)
    np.testing.assert_allclose(reordered[0], entropy[cells], rtol=1e-12)

    # Three times the counts of a cell over three times its total is the same cell.
    deeper = potency.network_scores(counts * 3, totals * 3, network)
    np.testing.assert_allclose(deeper[0], entropy, rtol=1e-12)
    np.testing.assert_allclose(deeper[1], correlation, rtol=1e-12)


def test_scores_refuse_blocks_that_do_not_fit_the_network():
    network = path_network(4)
    with pytest.raises(ValueError, match="counts of the network's 4 genes"):
        potency.network_scores(np.zeros((2, 3)), np.ones(2), network)
    with pytest.raises(ValueError, match="one total for each row"):
        potency.network_scores(np.zeros((2, 4)), np.ones(3), network)
    with pytest.raises(ValueError, match="total count must be positive"):
        potency.network_scores(np.ones((2, 4)), np.array([4.0, 0.0]), network)


# ---------------------------------------------------------------------------
# Networks


def write_network(path: Path, ids, symbols, edges, metadata="{}") -> Path:
    write_npz(
        path,
        {
            "ensembl_ids": np.asarray(ids),
            "symbols": np.asarray(symbols),
            "edges": np.asarray(edges, dtype=np.uint16),
            "scores": np.full(np.asarray(edges).shape[1], 900, dtype=np.uint16),
            "metadata": np.array(metadata),
        },
    )
    return path


def test_read_network_reads_the_synthetic_network():
    network = potency.read_network(
        Path(__file__).parent / "data" / "networks" / "synthetic_network.npz"
    )
    assert network.n_genes == 30
    assert network.edges.shape == (2, 40) and network.edges.dtype == np.int64
    assert network.metadata["source"] == "synthetic"


@pytest.mark.parametrize(
    ("edges", "metadata", "message"),
    [
        ([[0, 1], [1, 3]], "{}", "must name genes of the network"),
        ([[1, 0], [0, 2]], "{}", "lower gene position first"),
        ([[0, 1], [1, 2]], "not json", "metadata must be JSON text"),
        ([[0, 1], [1, 2]], "[1, 2]", "metadata must be a JSON object"),
    ],
)
def test_read_network_refuses_malformed_archives(tmp_path, edges, metadata, message):
    path = write_network(
        tmp_path / "bad.npz", ["E1", "E2", "E3"], ["A", "B", "C"], edges, metadata
    )
    with pytest.raises(ValueError, match=message):
        potency.read_network(path)


def test_read_network_refuses_malformed_genes_and_edges(tmp_path):
    def archive(ids, symbols, edges):
        path = tmp_path / "bad.npz"
        write_npz(
            path,
            {
                "ensembl_ids": np.asarray(ids),
                "symbols": np.asarray(symbols),
                "edges": np.asarray(edges),
                "metadata": np.array("{}"),
            },
        )
        return path

    two = (["E1", "E2"], ["A", "B"])
    with pytest.raises(ValueError, match="at least two genes"):
        potency.read_network(archive(["E1"], ["A"], [[0], [0]]))
    with pytest.raises(ValueError, match="one Ensembl ID and one symbol each"):
        potency.read_network(archive(["E1", "E2"], ["A"], [[0], [1]]))
    with pytest.raises(ValueError, match="two rows of gene positions"):
        potency.read_network(archive(*two, [[0], [1], [1]]))
    with pytest.raises(ValueError, match="two rows of gene positions"):
        potency.read_network(archive(*two, np.zeros((2, 0), dtype=np.int64)))
    with pytest.raises(ValueError, match="integer gene positions"):
        potency.read_network(archive(*two, [[0.0], [1.0]]))


def test_read_network_refuses_an_archive_without_edges(tmp_path):
    write_npz(tmp_path / "bad.npz", {"ensembl_ids": np.array(["E1", "E2"])})
    with pytest.raises(ValueError, match="lacks the arrays"):
        potency.read_network(tmp_path / "bad.npz")


def test_match_network_keeps_the_largest_connected_part_of_the_genes_matched():
    network = potency.ProteinNetwork(
        ensembl_ids=np.array(["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5", "ENSG6"]),
        symbols=np.array(["A", "B", "C", "D", "E", "F"]),
        # Two components among the genes, and an edge to an unmatched gene.
        edges=np.array([[0, 1, 3, 4], [1, 2, 4, 5]]),
        metadata={},
    )
    # ENSG2 twice, the first wins; versions are ignored; ENSG6 is not measured.
    features = np.array(["x", "ENSG2.4", "ENSG1", "ENSG3", "ENSG2", "ENSG4", "ENSG5"])
    matched = potency.match_network(network, features, "ensembl")
    assert matched.genes_matched == 5
    assert matched.features.tolist() == [2, 1, 3]
    assert matched.edges_used == 2
    assert matched.adjacency.toarray().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    by_symbol = potency.match_network(network, np.array(["a", "D", "E"]), "symbol")
    assert by_symbol.features.tolist() == [1, 2]
    with pytest.raises(ValueError, match="share no edge"):
        potency.match_network(network, np.array(["A", "C"]), "symbol")
    with pytest.raises(ValueError, match="'ensembl' or 'symbol'"):
        potency.match_network(network, features, "name")  # type: ignore[arg-type]


def test_features_outside_the_network_count_only_toward_the_total():
    rng = np.random.default_rng(5)
    ids = np.array([f"ENSG{i:011d}" for i in range(30)])
    ring = np.stack([np.arange(30), (np.arange(30) + 1) % 30])
    chords = np.stack([np.arange(0, 30, 3), (np.arange(0, 30, 3) + 7) % 30])
    network = potency.ProteinNetwork(
        ensembl_ids=ids,
        symbols=np.array([f"G{i}" for i in range(30)]),
        edges=np.sort(np.c_[ring, chords], axis=0),
        metadata={},
    )
    counts = rng.integers(0, 20, size=(5, 30))
    totals = counts.sum(axis=1) + 300.0
    alone = potency.match_network(network, ids, "ensembl")
    expected = potency.network_scores(counts, totals, alone)

    # The same genes among 20 features that the network lacks, in another order.
    order = rng.permutation(50)
    keys = np.r_[ids, [f"OTHER{i}" for i in range(20)]][order]
    wider = np.c_[counts, rng.integers(0, 50, size=(5, 20))][:, order]
    matched = potency.match_network(network, keys, "ensembl")
    np.testing.assert_array_equal(order[matched.features], np.arange(30))
    found = potency.network_scores(wider[:, matched.features], totals, matched)
    for value, reference in zip(found, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-12)


def test_largest_eigenvalue_agrees_between_the_dense_and_sparse_solvers():
    rng = np.random.default_rng(11)
    network = random_network(rng, 600, 1_200)
    dense = float(np.linalg.eigvalsh(network.adjacency.toarray())[-1])
    assert potency.largest_eigenvalue(network.adjacency) == pytest.approx(
        dense, rel=1e-10
    )
    assert math.exp(network.log_lambda) == pytest.approx(dense, rel=1e-10)


# ---------------------------------------------------------------------------
# SCENT's reference outputs

SCENT_REFERENCE = Path(__file__).parent / "data" / "potency" / "scent_reference"


def test_both_measures_match_scents_values_on_pancreas_cells():
    # SCENT scored these 64 cells on 5,778 genes of the mouse network; its
    # values come from scripts/export_scent_potency_golden.R.
    with np.load(SCENT_REFERENCE.with_suffix(".npz"), allow_pickle=False) as stored:
        reference = {name: stored[name] for name in stored.files}
    symbols = reference["symbols"]
    network = potency.ProteinNetwork(
        ensembl_ids=np.array([f"ENSMUSG{i:011d}" for i in range(len(symbols))]),
        symbols=symbols,
        edges=reference["edges"].astype(np.int64),
        metadata={},
    )
    matched = potency.match_network(network, symbols, "symbol")
    assert matched.genes_used == len(symbols) == 5_778
    assert matched.edges_used == reference["edges"].shape[1]
    np.testing.assert_array_equal(matched.features, np.arange(len(symbols)))

    entropy, correlation, detected = potency.network_scores(
        reference["counts"], reference["totals"], matched
    )

    np.testing.assert_allclose(
        entropy, reference["signaling_entropy"], rtol=1e-10, atol=0
    )
    np.testing.assert_allclose(correlation, reference["ccat"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(detected, (reference["counts"] > 0).sum(axis=1))


def test_scents_reference_records_how_it_was_made():
    provenance = json.loads(SCENT_REFERENCE.with_suffix(".json").read_text())
    with np.load(SCENT_REFERENCE.with_suffix(".npz"), allow_pickle=False) as stored:
        counts_shape = stored["counts"].shape
        edges_shape = stored["edges"].shape
        cells = stored["cell_ids"]

    version, commit, r_version = provenance["tool"]
    assert version == "SCENT 1.1"
    assert len(commit.removeprefix("SCENT commit ")) == 40
    assert r_version.startswith("R version ")
    assert (
        f"log2({potency.SCALE_TOTAL:,.0f} * counts / total + {potency.PSEUDOCOUNT})"
        in provenance["values"]
    )
    prepared = provenance["prepared"]
    mouse = datasets.STRING_NETWORKS["mouse"]
    assert prepared["networkSha256"] == mouse.sha256
    assert prepared["network"]["stringVersion"] == mouse.string_version
    assert prepared["network"]["droppedChannels"] == ["database"]
    assert (prepared["cells"], prepared["genes"]) == counts_shape == (len(cells), 5_778)
    assert prepared["edges"] == edges_shape[1]


# ---------------------------------------------------------------------------
# Smoothing and ranks


def two_component_graph() -> sparse.csr_matrix:
    """A weighted symmetric graph of 12 cells in two components of six."""
    rng = np.random.default_rng(5)
    weights = rng.random((12, 12)) * (rng.random((12, 12)) < 0.5) + 0.0
    weights = np.triu(weights, k=1)
    # A path in each component keeps it connected.
    for first in (0, 6):
        for cell in range(first, first + 5):
            weights[cell, cell + 1] = max(weights[cell, cell + 1], 0.2)
    weights[:6, 6:] = 0.0
    return sparse.csr_matrix(weights + weights.T)


def test_smoothing_reaches_the_dense_solve():
    graph = two_component_graph()
    values = np.random.default_rng(1).normal(size=12)
    alpha = 0.9
    walk = graph.toarray() / graph.toarray().sum(axis=1, keepdims=True)
    expected = (1 - alpha) * np.linalg.solve(np.eye(12) - alpha * walk, values)
    smoothed = potency.smooth_on_graph(values, graph, alpha)
    np.testing.assert_allclose(smoothed, expected, atol=1e-10)
    assert potency.smoothing_steps(alpha) == 263


def test_smoothing_keeps_constants_components_and_lonely_cells():
    graph = two_component_graph()
    assert connected_components(graph)[0] == 2
    np.testing.assert_allclose(
        potency.smooth_on_graph(np.full(12, 3.5), graph, 0.95), 3.5, atol=1e-12
    )
    values = np.r_[np.zeros(6), np.arange(6.0)]
    smoothed = potency.smooth_on_graph(values, graph, 0.9)
    np.testing.assert_array_equal(smoothed[:6], 0.0)
    # A cell without edges keeps its value.
    lonely = sparse.csr_matrix(graph.shape)
    np.testing.assert_allclose(potency.smooth_on_graph(values, lonely, 0.9), values)


def test_smoothing_refuses_its_range_and_other_graphs():
    graph = two_component_graph()
    for smoothing in (0.0, 0.995, 1.0, -0.1):
        with pytest.raises(ValueError, match="above 0 and at most 0.99"):
            potency.smooth_on_graph(np.zeros(12), graph, smoothing)
    with pytest.raises(ValueError, match="A graph of 12 cells cannot smooth 5 values"):
        potency.smooth_on_graph(np.zeros(5), graph, 0.9)


def test_unit_ranks_share_ties_and_refuse_too_few_or_infinite_values():
    np.testing.assert_allclose(
        potency.unit_ranks(np.array([3.0, 1.0, 3.0, 2.0])), [5 / 6, 0, 5 / 6, 1 / 3]
    )
    with pytest.raises(ValueError, match="at least two values"):
        potency.unit_ranks(np.array([1.0]))
    with pytest.raises(ValueError, match="finite values"):
        potency.unit_ranks(np.array([1.0, np.nan]))


# ---------------------------------------------------------------------------
# Species


def test_species_come_from_ensembl_ids_or_must_be_named():
    species = trajectory_operations._potency_species
    networks = {"human": None, "mouse": None}
    assert species(np.array(["ENSG00000000003", "x"]), None, networks) == (
        "human",
        "ensembl",
    )
    assert species(np.array(["ENSMUSG00000000001"]), None, networks) == (
        "mouse",
        "ensembl",
    )
    # A named species whose IDs the features do not carry matches by symbol.
    assert species(np.array(["Sox9", "Pdx1"]), "mouse", networks) == (
        "mouse",
        "symbol",
    )
    with pytest.raises(ValueError, match="not Ensembl gene IDs"):
        species(np.array(["Sox9"]), None, networks)
    with pytest.raises(ValueError, match="of human, mouse; pass species="):
        species(np.array(["ENSG00000000003", "ENSMUSG00000000001"]), None, networks)
    with pytest.raises(ValueError, match="of rat; pass species="):
        species(np.array(["ENSRNOG00000000001"]), None, networks)
    with pytest.raises(ValueError, match="No STRING network for species 'rat'"):
        species(np.array(["ENSG00000000003"]), "rat", networks)


# ---------------------------------------------------------------------------
# The operation on the 1K PBMC fixture


@pytest.fixture(scope="module")
def fixture_network(tmp_path_factory, datastore, auto_filter_cells):
    """A network of 300 of the fixture's genes detected in most of its cells."""
    cells = cell_selection_indices(datastore.zw, auto_filter_cells)
    counts = read_selection(datastore.RNA.z["counts"], np.sort(cells), slice(None))
    detected = (counts > 0).mean(axis=0)
    genes = np.sort(np.argsort(-detected, kind="stable")[:300])
    ids = np.asarray(datastore.RNA.feats.fetch_all("ids")).astype(str)
    names = np.asarray(datastore.RNA.feats.fetch_all("names")).astype(str)
    n = len(genes)
    pairs = {(i, i + 1) for i in range(n - 1)} | {
        tuple(sorted((i, (i + 17) % n))) for i in range(0, n, 2)
    }
    edges = np.array(sorted(pairs)).T
    path = write_network(
        tmp_path_factory.mktemp("network") / "human_9606.npz",
        ids[genes],
        names[genes],
        edges,
        json.dumps({"source": "synthetic"}),
    )
    return path, genes


@pytest.fixture
def recorded(monkeypatch, fixture_network):
    """Record the fixture network as the human network and accept its 300 genes."""
    path, genes = fixture_network
    data = path.read_bytes()
    entry = datasets.NetworkFile(
        species="human",
        taxid=9606,
        string_version="12.5",
        name="human_9606.npz",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    monkeypatch.setattr(datasets, "STRING_NETWORKS", {"human": entry})
    monkeypatch.setattr(trajectory_operations, "_MIN_NETWORK_GENES", 100)
    return path, genes, entry


def independent_scores(datastore, cell_selection, path):
    """SR and CCAT from counts read past the operation's stream, and full-row totals."""
    cells = cell_selection_indices(datastore.zw, cell_selection)
    network = potency.read_network(path)
    ids = np.asarray(datastore.RNA.feats.fetch_all("ids")).astype(str)
    matched = potency.match_network(network, ids, "ensembl")
    rows = read_selection(datastore.RNA.z["counts"], cells, slice(None)).astype(float)
    return potency.network_scores(rows[:, matched.features], rows.sum(axis=1), matched)


def test_potency_scores_the_cells_and_reuses_its_artifact(
    datastore, auto_filter_cells, recorded
):
    path, _, entry = recorded
    ref = datastore.scores.potency(auto_filter_cells, network=path)
    result = datastore.scores.load_potency(ref)

    entropy, correlation, _ = independent_scores(datastore, auto_filter_cells, path)
    np.testing.assert_allclose(result.signaling_entropy, entropy, rtol=1e-12)
    np.testing.assert_allclose(result.ccat, correlation, rtol=1e-12)
    # The two measures rank these cells differently, so the ranks name the default.
    assert not np.allclose(potency.unit_ranks(entropy), potency.unit_ranks(correlation))
    np.testing.assert_allclose(result.potency, potency.unit_ranks(correlation))
    np.testing.assert_allclose(result.differentiation, 1 - result.potency)
    assert (result.measure, result.species, result.graph, result.smoothing) == (
        "ccat",
        "human",
        None,
        None,
    )
    assert result.network == {
        "source": "STRING",
        "version": "12.5",
        "name": "human_9606.npz",
        "sha256": entry.sha256,
    }
    assert (result.genes_matched, result.genes_used, result.matched_by) == (
        300,
        300,
        "ensembl_id",
    )
    assert result.edges_used == 299 + 150
    status = datastore.artifacts.inspect(ref)
    assert set(status.parameters) == {"measure", "species", "string_network"}
    assert datastore.scores.potency(auto_filter_cells, network=path) == ref
    # CCAT is the default, so naming it finds the same artifact.
    assert (
        datastore.scores.potency(auto_filter_cells, measure="ccat", network=path) == ref
    )


def test_signaling_entropy_ranks_the_cells_when_asked_and_keeps_both_measures(
    datastore, auto_filter_cells, recorded
):
    path = recorded[0]
    ccat_ref = datastore.scores.potency(auto_filter_cells, network=path)
    entropy_ref = datastore.scores.potency(
        auto_filter_cells, measure="signaling_entropy", network=path
    )
    assert entropy_ref != ccat_ref
    by_entropy = datastore.scores.load_potency(entropy_ref)
    by_ccat = datastore.scores.load_potency(ccat_ref)
    np.testing.assert_array_equal(by_entropy.ccat, by_ccat.ccat)
    np.testing.assert_array_equal(
        by_entropy.signaling_entropy, by_ccat.signaling_entropy
    )
    np.testing.assert_allclose(
        by_entropy.potency, potency.unit_ranks(by_entropy.signaling_entropy)
    )
    assert by_entropy.measure == "signaling_entropy"


@pytest.mark.parametrize("rows", [1, 7, None])
def test_block_sizes_do_not_change_the_scores(
    datastore, auto_filter_cells, recorded, monkeypatch, rows
):
    path = recorded[0]
    streamed = trajectory_operations.feature_value_blocks

    def blocks_of(*args, **kwargs):
        most, blocks = streamed(*args, **kwargs)
        if rows is None:
            return most, blocks
        return rows, (
            block[start : start + rows]
            for block in blocks
            for start in range(0, block.shape[0], rows)
        )

    monkeypatch.setattr(trajectory_operations, "feature_value_blocks", blocks_of)
    ref = datastore.scores.potency(
        auto_filter_cells, network=path, invalidate_cache=True
    )
    entropy, correlation, _ = independent_scores(datastore, auto_filter_cells, path)
    result = datastore.scores.load_potency(ref)
    np.testing.assert_allclose(result.signaling_entropy, entropy, rtol=1e-12)
    np.testing.assert_allclose(result.ccat, correlation, rtol=1e-12)


@pytest.mark.parametrize("measure", potency.MEASURES)
def test_smoothing_over_the_graph_follows_the_dense_solve(
    datastore, auto_filter_cells, connectivity_graph, recorded, measure
):
    path = recorded[0]
    ref = datastore.scores.potency(
        auto_filter_cells,
        measure=measure,
        graph=connectivity_graph,
        smoothing=0.8,
        network=path,
    )
    result = datastore.scores.load_potency(ref)
    # The fuzzy union of the directed edges, as the operation loads the graph.
    # The graph stores float32 weights; normalizing them in float64, as the
    # operation does, keeps the solve exact enough to rank CCAT's close values.
    graph = datastore.graph.load(connectivity_graph, symmetric=True).toarray()
    graph = graph.astype(np.float64)
    walk = graph / graph.sum(axis=1, keepdims=True)
    entropy, correlation, _ = independent_scores(datastore, auto_filter_cells, path)
    raw = correlation if measure == "ccat" else entropy
    smoothed = 0.2 * np.linalg.solve(np.eye(len(walk)) - 0.8 * walk, raw)
    np.testing.assert_allclose(result.potency, potency.unit_ranks(smoothed))
    assert (result.graph, result.smoothing) == (connectivity_graph, 0.8)
    status = datastore.artifacts.inspect(ref)
    assert status.parameters["smoothing"] == 0.8
    assert "graph" in status.inputs


def test_a_potency_artifact_directs_fate_as_its_differentiation_does(
    datastore, auto_filter_cells, connectivity_graph, leiden_clustering, recorded
):
    path = recorded[0]
    ref = datastore.scores.potency(
        auto_filter_cells, graph=connectivity_graph, network=path
    )
    labels = np.asarray(datastore.artifacts.load_values(leiden_clustering).values)
    sinks = [int(label) for label in np.unique(labels)[:2]]
    by_artifact = datastore.trajectory.load_fate(
        datastore.trajectory.fate(
            sink_labels=leiden_clustering,
            graph=connectivity_graph,
            direction=ref,
            sinks=sinks,
        )
    )
    by_array = datastore.trajectory.load_fate(
        datastore.trajectory.fate(
            sink_labels=leiden_clustering,
            graph=connectivity_graph,
            direction=np.asarray(datastore.scores.load_potency(ref).differentiation),
            sinks=sinks,
        )
    )
    np.testing.assert_array_equal(by_artifact.valid, by_array.valid)
    np.testing.assert_allclose(by_artifact.values, by_array.values, atol=1e-12)
    assert by_artifact.direction == ref


def test_too_few_network_genes_are_refused_with_the_count_and_the_reason(
    datastore, auto_filter_cells, recorded, monkeypatch
):
    monkeypatch.setattr(trajectory_operations, "_MIN_NETWORK_GENES", 5_000)
    with pytest.raises(ValueError) as raised:
        datastore.scores.potency(
            auto_filter_cells, network=recorded[0], invalidate_cache=True
        )
    message = str(raised.value)
    assert message.startswith(
        "Only 300 genes of the STRING v12.5 human network match the features of "
        "assay 'RNA' by Ensembl gene ID, and potency scoring needs at least 5,000"
    )
    assert "no longer describes the cell's signaling" in message

    # The floor admits an assay that matches exactly as many genes as it names.
    monkeypatch.setattr(trajectory_operations, "_MIN_NETWORK_GENES", 301)
    with pytest.raises(ValueError, match="Only 300 genes .* needs at least 301:"):
        datastore.scores.potency(
            auto_filter_cells, network=recorded[0], invalidate_cache=True
        )
    monkeypatch.setattr(trajectory_operations, "_MIN_NETWORK_GENES", 300)
    ref = datastore.scores.potency(
        auto_filter_cells, network=recorded[0], invalidate_cache=True
    )
    assert datastore.scores.load_potency(ref).genes_matched == 300


def test_cells_without_counts_on_the_network_are_refused(
    tmp_path, datastore, auto_filter_cells, monkeypatch
):
    cells = cell_selection_indices(datastore.zw, auto_filter_cells)
    counts = read_selection(datastore.RNA.z["counts"], cells, slice(None))
    ids = np.asarray(datastore.RNA.feats.fetch_all("ids")).astype(str)
    # Genes that only a few cells detect leave most cells without counts.
    rare = np.flatnonzero(
        ((counts > 0).sum(axis=0) >= 1) & ((counts > 0).sum(axis=0) <= 3)
    )[:120]
    edges = np.array([np.arange(len(rare) - 1), np.arange(1, len(rare))])
    path = write_network(tmp_path / "human_9606.npz", ids[rare], ids[rare], edges)
    data = path.read_bytes()
    monkeypatch.setattr(
        datasets,
        "STRING_NETWORKS",
        {
            "human": datasets.NetworkFile(
                "human",
                9606,
                "12.5",
                "human_9606.npz",
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
        },
    )
    monkeypatch.setattr(trajectory_operations, "_MIN_NETWORK_GENES", 100)
    with pytest.raises(ValueError, match="have no counts on the network's 120 genes"):
        datastore.scores.potency(auto_filter_cells, network=path)


def test_arguments_are_refused_before_anything_is_computed(
    datastore, auto_filter_cells, connectivity_graph, recorded
):
    path = recorded[0]
    score = datastore.scores.potency
    with pytest.raises(
        ValueError, match="measure must be 'ccat' or 'signaling_entropy'"
    ):
        score(auto_filter_cells, measure="entropy", network=path)
    for smoothing in (0, 0.995, 1.5):
        with pytest.raises(ValueError, match="above 0 and at most 0.99"):
            score(auto_filter_cells, smoothing=smoothing, network=path)
    with pytest.raises(TypeError, match="smoothing must be a real number"):
        score(auto_filter_cells, smoothing="0.9", network=path)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        score("I", network=path)
    with pytest.raises(TypeError, match="species must be text"):
        score(auto_filter_cells, species=9606, network=path)
    with pytest.raises(TypeError, match="network must be a path"):
        score(auto_filter_cells, network=b"human_9606.npz")
    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        score(auto_filter_cells, graph="graph", network=path)
    with pytest.raises(TypeError, match="needs an RNAassay; received Assay"):
        score(auto_filter_cells, from_assay="assay2", network=path)
    with pytest.raises(ValueError, match="No STRING network for species 'rat'"):
        score(auto_filter_cells, species="rat", network=path)
    everyone = datastore.snapshot_cell_selection("I")
    with pytest.raises(
        ValueError, match="graph must be built on the cells of cell_selection"
    ):
        score(everyone, graph=connectivity_graph, network=path)


def test_a_selection_of_one_cell_or_of_cells_without_counts_is_refused(
    datastore, auto_filter_cells, recorded, monkeypatch
):
    path = recorded[0]
    top = float(np.max(datastore.cells.fetch_all("RNA_nCounts")))
    one = datastore.qc.filter(
        attrs=["RNA_nCounts"], lows=[top], highs=[None], keep_bounds=True
    )
    assert len(cell_selection_indices(datastore.zw, one)) == 1
    with pytest.raises(ValueError, match="at least two cells; it holds 1"):
        datastore.scores.potency(one, network=path)

    # Two cells whose prepared totals are zero cannot be scaled.
    totals_of = type(datastore.RNA)._cell_count_totals

    def without_counts(assay, cells):
        totals = np.array(totals_of(assay, cells), dtype=np.float64)
        totals[[3, 8]] = 0
        return totals

    monkeypatch.setattr(type(datastore.RNA), "_cell_count_totals", without_counts)
    cells = np.sort(cell_selection_indices(datastore.zw, auto_filter_cells))
    ids = np.asarray(datastore.cells.fetch_all("ids")).astype(str)[cells[[3, 8]]]
    with pytest.raises(ValueError) as raised:
        datastore.scores.potency(auto_filter_cells, network=path, invalidate_cache=True)
    message = str(raised.value)
    assert message.startswith(f"2 of the {len(cells):,} cells have no counts")
    assert all(repr(str(cell)) in message for cell in ids)


def test_a_stream_that_loses_cells_is_refused(
    datastore, auto_filter_cells, recorded, monkeypatch
):
    streamed = trajectory_operations.feature_value_blocks

    def short(*args, **kwargs):
        most, blocks = streamed(*args, **kwargs)
        return most, (block[:-1] for block in blocks)

    monkeypatch.setattr(trajectory_operations, "feature_value_blocks", short)
    with pytest.raises(RuntimeError, match="The count blocks held"):
        datastore.scores.potency(
            auto_filter_cells, network=recorded[0], invalidate_cache=True
        )


def test_smoothing_refuses_a_graph_of_other_cells_or_beyond_the_budget(
    datastore, connectivity_graph, monkeypatch
):
    n_cells = datastore.graph.load(connectivity_graph, symmetric=True).shape[0]
    with pytest.raises(ValueError, match="Graph cell count does not match"):
        datastore._smoothed_potency(connectivity_graph, np.zeros(n_cells + 1), 0.9)
    monkeypatch.setattr(datastore, "memoryBytes", 3 * 8 * n_cells)
    with pytest.raises(MemoryBudgetError, match="leave graph out"):
        datastore._smoothed_potency(connectivity_graph, np.zeros(n_cells), 0.9)


def test_a_read_only_store_returns_saved_scores_and_refuses_new_ones(
    datastore, auto_filter_cells, recorded
):
    path = recorded[0]
    ref = datastore.scores.potency(auto_filter_cells, network=path)
    reader = DataStore(datastore.zarr_loc, zarr_mode="r", default_assay="RNA")
    assert reader.scores.potency(auto_filter_cells, network=path) == ref
    with pytest.raises(PermissionError):
        reader.scores.potency(
            auto_filter_cells,
            measure="signaling_entropy",
            smoothing=0.5,
            network=path,
            invalidate_cache=True,
        )


def test_potency_results_hold_aligned_read_only_arrays(
    datastore, auto_filter_cells, recorded
):
    result = datastore.scores.load_potency(
        datastore.scores.potency(auto_filter_cells, network=recorded[0])
    )
    for array in (result.differentiation, result.potency, result.ccat):
        assert not array.flags.writeable
    with pytest.raises(ValueError, match="one-dimensional and align"):
        dataclasses.replace(result, ccat=result.ccat[:-1])
    with pytest.raises(ValueError, match="names both, and only then"):
        dataclasses.replace(result, smoothing=0.9)


def test_load_potency_refuses_other_artifacts(datastore, auto_filter_cells):
    with pytest.raises(TypeError, match="potency must be an ArtifactRef"):
        datastore.scores.load_potency("ref")
    with pytest.raises(ValueError, match="must be a cell_potency artifact"):
        datastore.scores.load_potency(auto_filter_cells)
    missing = ArtifactRef(
        scope="assay", kind="cell_potency", artifact_id=new_artifact_id(), assay="RNA"
    )
    with pytest.raises(ValueError, match="unavailable or invalid"):
        datastore.scores.load_potency(missing)


def test_load_potency_refuses_arrays_that_do_not_cover_their_cells(
    datastore, auto_filter_cells, recorded, monkeypatch
):
    ref = datastore.scores.potency(auto_filter_cells, network=recorded[0])
    counted = trajectory_operations._selection_size
    monkeypatch.setattr(
        trajectory_operations,
        "_selection_size",
        lambda store, selection: counted(store, selection) + 1,
    )
    with pytest.raises(ValueError, match="do not match their cell selection"):
        datastore.scores.load_potency(ref)


# ---------------------------------------------------------------------------
# The pancreas with the published mouse network

#: The stages of each endocrine lineage of the pancreas fixture, in order.
PANCREAS_LINEAGES = {
    terminal: ("Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", terminal)
    for terminal in ("Alpha", "Beta", "Delta", "Epsilon")
}


def lineage_correlations(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Spearman's rho between the values and the stage order of each lineage."""
    found = {}
    for terminal, stages in PANCREAS_LINEAGES.items():
        cells = np.isin(labels, stages)
        order = [stages.index(label) for label in labels[cells]]
        found[terminal] = float(spearmanr(values[cells], order).statistic)
    return found


@pytest.mark.slow
@pytest.mark.integration
def test_both_measures_order_every_pancreas_lineage(
    bastidas_ponce_data, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with hub_or_skip("the mouse STRING network in Nygen/cytebase"):
        if "cytearc_networks" not in datasets.list_repositories():
            pytest.skip(
                f"{NETWORK_SKIP} cytearc_networks: Nygen/cytebase does not hold the "
                "STRING networks yet"
            )
        try:
            datasets.string_network_file("mouse")
        except FileNotFoundError as error:
            # Let hub_or_skip judge why the hub could not serve the file.
            raise error.__cause__ or error
    reader = H5adReader(bastidas_ponce_data)
    try:
        H5adToZarr(reader, zarr_loc=str(tmp_path / "pancreas.zarr")).dump()
    finally:
        reader.h5.close()
    ds = DataStore(str(tmp_path / "pancreas.zarr"), default_assay="RNA")
    cells = ds.snapshot_cell_selection("I")
    labels = np.asarray(ds.cells.fetch_all("clusters")).astype(str)
    labels = labels[cell_selection_indices(ds.zw, cells)]

    # The lineages measure 0.72 to 0.81 for SR and 0.79 to 0.83 for CCAT.
    for measure, floor in (("signaling_entropy", 0.65), ("ccat", 0.75)):
        result = ds.scores.load_potency(
            ds.scores.potency(cells, measure=measure, species="mouse")
        )
        assert result.matched_by == "symbol"
        assert result.genes_used > 13_000
        correlations = lineage_correlations(result.differentiation, labels)
        assert min(correlations.values()) >= floor, (measure, correlations)
