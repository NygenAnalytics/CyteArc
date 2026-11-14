"""Score each cell's differentiation potency from its expression on a protein network.

Two measures, written from their papers:

- The signaling entropy rate (SR) of Teschendorff and Enver (2017), Nature
  Communications 8:15599, doi:10.1038/ncomms15599, after Banerji et al.
  (2013), Scientific Reports 3:3039, doi:10.1038/srep03039. A cell's
  expression ``x`` on a connected protein interaction network with adjacency
  ``A`` defines a walk ``p_ij = A_ij x_j / (Ax)_i`` whose stationary
  distribution is ``pi_i = x_i (Ax)_i / x'Ax``. SR is the walk's entropy rate,
  ``-sum_i pi_i sum_j p_ij log p_ij``, which reduces to
  ``sum_i pi_i log((Ax)_i / x_i)``, divided by the largest entropy rate of any
  walk on the network, ``log lambda_max(A)``. It measures how evenly a cell's
  expression lets signals spread across the network.
- CCAT of Teschendorff et al. (2021), Bioinformatics 37:1528-1534,
  doi:10.1093/bioinformatics/btaa987: the Pearson correlation between ``x`` and
  each gene's degree in the network, whether a cell expresses the network's
  hubs.

Both read ``x = log2(10,000 * counts / total + 1.1)`` over the network's
genes, each cell scaled to 10,000 counts of all its genes with the papers'
pseudocount of 1.1, so a network gene without counts enters at ``log2(1.1)``.
Fixing the scale, rather than scaling to the largest total of the cells
scored, gives each cell a score that depends on that cell alone. Higher
values of either measure mean a less differentiated cell.
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numba import prange
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata

from ..features.gene_reference import strip_ensembl_version
from ..neighbors.diffusion import transition_matrix
from ..utils.numba import njit

__all__ = [
    "MEASURES",
    "MIN_NETWORK_GENES",
    "PSEUDOCOUNT",
    "SCALE_TOTAL",
    "MatchedNetwork",
    "ProteinNetwork",
    "largest_eigenvalue",
    "match_network",
    "network_scores",
    "read_network",
    "smooth_on_graph",
    "smoothing_steps",
    "unit_ranks",
]

#: The measures that ``ds.scores.potency`` computes, the first its default.
MEASURES: tuple[str, ...] = ("ccat", "signaling_entropy")
#: The papers' pseudocount, which keeps a gene without counts above zero.
PSEUDOCOUNT = 1.1
#: The counts that every cell is scaled to before the log.
SCALE_TOTAL = 10_000.0
#: The fewest network genes that an assay's features must reach. Both
#: measures summarize a cell's expression across the network, and a network
#: cut down to a few genes no longer describes the cell's signaling, nor
#: scores cells that other assays score on the whole network alike.
MIN_NETWORK_GENES = 5_000
#: Networks this small take the dense eigenvalue solver.
_DENSE_EIGEN_LIMIT = 256
#: The bound on the smoothing's distance from its fixed point, relative to
#: the range of the values smoothed.
_SMOOTHING_TOLERANCE = 1e-12
#: The largest smoothing accepted: beyond it, reaching the tolerance takes
#: more than 2,700 steps over the graph.
MAX_SMOOTHING = 0.99


@dataclass(frozen=True, slots=True)
class ProteinNetwork:
    """A protein interaction network over genes.

    ``edges`` holds each edge once, as two rows of gene positions with the
    lower position first; ``metadata`` is the JSON text that names its
    source, as ``scripts/build_string_networks.py`` writes it.
    """

    ensembl_ids: np.ndarray = field(repr=False)
    symbols: np.ndarray = field(repr=False)
    edges: np.ndarray = field(repr=False)
    metadata: dict[str, Any]

    @property
    def n_genes(self) -> int:
        return int(len(self.ensembl_ids))


def read_network(path: str | os.PathLike[str]) -> ProteinNetwork:
    """Read and check a network archive written by ``scripts/build_string_networks.py``.

    Raises:
        ValueError: If the archive lacks an array, or its genes or edges are
            malformed.
    """
    with np.load(path, allow_pickle=False) as stored:
        missing = {"ensembl_ids", "symbols", "edges", "metadata"} - set(stored.files)
        if missing:
            raise ValueError(
                f"Network {os.fspath(path)} lacks the arrays {sorted(missing)}"
            )
        ensembl_ids = np.asarray(stored["ensembl_ids"]).astype(str)
        symbols = np.asarray(stored["symbols"]).astype(str)
        edges = np.asarray(stored["edges"])
        metadata_text = str(stored["metadata"])
    n_genes = len(ensembl_ids)
    if ensembl_ids.ndim != 1 or symbols.shape != ensembl_ids.shape or n_genes < 2:
        raise ValueError(
            "A network must name at least two genes, with one Ensembl ID and one "
            "symbol each"
        )
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] < 1:
        raise ValueError("A network's edges must be two rows of gene positions")
    if edges.dtype.kind not in "iu":
        raise ValueError("A network's edges must be integer gene positions")
    edges = edges.astype(np.int64)
    if int(edges.min()) < 0 or int(edges.max()) >= n_genes:
        raise ValueError("A network's edges must name genes of the network")
    if not bool(np.all(edges[0] < edges[1])):
        raise ValueError(
            "A network must list each edge once, lower gene position first"
        )
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError("A network's metadata must be JSON text") from error
    if not isinstance(metadata, dict):
        raise ValueError("A network's metadata must be a JSON object")
    return ProteinNetwork(
        ensembl_ids=ensembl_ids, symbols=symbols, edges=edges, metadata=metadata
    )


@dataclass(frozen=True, slots=True)
class MatchedNetwork:
    """The connected part of a network that an assay's features reach.

    ``features`` gives the feature position of each gene of ``adjacency``,
    a symmetric 0/1 matrix of one connected component: the largest one that
    the matched genes form. ``genes_matched`` counts every network gene that
    a feature matched, inside that component or not.
    """

    features: np.ndarray = field(repr=False)
    adjacency: sparse.csr_matrix = field(repr=False)
    degree: np.ndarray = field(repr=False)
    log_lambda: float
    genes_matched: int

    @property
    def genes_used(self) -> int:
        return int(len(self.features))

    @property
    def edges_used(self) -> int:
        return int(self.adjacency.nnz // 2)

    @property
    def nbytes(self) -> int:
        """The bytes of the arrays that scoring holds."""
        return int(
            self.features.nbytes
            + self.degree.nbytes
            + self.adjacency.data.nbytes
            + self.adjacency.indices.nbytes
            + self.adjacency.indptr.nbytes
        )


def largest_eigenvalue(adjacency: sparse.csr_matrix) -> float:
    """Return the largest eigenvalue of a symmetric non-negative matrix.

    Small matrices take the dense solver; larger ones ARPACK, started from the
    vector of ones, so repeated calls return the same value.
    """
    n = adjacency.shape[0]
    if n <= _DENSE_EIGEN_LIMIT:
        return float(np.linalg.eigvalsh(adjacency.toarray())[-1])
    value = eigsh(
        adjacency,
        k=1,
        which="LA",
        v0=np.ones(n),
        return_eigenvectors=False,
    )
    return float(value[0])


def match_network(
    network: ProteinNetwork,
    feature_keys: np.ndarray,
    key: Literal["ensembl", "symbol"],
) -> MatchedNetwork:
    """Match a network's genes to features and keep the largest connected part.

    Network genes match features by Ensembl gene ID, versions ignored, or by
    symbol, exactly. A key that several features share matches the first of
    them.

    Raises:
        ValueError: If ``key`` is unknown, or the matched genes share no edge.
    """
    if key == "ensembl":
        names = network.ensembl_ids
        keys = [strip_ensembl_version(str(value)) for value in feature_keys]
    elif key == "symbol":
        names = network.symbols
        keys = [str(value) for value in feature_keys]
    else:
        raise ValueError(f"key must be 'ensembl' or 'symbol', got {key!r}")
    first: dict[str, int] = {}
    for position, value in enumerate(keys):
        first.setdefault(value, position)
    node_features = np.fromiter(
        (first.get(str(name), -1) for name in names),
        dtype=np.int64,
        count=network.n_genes,
    )
    matched = np.flatnonzero(node_features >= 0)
    renumber = np.full(network.n_genes, -1, dtype=np.int64)
    renumber[matched] = np.arange(len(matched))
    low, high = renumber[network.edges[0]], renumber[network.edges[1]]
    inside = (low >= 0) & (high >= 0)
    low, high = low[inside], high[inside]
    if low.size == 0:
        raise ValueError(
            f"The {len(matched)} network genes that the features match share no "
            "edge of the network"
        )
    n = len(matched)
    induced = sparse.coo_matrix(
        (np.ones(2 * low.size), (np.r_[low, high], np.r_[high, low])), shape=(n, n)
    ).tocsr()
    _, component = connected_components(induced, directed=False)
    sizes = np.bincount(component)
    # The largest component, the first of equal size by its first gene.
    largest = np.flatnonzero(component == int(np.argmax(sizes)))
    adjacency = induced[largest][:, largest].tocsr()
    adjacency.sort_indices()
    adjacency.data[:] = 1.0
    # The kernel walks the edges by int64 positions.
    adjacency.indptr = adjacency.indptr.astype(np.int64)
    adjacency.indices = adjacency.indices.astype(np.int64)
    degree = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()
    return MatchedNetwork(
        features=node_features[matched[largest]],
        adjacency=adjacency,
        degree=degree,
        log_lambda=math.log(largest_eigenvalue(adjacency)),
        genes_matched=int(len(matched)),
    )


@njit(cache=True, nogil=True, parallel=True)
def _score_rows(
    block: np.ndarray,
    totals: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    degree: np.ndarray,
    log_lambda: float,
    pseudocount: float,
    scale_total: float,
    entropy: np.ndarray,
    correlation: np.ndarray,
    detected: np.ndarray,
) -> None:
    n_rows, n_genes = block.shape
    base = math.log2(pseudocount)
    mean_degree = 0.0
    for gene in range(n_genes):
        mean_degree += degree[gene]
    mean_degree /= n_genes
    degree_spread = 0.0
    for gene in range(n_genes):
        degree_spread += (degree[gene] - mean_degree) ** 2
    for row in prange(n_rows):
        values = np.full(n_genes, base)
        neighbors = base * degree
        scale = scale_total / totals[row]
        found = 0
        for gene in range(n_genes):
            count = block[row, gene]
            if count != 0:
                found += 1
                value = math.log2(scale * count + pseudocount)
                values[gene] = value
                extra = value - base
                for edge in range(indptr[gene], indptr[gene + 1]):
                    neighbors[indices[edge]] += extra
        flux = 0.0
        weighted = 0.0
        total_value = 0.0
        for gene in range(n_genes):
            share = values[gene] * neighbors[gene]
            flux += share
            weighted += share * (math.log(neighbors[gene]) - math.log(values[gene]))
            total_value += values[gene]
        entropy[row] = weighted / flux / log_lambda
        mean_value = total_value / n_genes
        spread = 0.0
        together = 0.0
        for gene in range(n_genes):
            centered = values[gene] - mean_value
            spread += centered * centered
            together += centered * (degree[gene] - mean_degree)
        if spread > 0.0:
            correlation[row] = together / math.sqrt(spread * degree_spread)
        else:
            correlation[row] = np.nan
        detected[row] = found


def network_scores(
    block: np.ndarray, totals: np.ndarray, network: MatchedNetwork
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the SR, CCAT and detected network genes of each cell of a block.

    ``block`` holds the raw counts of the network's genes, cells by genes in
    the order of ``network.features``; ``totals`` each cell's counts over all
    its genes. A cell without counts on the network's genes has CCAT NaN,
    since its values do not vary.

    Raises:
        ValueError: If the block does not match the network or the totals, or
            a total is not positive.
    """
    block = np.asarray(block)
    totals = np.asarray(totals, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != network.genes_used:
        raise ValueError(
            f"A block must hold the counts of the network's {network.genes_used} "
            f"genes, got shape {block.shape}"
        )
    if totals.shape != (block.shape[0],):
        raise ValueError("totals must give one total for each row of the block")
    if block.shape[0] and not bool(np.all(totals > 0)):
        raise ValueError("Every cell's total count must be positive")
    entropy = np.empty(block.shape[0], dtype=np.float64)
    correlation = np.empty(block.shape[0], dtype=np.float64)
    detected = np.empty(block.shape[0], dtype=np.int64)
    if block.shape[0]:
        _score_rows(
            np.ascontiguousarray(block),
            totals,
            network.adjacency.indptr,
            network.adjacency.indices,
            network.degree,
            network.log_lambda,
            PSEUDOCOUNT,
            SCALE_TOTAL,
            entropy,
            correlation,
            detected,
        )
    return entropy, correlation, detected


def smoothing_steps(smoothing: float) -> int:
    """Return the steps after which the smoothing is within its tolerance of its fixed point."""
    return int(math.ceil(math.log(_SMOOTHING_TOLERANCE) / math.log(smoothing)))


def smooth_on_graph(
    values: np.ndarray, graph: sparse.spmatrix, smoothing: float
) -> np.ndarray:
    """Smooth per-cell values by a random walk with restart on a graph.

    Each step keeps ``smoothing`` of the neighbors' values, weighted by the
    graph's row-normalized edges, and restarts ``1 - smoothing`` at the
    original value: ``d <- smoothing * P d + (1 - smoothing) * values``. The
    fixed point is ``(1 - smoothing) (I - smoothing P)^-1 values``, which the
    steps of :func:`smoothing_steps` reach within ``1e-12`` of the values'
    range, since every step shrinks the distance to it by ``smoothing``. A cell
    without edges keeps its value, and no value crosses from one connected
    component to another.

    Raises:
        ValueError: If the graph does not match the values, or ``smoothing``
            is not in (0, 0.99].
    """
    values = np.asarray(values, dtype=np.float64)
    if not 0.0 < smoothing <= MAX_SMOOTHING:
        raise ValueError(
            f"smoothing must be above 0 and at most {MAX_SMOOTHING}, got {smoothing}"
        )
    graph = sparse.csr_matrix(graph, dtype=np.float64)
    if graph.shape != (len(values), len(values)):
        raise ValueError(
            f"A graph of {graph.shape[0]} cells cannot smooth {len(values)} values"
        )
    step = transition_matrix(graph).tocsr()
    lonely = np.flatnonzero(np.diff(step.indptr) == 0)
    if lonely.size:
        # A cell without edges walks to itself.
        step = (
            step
            + sparse.csr_matrix(
                (np.ones(lonely.size), (lonely, lonely)), shape=step.shape
            )
        ).tocsr()
    smoothed = values.copy()
    restart = (1.0 - smoothing) * values
    for _ in range(smoothing_steps(smoothing)):
        smoothed = smoothing * (step @ smoothed) + restart
    return np.asarray(smoothed)


def unit_ranks(values: np.ndarray) -> np.ndarray:
    """Return the values' ranks scaled to [0, 1], ties sharing their mean rank.

    Raises:
        ValueError: If there are fewer than two values or one is not finite.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Ranks need at least two values")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("Ranks need finite values")
    ranks = rankdata(values, method="average")
    return np.asarray((ranks - 1.0) / (len(values) - 1.0), dtype=np.float64)
