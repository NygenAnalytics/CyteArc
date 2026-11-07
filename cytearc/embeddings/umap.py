import locale
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ..utils.arguments import integer_argument
from ..utils.compute import process_thread_limit
from ..utils.logging import logger
from ..utils.numba import enable_dispatcher_cache, import_umap, numba_threads
from ..utils.progress import tqdm_params as default_tqdm_params

locale.setlocale(locale.LC_NUMERIC, "C")

# Local radii whose logarithms span at most this are equal. The KNN distances
# are float32, with a relative resolution of about 1.2e-7, so such a spread
# reflects rounding rather than density.
_EQUAL_LOG_RADIUS_SPAN = 1e-6
#: The warning that densMAP gives over local radii that are equal for every cell.
EQUAL_DENSITY_RADII_WARNING = (
    "densMAP local radii are equal for every cell to float32 resolution, so "
    "densMAP adds no density term and lays the cells out as UMAP does"
)


def equal_density_radii(span: float) -> bool:
    """Whether local radii whose logarithms span ``span`` are equal, so densMAP adds no density term."""
    return span <= _EQUAL_LOG_RADIUS_SPAN


def _warn_equal_radii(span: float | None) -> None:
    """Warn when densMAP's local radii are equal; ``span`` is None without densMAP."""
    if span is not None and equal_density_radii(span):
        logger.warning(EQUAL_DENSITY_RADII_WARNING)


def umap_peak_bytes(
    n_cells: int,
    n_entries: int,
    union_entries: int,
    *,
    densmap_edges: int = 0,
) -> int:
    """Estimate the bytes ``fit_transform`` holds besides its graph and coordinates.

    The layout takes a COO copy of the graph's rows and forms the fuzzy union
    of its ``n_entries`` entries through CSR copies, sums, and products that
    SciPy sizes for the entries of both operands. It then lays out the
    union's ``union_entries`` edges with their COO arrays and four float64
    sampling schedules per edge.

    densMAP, given the ``densmap_edges`` KNN pairs it reads, also holds their
    uint32 indices and float32 distances, the symmetric distance matrix of up
    to twice as many entries, and float64 copies of the union's weights and
    edge distances while it fits the local radii; building the distance
    matrix holds int64 and int32 copies of the pairs.

    Args:
        n_cells: Cells of the graph.
        n_entries: Entries of the graph that the layout receives.
        union_entries: Entries of its fuzzy union, at most twice
            ``n_entries``.
        densmap_edges: KNN pairs that densMAP reads, or 0 without densMAP.

    Returns:
        The estimated peak in bytes.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below its minimum.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    entries = integer_argument(n_entries, "n_entries", minimum=0)
    union = integer_argument(union_entries, "union_entries", minimum=0)
    pairs = integer_argument(densmap_edges, "densmap_edges", minimum=0)
    layout = 24 * entries + 40 * union + 72 * cells
    if not pairs:
        return layout
    distances = 8 * pairs + 16 * pairs + 4 * (cells + 1)
    building = 8 * pairs + 32 * pairs + 12 * (cells + 1)
    radii = 32 * union + 16 * cells
    return max(building, layout + distances + radii)


def densmap_distance_graph(
    indices: np.ndarray,
    distances: np.ndarray,
) -> csr_matrix:
    """Build the symmetric cell-by-cell KNN distance matrix used by densMAP.

    A connectivity edge can come from either endpoint's neighbor list, so each
    pair keeps the larger of its two directed distances, as umap-learn does.
    """
    neighbor_indices = np.asarray(indices)
    neighbor_distances = np.asarray(distances)
    if neighbor_indices.ndim != 2 or neighbor_indices.shape != neighbor_distances.shape:
        raise ValueError("KNN indices and distances must be matching matrices")
    n_cells, n_neighbors = neighbor_indices.shape
    directed = csr_matrix(
        (
            neighbor_distances.ravel(),
            (
                np.repeat(np.arange(n_cells), n_neighbors),
                neighbor_indices.ravel(),
            ),
        ),
        shape=(n_cells, n_cells),
    )
    return csr_matrix(directed.maximum(directed.transpose()))


def calc_dens_map_params(
    graph: Any,
    dists: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute densMAP correction terms from graph and KNN distances.

    Args:
        graph: COO graph whose edges carry membership strengths.
        dists: Dense or sparse cell-by-cell distance matrix. Each graph edge
            reads the distance stored at its own row and column.

    Returns:
        Per-cell membership sums and standardized log local radii. Radii
        that are equal for every cell to float32 resolution standardize to
        zero, and a warning says that densMAP adds no density term.

    Raises:
        ValueError: If a local radius is not finite.
    """
    mu_sum, standardized_ro, span = _density_terms(graph, dists)
    _warn_equal_radii(span)
    return mu_sum, standardized_ro


def _density_terms(graph: Any, dists: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute densMAP's terms as :func:`calc_dens_map_params` does, without its warning.

    Returns:
        Per-cell membership sums, standardized log local radii, and the span
        of the log local radii, their largest minus their smallest.
    """
    n_vertices = graph.shape[0]
    head = np.asarray(graph.row)
    tail = np.asarray(graph.col)
    mu = np.asarray(graph.data, dtype=np.float64)
    if issparse(dists):
        edge_distances = np.asarray(
            csr_matrix(dists)[head, tail],
            dtype=np.float64,
        ).ravel()
    else:
        edge_distances = np.asarray(np.asarray(dists)[head, tail], dtype=np.float64)
    weighted = mu * edge_distances * edge_distances
    ro = np.bincount(head, weights=weighted, minlength=n_vertices) + np.bincount(
        tail, weights=weighted, minlength=n_vertices
    )
    mu_sum = np.bincount(head, weights=mu, minlength=n_vertices) + np.bincount(
        tail, weights=mu, minlength=n_vertices
    )

    epsilon = 1e-8
    # A cell without graph edges has radius 0 / 0, which the check below rejects.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ro = np.log(epsilon + (ro / mu_sum))
    span = float(np.ptp(log_ro)) if log_ro.size else 0.0
    if equal_density_radii(span):
        standardized_ro = np.zeros_like(log_ro)
    else:
        standardized_ro = (log_ro - np.mean(log_ro)) / np.std(log_ro)
    if not np.all(np.isfinite(standardized_ro)):
        raise ValueError(
            "densMAP local radii are not finite. Every cell needs a graph edge "
            "with a positive weight and a finite distance."
        )
    return mu_sum.astype(np.float32), standardized_ro.astype(np.float32), span


def layout_threads(parallel: bool, nthreads: int) -> int:
    """Return the number of Numba threads that a UMAP layout runs on."""
    if not parallel:
        return 1
    import numba

    return max(1, min(int(nthreads), int(numba.config.NUMBA_NUM_THREADS)))


def simplicial_set_embedding(
    g: Any,
    embedding: np.ndarray,
    n_epochs: int,
    a: float,
    b: float,
    random_seed: int,
    gamma: float,
    initial_alpha: float,
    negative_sample_rate: float,
    densmap_kwds: dict[str, Any],
    parallel: bool,
    nthreads: int,
    verbose: bool,
) -> np.ndarray:
    """Run UMAP simplicial-set embedding with optional densMAP.

    densMAP over local radii that are equal for every cell warns that it adds
    no density term.
    """
    layout, span = _embed_simplicial_set(
        g,
        embedding,
        n_epochs,
        a,
        b,
        random_seed,
        gamma,
        initial_alpha,
        negative_sample_rate,
        densmap_kwds,
        parallel,
        nthreads,
        verbose,
    )
    _warn_equal_radii(span)
    return layout


def _embed_simplicial_set(
    g: Any,
    embedding: np.ndarray,
    n_epochs: int,
    a: float,
    b: float,
    random_seed: int,
    gamma: float,
    initial_alpha: float,
    negative_sample_rate: float,
    densmap_kwds: dict[str, Any],
    parallel: bool,
    nthreads: int,
    verbose: bool,
) -> tuple[np.ndarray, float | None]:
    """Run :func:`simplicial_set_embedding` without its densMAP warning.

    Returns:
        The embedding, and the span of densMAP's log local radii, their
        largest minus their smallest, or None without densMAP.
    """
    from sklearn.utils import check_random_state

    import_umap()
    from umap import layouts as umap_layouts
    from umap.layouts import optimize_layout_euclidean
    from umap.umap_ import make_epochs_per_sample

    # umap-learn compiles the serial and parallel layout epochs, which it
    # keeps under these private names, without an on-disk cache. A release
    # that renames them lays out uncached.
    enable_dispatcher_cache(
        getattr(umap_layouts, "_nb_optimize_layout_euclidean_single_epoch", None),
        getattr(
            umap_layouts, "_nb_optimize_layout_euclidean_single_epoch_parallel", None
        ),
    )
    epochs_per_sample = make_epochs_per_sample(g.data, n_epochs)
    logger.trace("calculated epochs_per_sample")
    rng_state = (
        check_random_state(random_seed)
        .randint(np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3)
        .astype(np.int64)
    )
    span: float | None = None
    # The calling thread's count may be below the request, so always set it.
    with numba_threads(layout_threads(parallel, nthreads)):
        if densmap_kwds != {}:
            with process_thread_limit(nthreads):
                mu_sum, standardized_ro, span = _density_terms(
                    g,
                    densmap_kwds["knn_dists"],
                )
            densmap_kwds["mu_sum"] = mu_sum
            densmap_kwds["R"] = standardized_ro
            densmap_kwds["mu"] = g.data
            densmap = True
            logger.trace("calculated densmap params")
        else:
            densmap = False

        tqdm_params = dict(default_tqdm_params)
        tqdm_params["desc"] = "Training UMAP"
        tqdm_params["disable"] = not verbose

        # Numba's thread count is per thread, so the layout needs no process-wide
        # BLAS limit.
        embedding = optimize_layout_euclidean(
            head_embedding=embedding,
            tail_embedding=embedding,
            head=g.row,
            tail=g.col,
            n_epochs=n_epochs,
            n_vertices=g.shape[1],
            epochs_per_sample=epochs_per_sample,
            a=a,
            b=b,
            rng_state=rng_state,
            gamma=gamma,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            parallel=parallel,
            verbose=False,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
            tqdm_kwds=tqdm_params,
            move_other=True,
        )
    return np.asarray(embedding), span


def fuzzy_simplicial_set(g: Any, set_op_mix_ratio: float) -> Any:
    """Combine directed and undirected graph components for UMAP."""
    transposed_graph = g.transpose()
    product = g.multiply(transposed_graph)
    result = (
        set_op_mix_ratio * (g + transposed_graph - product)
        + (1.0 - set_op_mix_ratio) * product
    )
    result.eliminate_zeros()
    return result.tocoo()


def fit_transform(
    graph: Any,
    ini_embed: np.ndarray,
    spread: float,
    min_dist: float,
    n_epochs: int,
    random_seed: int,
    repulsion_strength: float,
    initial_alpha: float,
    negative_sample_rate: float,
    densmap_kwds: dict[str, Any],
    parallel: bool,
    nthreads: int,
    verbose: bool,
) -> tuple[np.ndarray, float, float]:
    """Lay out the fuzzy union of a graph of directed membership strengths.

    ``graph`` holds the directed edge weights, such as a connectivity map.
    This function forms their fuzzy union ``w_ij + w_ji - w_ij * w_ji``
    itself, so a graph that was already united this way would be united
    twice, each weight ``w`` becoming ``2w - w**2``. densMAP over local radii
    that are equal for every cell warns that it adds no density term.
    """
    embedding, a, b, span = _fit_layout(
        graph,
        ini_embed,
        spread,
        min_dist,
        n_epochs,
        random_seed,
        repulsion_strength,
        initial_alpha,
        negative_sample_rate,
        densmap_kwds,
        parallel,
        nthreads,
        verbose,
    )
    _warn_equal_radii(span)
    return embedding, a, b


def _fit_layout(
    graph: Any,
    ini_embed: np.ndarray,
    spread: float,
    min_dist: float,
    n_epochs: int,
    random_seed: int,
    repulsion_strength: float,
    initial_alpha: float,
    negative_sample_rate: float,
    densmap_kwds: dict[str, Any],
    parallel: bool,
    nthreads: int,
    verbose: bool,
) -> tuple[np.ndarray, float, float, float | None]:
    """Lay out a graph as :func:`fit_transform` does, without its densMAP warning.

    The UMAP operation saves the span of densMAP's log local radii and warns
    from it, so a reused embedding warns as a computed one does.

    Returns:
        The embedding, the curve parameters ``a`` and ``b``, and the span of
        densMAP's log local radii, or None without densMAP.
    """
    import warnings

    import_umap()
    from umap.umap_ import find_ab_params

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a, b = find_ab_params(spread=spread, min_dist=min_dist)
    logger.trace("Found ab params")

    embedding, span = _embed_simplicial_set(
        fuzzy_simplicial_set(graph, 1.0),
        ini_embed,
        n_epochs,
        a,
        b,
        random_seed,
        repulsion_strength,
        initial_alpha,
        negative_sample_rate,
        densmap_kwds,
        parallel,
        nthreads,
        verbose,
    )
    return embedding, a, b, span
