import math
import random
from numbers import Real
from threading import Lock
from typing import Literal

import numpy as np
from scipy.sparse import sparray, spmatrix

from ..utils.arguments import integer_argument


type LeidenBackend = Literal["cytearc", "igraph", "leidenalg"]

_IGRAPH_RNG_LOCK = Lock()


def canonical_resolution(value: object) -> float:
    """Return a Leiden resolution as one finite, positive float.

    Integer and float spellings of the same value share one canonical form, so
    they also share one artifact identity.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Leiden resolution must be a number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("Leiden resolution must be finite and positive")
    return resolved


def canonical_random_seed(value: object) -> int:
    """Return a Leiden random seed as a non-negative Python integer."""
    if value is None:
        raise TypeError(
            "random_seed must be an integer; Leiden labels are reused, so an "
            "unseeded run is not supported"
        )
    return integer_argument(value, "random_seed", minimum=0)


def _igraph_membership(
    graph: spmatrix | sparray,
    resolution: float,
    random_seed: int,
) -> np.ndarray:
    try:
        import igraph
    except ImportError:
        raise ImportError(
            "The 'igraph' Leiden backend needs the igraph package, which is not "
            "installed. CyteArc does not ship it because igraph is GPL-licensed, "
            "but you are welcome to install it yourself, for example with pip "
            "install igraph. The default backend='cytearc' needs no extra package."
        ) from None

    coo = graph.tocoo(copy=True)
    coo.eliminate_zeros()
    igraph_graph = igraph.Graph(
        n=graph.shape[0],
        edges=zip(coo.row, coo.col),
        directed=False,
    )
    with _IGRAPH_RNG_LOCK:
        igraph.set_random_number_generator(random.Random(random_seed))
        try:
            partition = igraph_graph.community_leiden(
                weights=coo.data,
                objective_function="modularity",
                resolution=resolution,
                n_iterations=2,
            )
        finally:
            # Restore igraph's default generator, the seedable random module.
            # None would switch to the C-layer generator, which cannot be seeded.
            igraph.set_random_number_generator(random)
    return np.array(partition.membership) + 1


def _leidenalg_membership(
    graph: spmatrix | sparray,
    resolution: float,
    random_seed: int,
) -> np.ndarray:
    try:
        import igraph
        import leidenalg
    except ImportError:
        raise ImportError(
            "The 'leidenalg' Leiden backend needs the leidenalg and igraph "
            "packages, which are not installed. CyteArc does not ship them because "
            "they are GPL-licensed, but you are welcome to install them yourself, "
            "for example with pip install leidenalg, which also installs igraph. "
            "The default backend='cytearc' needs no extra package."
        ) from None

    coo = graph.tocoo(copy=True)
    coo.eliminate_zeros()
    igraph_graph = igraph.Graph()
    igraph_graph.add_vertices(graph.shape[0])
    igraph_graph.add_edges(list(zip(coo.row, coo.col)))
    igraph_graph.es["weight"] = coo.data
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_seed,
    )
    return np.array(partition.membership) + 1


def leiden_membership(
    graph: spmatrix | sparray,
    resolution: float,
    random_seed: int,
    backend: LeidenBackend = "cytearc",
    *,
    nthreads: int | None = None,
) -> np.ndarray:
    """Cluster a sparse graph with the Leiden algorithm.

    The Leiden algorithm (Traag, Waltman and van Eck, 2019) maximises
    modularity with a resolution parameter ``gamma`` (the configuration null
    model). Every stored entry ``(i, j, w)`` of ``graph`` is one undirected
    edge of weight ``w``, so the adjacency is ``A = W + W^T``; explicitly
    stored zeros are ignored. Larger resolutions give more, smaller clusters.

    Args:
        graph: Square SciPy sparse matrix or array with one row per node and
            non-negative, finite weights.
        resolution: Finite, positive resolution ``gamma``.
        random_seed: Non-negative integer seed. The same graph, resolution
            and seed always give the same labels.
        backend: ``"cytearc"`` (the default) is CyteArc's own multi-threaded
            implementation; it labels clusters ``1..r`` by decreasing size,
            ties broken by the smallest node index, and reduces the seed
            modulo ``2**64``. ``"igraph"`` and ``"leidenalg"`` call those
            GPL-licensed packages, which CyteArc does not install, and are
            kept to reproduce earlier results.
        nthreads: Worker threads of the ``"cytearc"`` backend, or ``None``
            for Numba's current thread count. It only changes the speed,
            never the labels. The other backends ignore it.

    Returns:
        One positive integer label per row, starting at 1.

    Raises:
        ValueError: If ``backend`` is unknown, or for invalid arguments or
            graphs of the ``"cytearc"`` backend.
        TypeError: For arguments or graphs of the wrong type
            (``"cytearc"`` backend).
        ImportError: If the GPL backend's package is not installed.

    Reference:
        Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
        well-connected communities. Sci Rep 9, 5233 (2019).
        doi:10.1038/s41598-019-41695-z
    """
    if backend == "cytearc":
        from ._leiden_native import native_leiden_membership

        resolution = canonical_resolution(resolution)
        random_seed = canonical_random_seed(random_seed)
        threads = (
            None
            if nthreads is None
            else integer_argument(nthreads, "nthreads", minimum=1)
        )
        return native_leiden_membership(graph, resolution, random_seed, threads)
    if backend == "igraph":
        return _igraph_membership(graph, resolution, random_seed)
    if backend == "leidenalg":
        return _leidenalg_membership(graph, resolution, random_seed)
    raise ValueError("backend must be 'cytearc', 'igraph' or 'leidenalg'")
