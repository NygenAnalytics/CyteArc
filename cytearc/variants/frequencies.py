"""Allele frequencies of chosen variants in each cell, and the clones they define.

A cell's allele frequency of a variant is its count of the alternative
allele over its depth at the site. Where the depth is 0 the cell was not
covered, and its frequency is undefined: it is NaN here, never 0. Signac's
``AlleleFreq`` sets it to 0, which makes an uncovered cell look like one
that carries no alternative allele.

:func:`band_frequencies` turns the stored rows of a band of cells into a
dense block of frequencies. :func:`frequency_clones` calls clones from those
frequencies as Signac's ``FindClonotypes`` does: each cell's neighbourhood
is itself and the cells most similar to it by the cosine similarity of
square-root allele frequencies; neighbourhoods give a shared-neighbour graph
whose edges weigh the Jaccard index of two neighbourhoods, as Seurat's
``FindNeighbors`` builds it; and the Leiden algorithm partitions the graph
where Seurat uses SLM. The similarity of two cells counts only the variants
that both cover. Cells that cover no chosen variant, that show none, or
that are similar to no other cell carry no evidence of a clone and get
none. Every cell is compared with every other cell of its group, so the
time grows with the square of the cells. These functions never touch
storage.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .counts import SiteTable

if TYPE_CHECKING:
    from scipy.sparse import coo_matrix

__all__ = [
    "CLONE_STATUSES",
    "FrequencyClones",
    "band_frequencies",
    "clone_status_counts",
    "cosine_neighborhoods",
    "frequency_clones",
    "neighborhood_bytes_per_cell",
]

#: Why a cell has or lacks a clone, by code: clustered; not covered at any
#: chosen variant; covered without showing any; similar to no other cell;
#: no group.
CLONE_STATUSES: tuple[str, ...] = (
    "clustered",
    "no_coverage",
    "no_alternative",
    "no_neighbor",
    "no_group",
)
# Similarities are rounded to this many decimals before neighbours are
# ranked, so that the last bits of matrix products, which may depend on the
# BLAS library, do not reorder ties.
_SIMILARITY_DECIMALS = 12
# Arrays of the size of a block of similarities held at once while
# neighbours are chosen: the products, the similarities, and the masks and
# ranks of the selection.
_BLOCK_ARRAYS = 8
# Bytes per cell and variant held while a group's neighbourhoods are found:
# the square-root frequencies, the covered values, their squares and the
# coverage weights as float64, and the coverage mask, rounded up.
_VALUE_BYTES = 40
# Bytes per cell of the per-cell arrays: its group, status and clone.
_CELL_BYTES = 24
# Bytes per neighbourhood slot while the shared-neighbour graph is built:
# the neighbourhoods, the matrix of who holds whom and its transpose, and
# the arrays that build them and bound each cell's joins.
_SLOT_BYTES = 64
# Bytes of a kept join of the shared-neighbour graph (its cells and weight,
# and their copies while the blocks' joins are merged), and of a candidate
# join while a block is computed (the product's entries, their cells,
# Jaccard indices, masks and order).
_JOIN_BYTES = 64
_CANDIDATE_BYTES = 96
# Bytes per entry of the CSR graph that the Leiden algorithm reads.
_CSR_ENTRY_BYTES = 12


def band_frequencies(
    *,
    n_rows: int,
    depth_rows: np.ndarray,
    depth_sites: np.ndarray,
    depth: np.ndarray,
    count_rows: np.ndarray,
    count_alleles: np.ndarray,
    counts: np.ndarray,
    alleles: np.ndarray,
    sites: SiteTable,
) -> np.ndarray:
    """Return a band of cells' allele frequencies of chosen variants.

    Args:
        n_rows: The cells of the band.
        depth_rows: Each depth row's cell, as a row of the band.
        depth_sites: Each depth row's site.
        depth: Each depth row's depth.
        count_rows: Each count row's cell, as a row of the band.
        count_alleles: Each count row's allele.
        counts: Each count row's count.
        alleles: The chosen variants, as alleles of ``sites``.
        sites: The sites and alleles of the counts.

    Returns:
        Frequencies shaped (cells, variants), float64, NaN where the cell's
        depth at the variant's site is 0.

    Raises:
        ValueError: If an allele is counted where its site has no depth, or
            a count exceeds the depth.
    """
    alleles = np.asarray(alleles, dtype=np.int64)
    variant_sites = sites.allele_site[alleles].astype(np.int64)
    distinct_sites, site_slots = np.unique(variant_sites, return_inverse=True)
    slot_of_site = np.full(sites.n_sites, -1, dtype=np.int64)
    slot_of_site[distinct_sites] = np.arange(len(distinct_sites))
    column_of_allele = np.full(sites.n_alleles, -1, dtype=np.int64)
    column_of_allele[alleles] = np.arange(len(alleles))
    site_depth = np.zeros((n_rows, len(distinct_sites)), dtype=np.float64)
    slots = slot_of_site[np.asarray(depth_sites, dtype=np.int64)]
    kept = slots >= 0
    site_depth[np.asarray(depth_rows, dtype=np.int64)[kept], slots[kept]] = np.asarray(
        depth, dtype=np.float64
    )[kept]
    allele_counts = np.zeros((n_rows, len(alleles)), dtype=np.float64)
    columns = column_of_allele[np.asarray(count_alleles, dtype=np.int64)]
    kept = columns >= 0
    allele_counts[np.asarray(count_rows, dtype=np.int64)[kept], columns[kept]] = (
        np.asarray(counts, dtype=np.float64)[kept]
    )
    variant_depth = site_depth[:, site_slots]
    if np.any((allele_counts > 0) & (variant_depth == 0)):
        raise ValueError("An allele is counted in a cell that has no depth at its site")
    if np.any(allele_counts > variant_depth):
        raise ValueError("A count exceeds the depth at its site")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variant_depth > 0, allele_counts / variant_depth, np.nan)


def neighborhood_bytes_per_cell(n_cells: int, n_variants: int) -> int:
    """Bytes that :func:`cosine_neighborhoods` holds per row of a block, beside
    the arrays of every cell's values: arrays the width of the cells, and
    copies of the block's variants."""
    return _BLOCK_ARRAYS * 8 * max(n_cells, 1) + 3 * 8 * max(n_variants, 1)


def _value_range(values: np.ndarray) -> tuple[float, float]:
    """The lowest and highest value that is not NaN, without a copy; (inf,
    -inf) when every value is NaN."""
    low = float(np.fmin.reduce(values, axis=None, initial=np.inf))
    high = float(np.fmax.reduce(values, axis=None, initial=-np.inf))
    return low, high


def cosine_neighborhoods(
    values: np.ndarray, *, n_neighbors: int, block_rows: int
) -> np.ndarray:
    """Find each cell's neighbourhood by the cosine similarity of its values.

    ``values`` are square-root allele frequencies, NaN where a cell was not
    covered. The similarity of two cells is the cosine of their values over
    the variants that both cover; it is 0 when they share no covered variant
    or one of them shows none of the shared ones. A cell's neighbourhood is
    the cell itself and its ``n_neighbors - 1`` most similar other cells, as
    Seurat counts a cell among its own neighbours, ranked by similarity and
    then by position; cells of similarity 0 are never neighbours, so a
    neighbourhood can hold fewer cells. Every similarity is exact up to the
    rounding of matrix products, so the neighbourhoods do not depend on
    ``block_rows``.

    Args:
        values: Values shaped (cells, variants), non-negative, NaN where a
            cell was not covered.
        n_neighbors: The cells of a full neighbourhood, the cell included.
        block_rows: The cells whose similarities to every cell are computed
            at once.

    Returns:
        Each cell's neighbourhood shaped (cells, ``n_neighbors``): the cell
        itself, then its neighbours from the most similar, then -1.

    Raises:
        ValueError: If ``values`` is not a matrix of non-negative values, or
            ``n_neighbors`` or ``block_rows`` is below 1.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a matrix of cells by variants")
    low, high = _value_range(values)
    if low < 0 or high == np.inf:
        raise ValueError("values must be finite and non-negative, or NaN")
    if n_neighbors < 1 or block_rows < 1:
        raise ValueError("n_neighbors and block_rows must be at least 1")
    n_cells = values.shape[0]
    neighborhoods = np.full((n_cells, n_neighbors), -1, dtype=np.int64)
    neighborhoods[:, 0] = np.arange(n_cells)
    others = min(n_neighbors - 1, n_cells - 1)
    if others < 1:
        return neighborhoods
    covered = ~np.isnan(values)
    shown = np.where(covered, values, 0.0)
    weights = covered.astype(np.float64)
    del covered
    squares = shown * shown
    columns = np.arange(n_cells, dtype=np.int64)
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        # Products with the transposed arrays read them in place.
        similarity = _block_similarity(
            shown[start:stop],
            squares[start:stop],
            weights[start:stop],
            shown.T,
            squares.T,
            weights.T,
        )
        rows = np.arange(stop - start)
        # A cell is its own first neighbour, never one of the others.
        similarity[rows, rows + start] = 0.0
        neighborhoods[start:stop, 1 : others + 1] = _most_similar(
            similarity, others, columns
        )
    return neighborhoods


def _block_similarity(
    shown: np.ndarray,
    squares: np.ndarray,
    weights: np.ndarray,
    shown_t: np.ndarray,
    squares_t: np.ndarray,
    weights_t: np.ndarray,
) -> np.ndarray:
    """Cosine similarities of a block of cells to every cell, over the
    variants that both cover, rounded."""
    dot = shown @ shown_t
    # Each cell's squared norm over the variants that the other cell covers;
    # a variant that a cell does not cover holds 0 in ``shown``, so these sum
    # over the variants that both cover.
    own = squares @ weights_t
    other = weights @ squares_t
    own *= other
    del other
    np.sqrt(own, out=own)
    similarity: np.ndarray = np.zeros_like(dot)
    np.divide(dot, own, out=similarity, where=own > 0)
    del dot, own
    np.round(similarity, _SIMILARITY_DECIMALS, out=similarity)
    # Cosines of non-negative values lie in [0, 1]; rounding can overshoot.
    np.clip(similarity, 0.0, 1.0, out=similarity)
    return similarity


def _most_similar(
    similarity: np.ndarray, others: int, columns: np.ndarray
) -> np.ndarray:
    """Each row's ``others`` most similar positive columns, by decreasing
    similarity and then by position, padded with -1; ``others`` is below the
    columns, since a cell is never its own neighbour."""
    n_rows = similarity.shape[0]
    chosen = np.full((n_rows, others), -1, dtype=np.int64)
    # The value of each row's last neighbour; ties at it are taken from the
    # lowest positions.
    threshold = -np.partition(-similarity, others - 1, axis=1)[:, others - 1]
    above = similarity > threshold[:, None]
    tied = similarity == threshold[:, None]
    needed = others - above.sum(axis=1)
    tie_rank = np.cumsum(tied, axis=1)
    selected = (above | (tied & (tie_rank <= needed[:, None]))) & (similarity > 0)
    del above, tied, tie_rank
    rows, picked = np.nonzero(selected)
    del selected
    order = np.lexsort((columns[picked], -similarity[rows, picked], rows))
    rows = rows[order]
    picked = picked[order]
    starts = np.searchsorted(rows, np.arange(n_rows))
    slots = np.arange(len(rows)) - starts[rows]
    chosen[rows, slots] = picked
    return chosen


@dataclass(frozen=True, slots=True)
class FrequencyClones:
    """Clones called from allele frequencies.

    Attributes:
        clusters: Each cell's clone, numbered from 1 within its group by
            decreasing size, 0 for none.
        groups: Each cell's group, -1 for none.
        status: Why each cell has or lacks a clone, as a position in
            ``CLONE_STATUSES``.
        n_edges: The edges of the shared-neighbour graphs.
    """

    clusters: np.ndarray
    groups: np.ndarray
    status: np.ndarray
    n_edges: int


def frequency_clones(
    frequencies: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_neighbors: int,
    resolution: float,
    random_seed: int,
    memory_bytes: int,
    nthreads: int | None = None,
) -> FrequencyClones:
    """Call clones from allele frequencies, as Signac's ``FindClonotypes`` does.

    Within each group, each cell's neighbourhood holds itself and its most
    similar cells by :func:`cosine_neighborhoods` of the square roots of its
    frequencies. Two cells that share a neighbour are joined with the Jaccard
    index of their neighbourhoods, the shared cells over the cells in either,
    and joins below 1/15 are dropped, as in Seurat's ``FindNeighbors``. Cells
    that tie, such as cells that each show one variant, take the same few
    cells as neighbours and join every other such cell, so the graph grows
    with the square of their number. The Leiden algorithm partitions each
    group's graph at ``resolution``. A cell without coverage of any variant,
    without an alternative allele at any covered variant, or without a join,
    gets no clone.

    Args:
        frequencies: Allele frequencies shaped (cells, variants), NaN where
            a cell was not covered.
        groups: Each cell's group, -1 for a cell without one, or None for one
            group. Clones never join cells of two groups.
        n_neighbors: The cells of a full neighbourhood, the cell included.
        resolution: The Leiden resolution.
        random_seed: The Leiden seed.
        memory_bytes: The bytes that the call may hold beside its input.
        nthreads: Threads of the Leiden algorithm, None for Numba's count.

    Returns:
        The clones.

    Raises:
        ValueError: If the frequencies lie outside [0, 1] or the groups do not
            align with the cells.
        MemoryError: If a group's similarities, graph or partition do not fit
            ``memory_bytes``.
    """
    from scipy.sparse import coo_matrix

    from ..clustering.leiden import leiden_membership, native_leiden_peak_bytes

    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 2:
        raise ValueError("frequencies must be a matrix of cells by variants")
    low, high = _value_range(frequencies)
    if low < 0 or high > 1:
        raise ValueError("Allele frequencies must lie in [0, 1], or be NaN")
    n_cells, n_variants = frequencies.shape
    group_codes = (
        np.zeros(n_cells, dtype=np.int64)
        if groups is None
        else np.asarray(groups, dtype=np.int64)
    )
    if group_codes.shape != (n_cells,):
        raise ValueError("groups must hold one group per cell")
    budget = (
        f"the {memory_bytes} bytes that the memory budget leaves beside the "
        "allele frequencies"
    )
    cell_bytes = _CELL_BYTES * n_cells
    # Each cell's status reads a mask of a byte per frequency, one at a time.
    if cell_bytes + frequencies.size > memory_bytes:
        raise MemoryError(
            f"Checking the coverage of {n_cells} cells over {n_variants} variants "
            f"needs about {cell_bytes + frequencies.size} bytes, more than {budget}. "
            "Raise mem_budget."
        )
    status = np.zeros(n_cells, dtype=np.int8)
    covered = ~np.isnan(frequencies).all(axis=1)
    shown = (frequencies > 0).any(axis=1)
    status[~covered] = CLONE_STATUSES.index("no_coverage")
    status[covered & ~shown] = CLONE_STATUSES.index("no_alternative")
    status[group_codes < 0] = CLONE_STATUSES.index("no_group")
    del covered, shown
    clusters = np.zeros(n_cells, dtype=np.int64)
    n_edges = 0
    for group in np.unique(group_codes[status == 0]).tolist():
        members = np.flatnonzero((group_codes == group) & (status == 0))
        n_members = len(members)
        per_row = neighborhood_bytes_per_cell(n_members, n_variants)
        held = (
            cell_bytes
            + _VALUE_BYTES * n_members * n_variants
            + 8 * n_neighbors * n_members
        )
        block_rows = int((memory_bytes - held) // per_row)
        if block_rows < 1:
            raise MemoryError(
                f"Comparing the allele frequencies of {n_members} cells over "
                f"{n_variants} variants needs about {held + per_row} bytes, more "
                f"than {budget}. Raise mem_budget, or call clones within smaller "
                "groups."
            )
        values = frequencies[members]
        np.sqrt(values, out=values)
        neighborhoods = cosine_neighborhoods(
            values, n_neighbors=n_neighbors, block_rows=min(block_rows, n_members)
        )
        del values
        graph = _shared_neighbor_graph(
            neighborhoods, memory_bytes=memory_bytes - cell_bytes, budget=budget
        )
        del neighborhoods
        degree = np.bincount(graph.row, minlength=n_members) + np.bincount(
            graph.col, minlength=n_members
        )
        joined = degree > 0
        status[members[~joined]] = CLONE_STATUSES.index("no_neighbor")
        if not joined.any():
            continue
        kept = np.flatnonzero(joined)
        renumber = np.full(n_members, -1, dtype=np.int64)
        renumber[kept] = np.arange(len(kept))
        subgraph = coo_matrix(
            (graph.data, (renumber[graph.row], renumber[graph.col])),
            shape=(len(kept), len(kept)),
        ).tocsr()
        del graph, renumber
        n_edges += int(subgraph.nnz)
        leiden_bytes = (
            cell_bytes
            + native_leiden_peak_bytes(len(kept), 2 * int(subgraph.nnz))
            + _CSR_ENTRY_BYTES * int(subgraph.nnz)
            + 8 * (len(kept) + 1)
        )
        if leiden_bytes > memory_bytes:
            raise MemoryError(
                f"Partitioning the shared-neighbour graph of {len(kept)} cells needs "
                f"about {leiden_bytes} bytes, more than {budget}. Raise mem_budget."
            )
        labels = leiden_membership(subgraph, resolution, random_seed, nthreads=nthreads)
        clusters[members[kept]] = np.asarray(labels, dtype=np.int64)
    return FrequencyClones(
        clusters=clusters, groups=group_codes, status=status, n_edges=n_edges
    )


def _shared_neighbor_graph(
    neighborhoods: np.ndarray, *, memory_bytes: int, budget: str
) -> "coo_matrix":
    """Seurat's shared-neighbour graph of a group's neighbourhoods, built in
    blocks of candidate joins within ``memory_bytes``.

    A quarter of the room beside the neighbourhoods holds a block of
    candidate joins, at least the candidates of the cell with the most, and
    the rest the kept joins; the graph does not depend on the split.
    """
    from ..neighbors.graph import shared_neighbor_bounds, shared_neighbor_graph

    n_members = neighborhoods.shape[0]
    fixed = _SLOT_BYTES * neighborhoods.size
    largest = int(shared_neighbor_bounds(neighborhoods).max(initial=0))
    room = memory_bytes - fixed
    max_entries = max(largest, room // (4 * _CANDIDATE_BYTES), 1)
    max_joins = (room - _CANDIDATE_BYTES * max_entries) // _JOIN_BYTES
    if max_joins < 1:
        raise MemoryError(
            f"The shared-neighbour graph of {n_members} cells needs about "
            f"{fixed + _CANDIDATE_BYTES * max_entries + _JOIN_BYTES} bytes before it "
            f"holds a join, more than {budget}. Raise mem_budget, lower "
            "n_neighbors, or call clones within smaller groups."
        )
    try:
        return shared_neighbor_graph(
            neighborhoods, max_joins=int(max_joins), max_entries=int(max_entries)
        )
    except MemoryError as error:
        raise MemoryError(
            f"The shared-neighbour graph of {n_members} cells does not fit {budget}: "
            f"{error}. Cells that tie, such as cells that each show one informative "
            "variant, take the same few neighbours and join every other such cell. "
            "Raise mem_budget, or call clones within smaller groups."
        ) from error


def clone_status_counts(status: np.ndarray) -> dict[str, int]:
    """The cells of each status of ``CLONE_STATUSES``, by name."""
    counts = np.bincount(
        np.asarray(status, dtype=np.int64), minlength=len(CLONE_STATUSES)
    )
    return {
        name: int(count) for name, count in zip(CLONE_STATUSES, counts, strict=True)
    }
