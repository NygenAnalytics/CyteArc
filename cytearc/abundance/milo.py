"""Neighbourhood differential abundance, ported from miloR 2.6.0.

The functions here are the storage-free steps of ``makeNhoods``,
``countCells``, ``testNhoods``, ``graphSpatialFDR``, and ``annotateNhoods``.
They work on blocks of a k-nearest-neighbour graph whose rows exclude the
cell itself, so a neighbourhood is built without a cells x neighbourhoods
matrix in memory:

- A neighbourhood is miloR's ``neighborhood(graph, order=1)`` of an index
  cell on the undirected kNN graph: the index cell, its k nearest
  neighbours, and every cell that lists the index cell among its own k
  nearest neighbours.
- Membership is stored cell-major, as miloR's ``nhoods(x)`` is cells x
  neighbourhoods.
- Spatial FDR weights each neighbourhood by the inverse Euclidean distance
  from its index cell to its k-th nearest neighbour (``k-distance``) and
  applies cydar's weighted Benjamini-Hochberg correction.

The edgeR steps of ``testNhoods`` run in edgePython, a Python port of edgeR
distributed under the GPL-3.0 as the optional ``milo`` extra, and are
imported only when a neighbourhood test runs.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..stats.multiple_testing import p_adjust_bh, weighted_bh

__all__ = [
    "NeighborhoodAnnotation",
    "NeighborhoodTestFit",
    "ReverseNeighbors",
    "annotate_neighborhoods",
    "block_membership",
    "count_block",
    "differential_abundance",
    "kth_distances",
    "refine_index_cells",
    "reverse_neighbors",
    "sample_index_cells",
    "spatial_fdr",
    "unique_in_order",
]


@dataclass(frozen=True, slots=True, eq=False)
class ReverseNeighbors:
    """Cells that index cells list as neighbours, sorted by cell.

    ``targets[i]`` is a cell among the k nearest neighbours of the index cell
    of neighbourhood ``nhoods[i]``.
    """

    targets: np.ndarray
    nhoods: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class NeighborhoodAnnotation:
    """``annotateNhoods`` output: label counts and the majority label."""

    levels: tuple[str, ...]
    counts: np.ndarray
    label: np.ndarray
    fraction: np.ndarray


def unique_in_order(values: np.ndarray) -> np.ndarray:
    """Return distinct values in order of first appearance, as R's ``unique``."""
    array = np.asarray(values)
    _, first = np.unique(array, return_index=True)
    return array[np.sort(first)]


def sample_index_cells(n_cells: int, prop: float, *, seed: int) -> np.ndarray:
    """Draw ``floor(prop * n_cells)`` distinct cells, as ``.sample_vertices`` does.

    Raises:
        ValueError: If ``prop`` is not in ``(0, 1]`` or no cell is drawn.
    """
    if not 0 < prop <= 1:
        raise ValueError("prop must be in (0, 1]")
    size = int(np.floor(prop * n_cells))
    if size < 1:
        raise ValueError(
            f"prop={prop} samples no index cells from {n_cells} cells; raise prop"
        )
    rng = np.random.default_rng(seed)
    return rng.choice(n_cells, size=size, replace=False).astype(np.int64)


def refine_index_cells(
    sampled: np.ndarray,
    neighbor_rows: np.ndarray,
    coordinates: Callable[[np.ndarray], np.ndarray],
    nearest: Callable[[np.ndarray], np.ndarray],
    *,
    block_size: int = 4096,
) -> np.ndarray:
    """Move each sampled cell to the cell nearest its neighbours' median.

    This is miloR's ``.refined_sampling`` with ``refinement_scheme =
    "reduced_dim"``: the coordinate-wise median of a sampled cell's k nearest
    neighbours (itself excluded) is computed, the nearest cell to that point
    becomes the index cell, and duplicates are dropped in order.

    Args:
        sampled: Sampled cells.
        neighbor_rows: The k nearest neighbours of each sampled cell.
        coordinates: Returns the coordinates of the given cells.
        nearest: Returns the nearest cell to each given point.
        block_size: Sampled cells refined per block.
    """
    cells = np.asarray(sampled, dtype=np.int64)
    rows = np.asarray(neighbor_rows, dtype=np.int64)
    if rows.shape[0] != cells.size:
        raise ValueError("neighbor_rows must have one row per sampled cell")
    refined = np.empty(cells.size, dtype=np.int64)
    k = rows.shape[1]
    for start in range(0, cells.size, block_size):
        stop = min(start + block_size, cells.size)
        block = rows[start:stop]
        flat = block.ravel()
        unique_cells, inverse = np.unique(flat, return_inverse=True)
        points = np.asarray(coordinates(unique_cells), dtype=np.float64)
        neighbour_points = points[inverse.ravel()].reshape(stop - start, k, -1)
        medians = np.median(neighbour_points, axis=1)
        refined[start:stop] = np.asarray(nearest(medians), dtype=np.int64)
    return unique_in_order(refined)


def reverse_neighbors(
    index_cells: np.ndarray, index_rows: np.ndarray
) -> ReverseNeighbors:
    """Pair each index cell's neighbours with its neighbourhood, sorted by cell."""
    rows = np.asarray(index_rows, dtype=np.int64)
    nhoods = np.repeat(np.arange(len(index_cells), dtype=np.int32), rows.shape[1])
    targets = rows.ravel()
    order = np.argsort(targets, kind="stable")
    return ReverseNeighbors(targets=targets[order], nhoods=nhoods[order])


def block_membership(
    start: int,
    neighbor_block: np.ndarray,
    index_position: np.ndarray,
    reverse: ReverseNeighbors,
    n_nhoods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the neighbourhoods of the cells ``start:start + len(block)``.

    Args:
        start: First cell of the block.
        neighbor_block: The k nearest neighbours of each cell of the block.
        index_position: The neighbourhood that each cell indexes, or -1.
        reverse: Neighbours of the index cells.
        n_nhoods: Number of neighbourhoods.

    Returns:
        The number of neighbourhoods of each cell, and their ids, sorted
        within each cell.
    """
    block = np.asarray(neighbor_block, dtype=np.int64)
    n_rows = block.shape[0]
    stop = start + n_rows
    local = np.arange(n_rows, dtype=np.int64)
    own = index_position[start:stop]
    own_rows = local[own >= 0]
    own_nhoods = own[own >= 0].astype(np.int64)
    out = index_position[block]
    out_rows, out_cols = np.nonzero(out >= 0)
    out_nhoods = out[out_rows, out_cols].astype(np.int64)
    lo, hi = np.searchsorted(reverse.targets, [start, stop])
    in_rows = reverse.targets[lo:hi] - start
    in_nhoods = reverse.nhoods[lo:hi].astype(np.int64)
    rows = np.concatenate([own_rows, out_rows, in_rows])
    nhoods = np.concatenate([own_nhoods, out_nhoods, in_nhoods])
    width = max(n_nhoods, 1)
    keys = np.unique(rows * width + nhoods)
    counts = np.bincount(keys // width, minlength=n_rows).astype(np.int64)
    return counts, (keys % width).astype(np.int32)


def kth_distances(index_points: np.ndarray, neighbour_points: np.ndarray) -> np.ndarray:
    """Euclidean distance from each index cell to its farthest stored neighbour."""
    centre = np.asarray(index_points, dtype=np.float64)[:, None, :]
    neighbours = np.asarray(neighbour_points, dtype=np.float64)
    return np.asarray(np.sqrt(np.sum((neighbours - centre) ** 2, axis=2)).max(axis=1))


def count_block(
    counts: np.ndarray,
    row_counts: np.ndarray,
    nhood_ids: np.ndarray,
    codes: np.ndarray,
) -> None:
    """Add the cells of one membership block to ``counts`` (neighbourhoods x groups).

    Cells whose code is negative are left out.
    """
    row_codes = np.repeat(np.asarray(codes, dtype=np.int64), row_counts)
    keep = row_codes >= 0
    n_groups = counts.shape[1]
    keys = nhood_ids[keep].astype(np.int64) * n_groups + row_codes[keep]
    counts += np.bincount(keys, minlength=counts.size).reshape(counts.shape)


def spatial_fdr(p_values: np.ndarray, kth_distance: np.ndarray) -> np.ndarray:
    """``graphSpatialFDR`` with ``weighting = "k-distance"``."""
    with np.errstate(divide="ignore"):
        weights = 1 / np.asarray(kth_distance, dtype=np.float64)
    weights[np.isinf(weights)] = 1
    return weighted_bh(p_values, weights)


def annotate_neighborhoods(
    label_counts: np.ndarray, levels: tuple[str, ...]
) -> NeighborhoodAnnotation:
    """Return ``annotateNhoods``'s majority label and its fraction per neighbourhood.

    Args:
        label_counts: Neighbourhoods x labels cell counts.
        levels: Label order; the first of tied labels wins, as ``which.max``.
    """
    counts = np.asarray(label_counts)
    best = np.argmax(counts, axis=1)
    totals = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = counts[np.arange(counts.shape[0]), best] / totals
    return NeighborhoodAnnotation(
        levels=levels,
        counts=counts,
        label=np.asarray([levels[i] for i in best], dtype=object),
        fraction=fraction,
    )


@dataclass(frozen=True, slots=True, eq=False)
class NeighborhoodTestFit:
    """``testNhoods`` statistics for the tested neighbourhoods.

    Arrays over neighbourhoods hold ``NaN`` where ``tested`` is ``False``.
    ``trended_dispersion`` and ``ave_log_cpm`` cover the tested rows only.
    """

    tested: np.ndarray
    log_fc: np.ndarray
    log_cpm: np.ndarray
    f_statistic: np.ndarray
    p_value: np.ndarray
    fdr: np.ndarray
    spatial_fdr: np.ndarray
    lib_size: np.ndarray
    norm_factors: np.ndarray
    common_dispersion: float
    trended_dispersion: np.ndarray
    df_prior: np.ndarray
    s2_prior: np.ndarray
    average_ql_dispersion: float | None


def differential_abundance(
    counts: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
    *,
    kth_distance: np.ndarray | None,
    norm_method: str = "TMM",
    min_mean: float = 0.0,
    robust: bool = True,
    legacy: bool = True,
) -> NeighborhoodTestFit:
    """Test neighbourhood x sample counts as ``testNhoods`` does with edgeR.

    Args:
        counts: Neighbourhoods x samples cell counts.
        design: Samples x coefficients design matrix.
        contrast: Weights of the tested contrast over the design columns.
        kth_distance: k-distance of each neighbourhood for spatial FDR, or
            ``None`` to skip spatial FDR (``fdr.weighting = "none"``).
        norm_method: ``"TMM"``, ``"RLE"``, or ``"logMS"`` (library sizes only).
        min_mean: Test neighbourhoods whose mean count is at least this.
        robust: Robust quasi-likelihood moderation.
        legacy: edgeR's legacy quasi-likelihood method, which miloR uses.

    Raises:
        ValueError: If no neighbourhood passes ``min_mean``.
    """
    values = np.asarray(counts, dtype=np.float64)
    n_nhoods = values.shape[0]
    tested = (
        values.mean(axis=1) >= min_mean if min_mean > 0 else np.ones(n_nhoods, bool)
    )
    if not np.any(tested):
        raise ValueError(f"No neighbourhood has a mean count of at least {min_mean}")
    kept = values[tested]
    lib_size = kept.sum(axis=0)
    if np.any(lib_size == 0):
        raise ValueError("Every sample needs cells in the tested neighbourhoods")
    if norm_method not in ("TMM", "RLE", "logMS"):
        raise ValueError("norm_method must be 'TMM', 'RLE', or 'logMS'")
    try:
        import edgepython as ep
    except ImportError as error:
        raise ImportError(
            "Testing Milo neighbourhoods needs edgePython; install it with "
            'pip install "cytearc[milo]"'
        ) from error
    # testNhoods: DGEList, normalization factors, estimateDisp, then glmQLFit
    # and glmQLFTest on the DGEList, with library sizes of the tested rows.
    dge = ep.make_dgelist(kept, lib_size=lib_size)
    if norm_method != "logMS":
        dge = ep.calc_norm_factors(dge, method=norm_method)
    factors = np.asarray(dge["samples"]["norm.factors"], dtype=np.float64)
    dge = ep.estimate_disp(dge, design=design, tagwise=False)
    fit = ep.glm_ql_fit(dge, design=design, robust=robust, legacy=legacy)
    table = ep.glm_ql_ftest(fit, contrast=contrast)["table"]
    p_value = np.asarray(table["PValue"], dtype=np.float64)

    def expand(row_values: np.ndarray) -> np.ndarray:
        full = np.full(n_nhoods, np.nan)
        full[tested] = row_values
        return full

    spatial = (
        np.full(int(tested.sum()), np.nan)
        if kth_distance is None
        else spatial_fdr(p_value, np.asarray(kth_distance)[tested])
    )
    return NeighborhoodTestFit(
        tested=tested,
        log_fc=expand(np.asarray(table["logFC"], dtype=np.float64)),
        log_cpm=expand(np.asarray(table["logCPM"], dtype=np.float64)),
        f_statistic=expand(np.asarray(table["F"], dtype=np.float64)),
        p_value=expand(p_value),
        fdr=expand(p_adjust_bh(p_value)),
        spatial_fdr=expand(spatial),
        lib_size=lib_size,
        norm_factors=factors,
        common_dispersion=float(dge["common.dispersion"]),
        trended_dispersion=np.asarray(dge["trended.dispersion"], dtype=np.float64),
        df_prior=np.asarray(fit["df.prior"], dtype=np.float64),
        s2_prior=np.asarray(fit["s2.prior"], dtype=np.float64),
        average_ql_dispersion=(None if legacy else float(fit["average.ql.dispersion"])),
    )
