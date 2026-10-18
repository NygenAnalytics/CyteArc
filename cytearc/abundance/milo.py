"""Storage-free steps of Milo's neighbourhood differential abundance test.

Milo (Dann et al. 2022) samples index cells on a kNN graph, refines each to
the cell nearest the median position of its neighbours, and defines the
neighbourhood of an index cell as its order-1 neighbourhood on the
undirected graph: the index cell, its k nearest neighbours, and every cell
that lists it among its own k nearest neighbours. Cells are counted per
neighbourhood and sample, each neighbourhood is tested with a negative
binomial quasi-likelihood F-test, and the p-values are corrected with a
spatial FDR that weights each neighbourhood by the inverse of the distance
from its index cell to its k-th neighbour (Lun, Richard and Marioni 2017).

These functions work on blocks of a stored kNN graph whose rows exclude the
cell itself, so no cells x neighbourhoods matrix is ever built; membership
is produced cell-major. The negative binomial steps run in the optional
edgePython package. The module was implemented from those papers and
miloR's documentation; agreement with miloR is checked by comparing
outputs.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
    """The k nearest neighbours of every index cell, sorted by cell.

    ``targets[i]`` is one of the k nearest neighbours of the index cell of
    neighbourhood ``nhoods[i]``.
    """

    targets: np.ndarray
    nhoods: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class NeighborhoodAnnotation:
    """The majority label of each neighbourhood.

    ``counts`` is neighbourhoods x ``levels``; ``fraction`` is the majority
    label's share of the neighbourhood's labelled cells.
    """

    levels: tuple[str, ...]
    counts: np.ndarray
    label: np.ndarray
    fraction: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class NeighborhoodTestFit:
    """Quasi-likelihood tests of neighbourhood counts.

    Arrays over neighbourhoods hold NaN where ``tested`` is False;
    ``trended_dispersion`` covers the tested neighbourhoods only. ``log_fc``
    is a log2 fold change and ``lib_size`` and ``norm_factors`` hold one
    value per sample.
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


def unique_in_order(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return the distinct values in order of first appearance."""
    array = np.asarray(values)
    _, first = np.unique(array, return_index=True)
    return array[np.sort(first)]


def sample_index_cells(n_cells: int, prop: float, *, seed: int) -> np.ndarray:
    """Draw ``floor(prop * n_cells)`` distinct candidate index cells.

    Raises:
        ValueError: If ``prop`` is not in ``(0, 1]`` or no cell is drawn.
    """
    if not 0 < float(prop) <= 1:
        raise ValueError("prop must be in (0, 1]")
    size = int(np.floor(float(prop) * int(n_cells)))
    if size < 1:
        raise ValueError(
            f"prop={prop} draws no index cell from {n_cells} cells; increase prop"
        )
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(int(n_cells), size=size, replace=False), np.int64)


def refine_index_cells(
    sampled: np.ndarray,
    neighbor_rows: np.ndarray,
    coordinates: Callable[[np.ndarray], np.ndarray],
    nearest: Callable[[np.ndarray], np.ndarray],
    *,
    block_size: int = 4096,
) -> np.ndarray:
    """Move each sampled cell to the cell nearest its neighbours' median.

    ``neighbor_rows[i]`` lists the k nearest neighbours of ``sampled[i]``,
    itself excluded. The coordinate-wise median of their coordinates is
    taken, ``nearest`` maps it to a cell, and repeated cells are dropped,
    keeping first occurrences in order.

    Args:
        sampled: Sampled cells.
        neighbor_rows: Their k nearest neighbours, one row per sampled cell.
        coordinates: Returns the coordinates of the given cells.
        nearest: Returns the nearest cell to each point.
        block_size: Sampled cells refined per block.
    """
    cells = np.asarray(sampled, dtype=np.int64)
    rows = np.asarray(neighbor_rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[0] != cells.size:
        raise ValueError("neighbor_rows must have one row per sampled cell")
    refined = np.empty(cells.size, dtype=np.int64)
    for start in range(0, cells.size, block_size):
        block = rows[start : start + block_size]
        points = np.asarray(coordinates(block.ravel()), dtype=np.float64)
        medians = np.median(points.reshape(*block.shape, -1), axis=1)
        refined[start : start + block.shape[0]] = np.asarray(nearest(medians))
    return np.asarray(unique_in_order(refined), dtype=np.int64)


def reverse_neighbors(
    index_cells: np.ndarray, index_rows: np.ndarray
) -> ReverseNeighbors:
    """Invert the index cells' neighbour lists.

    ``index_rows[j]`` lists the k nearest neighbours of ``index_cells[j]``,
    the index cell of neighbourhood ``j``.
    """
    rows = np.asarray(index_rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[0] != np.asarray(index_cells).size:
        raise ValueError("index_rows must have one row per index cell")
    targets = rows.ravel()
    nhoods = np.repeat(np.arange(rows.shape[0], dtype=np.int64), rows.shape[1])
    order = np.lexsort((nhoods, targets))
    return ReverseNeighbors(targets=targets[order], nhoods=nhoods[order])


def block_membership(
    start: int,
    neighbor_block: np.ndarray,
    index_position: np.ndarray,
    reverse: ReverseNeighbors,
    n_nhoods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the neighbourhoods of cells ``start:start + len(neighbor_block)``.

    A cell belongs to the neighbourhood it indexes, to the neighbourhood of
    every index cell among its own k nearest neighbours, and to the
    neighbourhood of every index cell that lists it among theirs.

    Args:
        start: First cell of the block.
        neighbor_block: The block's k nearest neighbours, one row per cell.
        index_position: The neighbourhood each cell indexes, or -1.
        reverse: Inverted neighbour lists of the index cells.
        n_nhoods: Number of neighbourhoods.

    Returns:
        Each cell's number of neighbourhoods, and their ids sorted within
        each cell.
    """
    block = np.asarray(neighbor_block, dtype=np.int64)
    n_rows = block.shape[0]
    stop = start + n_rows
    positions = np.asarray(index_position)

    own = positions[start:stop].astype(np.int64)
    own_cells = np.flatnonzero(own >= 0)
    listed = positions[block].astype(np.int64)
    listed_cells, listed_slots = np.nonzero(listed >= 0)
    low, high = np.searchsorted(reverse.targets, [start, stop])
    listing_cells = reverse.targets[low:high] - start

    cells = np.concatenate([own_cells, listed_cells, listing_cells])
    nhoods = np.concatenate(
        [own[own_cells], listed[listed_cells, listed_slots], reverse.nhoods[low:high]]
    )
    keys = np.unique(cells.astype(np.int64) * int(n_nhoods) + nhoods)
    member_cells = keys // int(n_nhoods)
    row_counts = np.bincount(member_cells, minlength=n_rows).astype(np.int64)
    return row_counts, (keys % int(n_nhoods)).astype(np.int32)


def kth_distances(index_points: np.ndarray, neighbour_points: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance from each index cell to its farthest neighbour.

    ``index_points`` is index cells x dimensions and ``neighbour_points``
    index cells x k x dimensions.
    """
    centres = np.asarray(index_points, dtype=np.float64)
    neighbours = np.asarray(neighbour_points, dtype=np.float64)
    squared = ((neighbours - centres[:, None, :]) ** 2).sum(axis=2)
    return np.sqrt(squared.max(axis=1))


def count_block(
    counts: np.ndarray, row_counts: np.ndarray, nhood_ids: np.ndarray, codes: np.ndarray
) -> None:
    """Add one membership block's cells to ``counts`` in place.

    ``counts`` is neighbourhoods x groups and ``codes`` holds each cell's
    group, with negative codes left out.
    """
    cells = np.repeat(np.arange(np.asarray(row_counts).size), row_counts)
    groups = np.asarray(codes, dtype=np.int64)[cells]
    keep = groups >= 0
    flat = np.asarray(nhood_ids, dtype=np.int64)[keep] * counts.shape[1] + groups[keep]
    counts += (
        np.bincount(flat, minlength=counts.size)
        .reshape(counts.shape)
        .astype(counts.dtype, copy=False)
    )


def spatial_fdr(p_values: np.ndarray, kth_distance: np.ndarray) -> np.ndarray:
    """Weighted BH over the tested neighbourhoods, weighted by inverse k-distance.

    Neighbourhoods with a missing p-value are not tested and take no part
    in the correction. miloR instead weights over every neighbourhood even
    when some are untested.
    """
    values = np.asarray(p_values, dtype=np.float64)
    distances = np.asarray(kth_distance, dtype=np.float64)
    if distances.shape != values.shape:
        raise ValueError("kth_distance must have one value per neighbourhood")
    weights = np.ones_like(distances)
    tested = ~np.isnan(values)
    weights[tested] = 1 / distances[tested]
    return weighted_bh(values, weights)


def annotate_neighborhoods(
    label_counts: np.ndarray, levels: Sequence[str]
) -> NeighborhoodAnnotation:
    """Return each neighbourhood's most frequent label and its share.

    The first of tied labels wins. Neighbourhoods without labelled cells
    get no label and a missing fraction.
    """
    counts = np.asarray(label_counts)
    names = tuple(str(level) for level in levels)
    if counts.ndim != 2 or counts.shape[1] != len(names):
        raise ValueError("label_counts must have one column per level")
    totals = counts.sum(axis=1)
    label = np.full(counts.shape[0], None, dtype=object)
    fraction = np.full(counts.shape[0], np.nan)
    if names:
        best = np.argmax(counts, axis=1)
        labelled = totals > 0
        label[labelled] = np.asarray(names, dtype=object)[best[labelled]]
        fraction[labelled] = (
            counts[np.flatnonzero(labelled), best[labelled]] / totals[labelled]
        )
    return NeighborhoodAnnotation(
        levels=names, counts=counts, label=label, fraction=fraction
    )


def _edgepython() -> Any:
    try:
        import edgepython
    except ImportError as error:
        raise ImportError(
            "Milo's neighbourhood test needs edgePython. Install it with "
            "pip install 'cytearc[milo]'."
        ) from error
    return edgepython


def _expand(values: Any, tested: np.ndarray) -> np.ndarray:
    full = np.full(tested.size, np.nan)
    full[tested] = np.broadcast_to(
        np.asarray(values, dtype=np.float64), (int(tested.sum()),)
    )
    return full


def differential_abundance(
    counts: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
    *,
    kth_distance: np.ndarray | None,
    norm_method: Literal["TMM", "RLE", "logMS"] = "TMM",
    min_mean: float = 0.0,
    robust: bool = True,
    legacy: bool = True,
) -> NeighborhoodTestFit:
    """Test neighbourhood x sample counts for differential abundance.

    Neighbourhoods whose mean count is at least ``min_mean`` are tested.
    Library sizes are the tested neighbourhoods' column sums, normalized by
    TMM or RLE factors (``"logMS"`` keeps the library sizes alone). Negative
    binomial dispersions are estimated on ``design``, quasi-likelihood GLMs
    are fitted with ``robust`` and ``legacy``, and ``contrast`` is tested
    with the quasi-likelihood F-test, all in edgePython. ``fdr`` is the
    Benjamini-Hochberg correction and ``spatial_fdr`` that of
    :func:`spatial_fdr`, missing when ``kth_distance`` is None.

    Raises:
        ImportError: If edgePython is not installed.
        ValueError: If shapes disagree or no neighbourhood passes
            ``min_mean``.
    """
    edgepython = _edgepython()
    values = np.asarray(counts, dtype=np.float64)
    matrix = np.asarray(design, dtype=np.float64)
    weights = np.asarray(contrast, dtype=np.float64)
    if values.ndim != 2 or matrix.ndim != 2 or matrix.shape[0] != values.shape[1]:
        raise ValueError("design must have one row per sample of counts")
    if weights.shape != (matrix.shape[1],):
        raise ValueError("contrast must have one weight per design column")
    if norm_method not in ("TMM", "RLE", "logMS"):
        raise ValueError("norm_method must be 'TMM', 'RLE', or 'logMS'")
    tested = values.mean(axis=1) >= float(min_mean)
    if not tested.any():
        raise ValueError(f"No neighbourhood has a mean count of at least {min_mean}")
    kept = values[tested]
    dge = edgepython.make_dgelist(kept, lib_size=kept.sum(axis=0))
    if norm_method != "logMS":
        dge = edgepython.calc_norm_factors(dge, method=norm_method)
    dge = edgepython.estimate_disp(dge, matrix)
    fit = edgepython.glm_ql_fit(dge, matrix, robust=bool(robust), legacy=bool(legacy))
    test = edgepython.glm_ql_ftest(fit, contrast=weights)
    table = test["table"]
    p_value = _expand(table["PValue"].to_numpy(), tested)
    average = fit.get("average.ql.dispersion")
    return NeighborhoodTestFit(
        tested=tested,
        log_fc=_expand(table["logFC"].to_numpy(), tested),
        log_cpm=_expand(table["logCPM"].to_numpy(), tested),
        f_statistic=_expand(table["F"].to_numpy(), tested),
        p_value=p_value,
        fdr=p_adjust_bh(p_value),
        spatial_fdr=(
            np.full(tested.size, np.nan)
            if kth_distance is None
            else spatial_fdr(p_value, kth_distance)
        ),
        lib_size=np.asarray(dge["samples"]["lib.size"], dtype=np.float64),
        norm_factors=np.asarray(dge["samples"]["norm.factors"], dtype=np.float64),
        common_dispersion=float(dge["common.dispersion"]),
        trended_dispersion=np.asarray(dge["trended.dispersion"], dtype=np.float64),
        df_prior=_expand(fit["df.prior"], tested),
        s2_prior=_expand(fit["s2.prior"], tested),
        average_ql_dispersion=None if average is None else float(average),
    )
