"""Which groups of cells share clones, and which fates the clones of early cells took.

Lineage tracing studies ask two questions of clones. Fate coupling: cell types
whose cells often share a clone descend from common progenitors, so the clones
that two groups share measure how closely their fates are coupled. Fate bias:
the cells of an early cell's clone sampled later show which fates its
progenitor took. CoSpar (Wang et al. 2022) defines both from each clone's
cells in each group, which :func:`~cytearc.clones.overlap.group_clone_matrix`
counts, and these functions compute them from those counts.

Coupling is compared with a null that shuffles the group labels of the cells
with a clone, which keeps each clone's cells and each group's cells, as
CoSpar's ``pvalue_for_fate_coupling`` does. A compiled kernel shuffles and
measures each permutation from a random stream keyed by the seed and the
permutation, so the null does not depend on threads. These functions work on
codes and never touch storage.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import prange

from ..storage.refs import ArtifactRef
from ..utils.numba import njit, restore_numba_threads
from .overlap import group_clone_matrix

__all__ = [
    "COUPLING_METHODS",
    "FATE_CELL_COLUMNS",
    "CloneCoupling",
    "CloneFateBias",
    "FateBiasArrays",
    "clone_coupling",
    "coupling_bytes",
    "coupling_linkage",
    "coupling_plan",
    "dominant_fates",
    "fate_bias",
    "fate_bias_bytes",
    "fate_bias_tables",
    "permuted_labels",
]

#: The coupling methods, CoSpar's default first. ``share_cosine`` is CoSpar's
#: ``method="SW"`` with ``normalize=True``, ``abundance_cosine`` its ``"SW"``
#: with ``normalize=False``, ``normalized_covariance`` its ``"Weinreb"`` with
#: ``normalize=False`` and ``jaccard`` its ``"Jaccard"``.
COUPLING_METHODS = (
    "share_cosine",
    "abundance_cosine",
    "normalized_covariance",
    "jaccard",
)
_SHARE_COSINE, _ABUNDANCE_COSINE, _NORMALIZED_COVARIANCE, _JACCARD = range(4)

# SplitMix64 (Steele, Lea and Flood 2014): the finaliser of a key plus a
# counter gives each draw, so a draw depends only on its stream and counter.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_SALT = np.uint64(0x2545F4914F6CDD1D)
_SHIFT_11 = np.uint64(11)
_SHIFT_27 = np.uint64(27)
_SHIFT_30 = np.uint64(30)
_SHIFT_31 = np.uint64(31)
_UNIT_INTERVAL = 2.0**-53

# Bytes that measuring coupling holds, besides the clones read and their
# counts in each group: for each pair of groups its positions, coupling,
# null and table; for each product of two groups the coupling matrix and the
# distances of the tree; for each clone counted in a group the matrices of
# counts, 64 bytes at their peak as traced on a million counts; for
# each cell with a clone its label; and for each worker of the null its
# shuffled labels, its products and its counts of each group.
_COUPLING_BYTES_PER_PAIR = 250
_COUPLING_BYTES_PER_PRODUCT = 40
_COUPLING_BYTES_PER_ENTRY = 80
_COUPLING_BYTES_PER_CELL = 12
_WORKER_BYTES_PER_CELL = 4
_WORKER_BYTES_PER_PRODUCT = 8
_WORKER_BYTES_PER_GROUP = 32
_BATCH_BYTES_PER_VALUE = 8
# The permutations measured at once hold at most this many bytes beyond one
# per worker, since more save little time.
_BATCH_BYTES = 64 * 1024**2
# Bytes that measuring fate bias holds for each early cell, besides its
# fractions, for each of its fractions, the value and its copy as it is
# written, for each fate of each clone with late cells, and for each fate.
_BIAS_BYTES_PER_EARLY_CELL = 64
_BIAS_BYTES_PER_FRACTION = 24
_BIAS_BYTES_PER_CLONE_FATE = 80
_BIAS_BYTES_PER_FATE = 1_000


def _checked_method(method: Any) -> int:
    if not isinstance(method, str) or method not in COUPLING_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(COUPLING_METHODS)}, not {method!r}"
        )
    return COUPLING_METHODS.index(method)


@njit(cache=True, nogil=True)
def _mix64(value: np.uint64) -> np.uint64:
    """Return the SplitMix64 finaliser of a 64-bit unsigned integer."""
    mixed = (value ^ (value >> _SHIFT_30)) * _MIX_1
    mixed = (mixed ^ (mixed >> _SHIFT_27)) * _MIX_2
    return mixed ^ (mixed >> _SHIFT_31)


def _seed_key(seed: int) -> np.uint64:
    """The key of a seed's streams."""
    return np.uint64(int(_mix64(np.uint64(int(seed) % 2**64) ^ _SALT)))


@njit(cache=True, nogil=True)
def _permutation_key(seed_key: np.uint64, permutation: np.int64) -> np.uint64:
    """The key of one permutation's stream."""
    return _mix64(seed_key ^ _mix64(np.uint64(permutation) + _GOLDEN))


@njit(cache=True, nogil=True)
def _shuffle_within(labels: np.ndarray, bounds: np.ndarray, key: np.uint64) -> None:
    """Shuffle the labels of each stratum, ``bounds[s]`` to ``bounds[s + 1]``.

    Fisher and Yates's shuffle draws, for each position from the last, a
    position at or before it within its stratum. The draw for position ``i``
    is the uniform ``i`` of the stream ``key``: 53 random bits make a number
    below 1, whose product with the positions to choose from rounds below
    their count.
    """
    for stratum in range(len(bounds) - 1):
        start = bounds[stratum]
        for position in range(bounds[stratum + 1] - 1, start, -1):
            mixed = _mix64(key + np.uint64(position) * _GOLDEN)
            chosen = start + np.int64(
                np.float64(mixed >> _SHIFT_11)
                * _UNIT_INTERVAL
                * np.float64(position - start + 1)
            )
            held = labels[position]
            labels[position] = labels[chosen]
            labels[chosen] = held


@njit(cache=True, nogil=True)
def _arrangement_coupling(
    labels: np.ndarray,
    clone_bounds: np.ndarray,
    method: int,
    group_cells: np.ndarray,
    group_strata: np.ndarray,
    stratum_clones: np.ndarray,
    pair_first: np.ndarray,
    pair_second: np.ndarray,
    counts: np.ndarray,
    touched: np.ndarray,
    values: np.ndarray,
    dot: np.ndarray,
    squares: np.ndarray,
    out: np.ndarray,
) -> None:
    """Measure every pair's coupling when each cell holds its label in ``labels``.

    The cells of clone ``c`` are ``clone_bounds[c]`` to ``clone_bounds[c + 1]``.
    Each clone's groups are taken in ascending order, so the coupling depends
    only on the cells of each clone in each group and never on the order of
    a clone's cells. ``counts`` holds zeros before and after; ``touched``,
    ``values``, ``dot`` and ``squares`` are scratch.
    """
    dot[:, :] = 0.0
    squares[:] = 0.0
    for clone in range(len(clone_bounds) - 1):
        n_touched = 0
        for position in range(clone_bounds[clone], clone_bounds[clone + 1]):
            group = labels[position]
            if counts[group] == 0:
                touched[n_touched] = group
                n_touched += 1
            counts[group] += 1
        # Insertion sort: a clone touches few groups.
        for a in range(1, n_touched):
            group = touched[a]
            b = a - 1
            while b >= 0 and touched[b] > group:
                touched[b + 1] = touched[b]
                b -= 1
            touched[b + 1] = group
        if method == _SHARE_COSINE:
            # Each group's share of its cells with a clone in the clone, then
            # each group's share of the clone's shares.
            total = 0.0
            for a in range(n_touched):
                values[a] = counts[touched[a]] / group_cells[touched[a]]
                total += values[a]
            for a in range(n_touched):
                values[a] /= total
        elif method == _JACCARD:
            for a in range(n_touched):
                values[a] = 1.0
        else:
            for a in range(n_touched):
                values[a] = np.float64(counts[touched[a]])
        for a in range(n_touched):
            first = touched[a]
            value = values[a]
            squares[first] += value * value
            for b in range(a + 1, n_touched):
                dot[first, touched[b]] += value * values[b]
        for a in range(n_touched):
            counts[touched[a]] = 0
    for pair in range(len(pair_first)):
        first = pair_first[pair]
        second = pair_second[pair]
        shared = dot[first, second]
        if method == _JACCARD:
            out[pair] = shared / (squares[first] + squares[second] - shared)
        elif method == _NORMALIZED_COVARIANCE:
            # The covariance over the stratum's clones of the two groups'
            # cells, (C sum(x y) - X Y) / (C (C - 1)), over the product of
            # their mean cells per clone, X Y / C^2.
            n_clones = stratum_clones[group_strata[first]]
            cells = group_cells[first] * group_cells[second]
            out[pair] = (
                n_clones * (n_clones * shared - cells) / ((n_clones - 1) * cells)
            )
        else:
            out[pair] = shared / np.sqrt(squares[first] * squares[second])


@njit(cache=True, nogil=True, parallel=True)
def _permuted_couplings(
    labels: np.ndarray,
    stratum_bounds: np.ndarray,
    clone_bounds: np.ndarray,
    method: int,
    group_cells: np.ndarray,
    group_strata: np.ndarray,
    stratum_clones: np.ndarray,
    pair_first: np.ndarray,
    pair_second: np.ndarray,
    seed_key: np.uint64,
    first_permutation: np.int64,
    shuffled: np.ndarray,
    counts: np.ndarray,
    touched: np.ndarray,
    values: np.ndarray,
    dot: np.ndarray,
    squares: np.ndarray,
    out: np.ndarray,
) -> None:
    """Measure the coupling of permutations ``first_permutation`` onwards, one per row of ``out``.

    Each worker, a row of the scratch arrays, takes every ``n_workers``-th
    row. A permutation depends only on the seed and its number, so the rows
    do not depend on the workers.
    """
    n_workers = shuffled.shape[0]
    n_rows = out.shape[0]
    for worker in prange(n_workers):
        # prange's index is unsigned; np.int64 keeps one compiled version.
        own = np.int64(worker)
        for row in range(own, n_rows, n_workers):
            key = _permutation_key(seed_key, first_permutation + row)
            shuffled[own, :] = labels
            _shuffle_within(shuffled[own], stratum_bounds, key)
            _arrangement_coupling(
                shuffled[own],
                clone_bounds,
                method,
                group_cells,
                group_strata,
                stratum_clones,
                pair_first,
                pair_second,
                counts[own],
                touched[own],
                values[own],
                dot[own],
                squares[own],
                out[row],
            )


@njit(cache=True, nogil=True)
def _accumulate(
    rows: np.ndarray, n_rows: int, count: int, mean: np.ndarray, m2: np.ndarray
) -> int:
    """Add rows to running means and sums of squared deviations, in row order (Welford)."""
    for row in range(n_rows):
        count += 1
        for pair in range(rows.shape[1]):
            value = rows[row, pair]
            delta = value - mean[pair]
            mean[pair] += delta / count
            m2[pair] += delta * (value - mean[pair])
    return count


def permuted_labels(
    labels: np.ndarray, stratum_bounds: np.ndarray, seed: int, permutation: int
) -> np.ndarray:
    """The labels as permutation ``permutation`` of ``seed`` shuffles them within strata.

    The null of :func:`clone_coupling` lays out the group labels of the cells
    with a clone stratum after stratum, clone after clone in the order of
    their codes, and each clone's groups in ascending order, and measures
    the coupling of permutation ``k`` on these labels shuffled. The stratum
    ``s`` holds the positions ``stratum_bounds[s]`` to
    ``stratum_bounds[s + 1]``.
    """
    shuffled = np.array(labels, dtype=np.int32)
    # Compiled functions return unsigned keys as Python integers.
    key = np.uint64(_permutation_key(_seed_key(seed), np.int64(permutation)))
    _shuffle_within(shuffled, np.asarray(stratum_bounds, dtype=np.int64), key)
    return shuffled


def coupling_bytes(
    n_groups: int,
    n_pairs: int,
    n_cells: int,
    n_entries: int,
    *,
    workers: int,
    batch: int,
) -> int:
    """The bytes that :func:`clone_coupling` holds at its peak.

    Args:
        n_groups: The groups.
        n_pairs: The pairs of groups compared.
        n_cells: The cells with a clone and a group.
        n_entries: The counts of a clone's cells in a group.
        workers: The workers of the null, 0 without permutations; one set of
            scratch arrays measures the observed coupling.
        batch: The permutations measured at once, 0 without permutations.
    """
    products = n_groups * n_groups
    scratch = _WORKER_BYTES_PER_PRODUCT * products + _WORKER_BYTES_PER_GROUP * n_groups
    return int(
        _COUPLING_BYTES_PER_PAIR * n_pairs
        + _COUPLING_BYTES_PER_PRODUCT * products
        + _COUPLING_BYTES_PER_ENTRY * n_entries
        + _COUPLING_BYTES_PER_CELL * n_cells
        + max(workers, 1) * scratch
        + workers * _WORKER_BYTES_PER_CELL * n_cells
        + _BATCH_BYTES_PER_VALUE * batch * n_pairs
    )


def coupling_plan(
    n_groups: int,
    n_pairs: int,
    n_cells: int,
    n_entries: int,
    *,
    permutations: int,
    nthreads: int,
    available: int,
) -> tuple[int, int, int]:
    """The workers and batch of the null that fit ``available`` bytes.

    The null takes as many workers as ``nthreads`` allows and up to 16
    permutations at once for each, within 64 MiB of coupling values beyond
    one permutation per worker; fewer permutations at once and then fewer
    workers when they do not fit. Its permutations depend on neither, so the
    plan changes only the memory and the time.

    Returns:
        The workers, the permutations measured at once, both 0 without
        permutations, and the bytes that :func:`coupling_bytes` counts for
        them, more than ``available`` only when one worker measuring one
        permutation at a time does not fit.
    """
    sizes = (n_groups, n_pairs, n_cells, n_entries)
    if not permutations:
        return 0, 0, coupling_bytes(*sizes, workers=0, batch=0)
    workers = max(1, min(int(nthreads), int(permutations)))
    per_row = _BATCH_BYTES_PER_VALUE * n_pairs
    while True:
        base = coupling_bytes(*sizes, workers=workers, batch=0)
        rows = (int(available) - base) // per_row
        if rows >= workers or workers == 1:
            capped = max(workers, _BATCH_BYTES // per_row)
            batch = max(1, min(int(permutations), 16 * workers, capped, rows))
            return workers, batch, base + per_row * batch
        workers = max(1, workers // 2)


def _pairs_within(
    n_groups: int, strata: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair ``first < second`` of groups of one stratum, by first and then second."""
    first, second = np.triu_indices(n_groups, k=1)
    if strata is not None:
        same = strata[first] == strata[second]
        first, second = first[same], second[same]
    return first.astype(np.int64), second.astype(np.int64)


def coupling_linkage(
    first: np.ndarray,
    second: np.ndarray,
    coupling: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """The average-linkage tree of the groups, joining the most coupled first.

    Average linkage on the distance ``c - coupling``, with ``c`` the largest
    coupling, merges the two clusters of groups whose average coupling is
    highest, whatever ``c``. Pairs that were not compared, groups of two
    strata, are farther apart than any compared pair, so each stratum's
    groups join before the strata do.

    Returns:
        SciPy's linkage matrix of the groups' positions.

    Raises:
        ValueError: If a compared pair's coupling is not finite.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    values = np.asarray(coupling, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            "Every compared pair needs a finite coupling to order the groups"
        )
    top, bottom = float(values.max()), float(values.min())
    distances = np.full((n_groups, n_groups), top - bottom + 1.0)
    distances[first, second] = top - values
    distances[second, first] = top - values
    np.fill_diagonal(distances, 0.0)
    return np.asarray(linkage(squareform(distances, checks=False), method="average"))


@restore_numba_threads
def clone_coupling(
    pair_groups: np.ndarray,
    pair_clones: np.ndarray,
    pair_cells: np.ndarray,
    n_groups: int,
    *,
    method: str = COUPLING_METHODS[0],
    strata: np.ndarray | None = None,
    permutations: int = 1000,
    seed: int = 0,
    workers: int = 1,
    batch: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Measure how strongly every pair of groups shares clones, against shuffled labels.

    For each clone ``c`` and group ``g``, ``x[g, c]`` counts the clone's cells
    in the group and ``n[g]`` the group's cells with a clone. The methods
    follow CoSpar's ``fate_coupling`` on a clone matrix:

    - ``share_cosine``: the cosine over the clones of the two groups' shares
      ``q[g, c] = (x[g, c] / n[g]) / sum_h(x[h, c] / n[h])``, each group's
      frequency of the clone over the clone's frequencies in all groups, so
      every clone counts once whatever its size.
    - ``abundance_cosine``: the cosine of the two groups' clone sizes, ``sum(x
      y) / sqrt(sum(x^2) sum(y^2))``, as overlap reports it.
    - ``normalized_covariance``: the covariance over the clones of the two
      groups' cells, ``(C sum(x y) - X Y) / (C (C - 1))`` for ``C`` clones and
      ``X`` and ``Y`` cells, over the product of their mean cells per clone,
      ``X Y / C^2``; negative when the groups share fewer clones than chance.
    - ``jaccard``: the clones of both groups over the clones of either.

    The null shuffles the group labels of the cells with a clone within each
    stratum, ``permutations`` times, which keeps each clone's cells and each
    group's cells. Permutation ``k`` draws from a stream keyed by ``seed`` and
    ``k``, so the null depends neither on ``workers`` nor on ``batch``.

    Args:
        pair_groups: The group of each pair of group and clone, as for
            :func:`~cytearc.clones.overlap.group_clone_matrix`.
        pair_clones: The clone code of each pair.
        pair_cells: The cells of the clone in the group, whole numbers.
        n_groups: The number of groups; each needs a cell with a clone.
        method: One of :data:`COUPLING_METHODS`.
        strata: Each group's stratum, such as its labels in the columns that
            the clones keep apart: only groups of one stratum are compared,
            labels are shuffled within strata, and ``normalized_covariance``
            counts the clones of each stratum. None makes one stratum.
        permutations: The permutations of the null, 0 for none.
        seed: The seed of the permutations, a non-negative integer.
        workers: The threads that measure the null, each with its own
            scratch arrays.
        batch: The permutations measured at once, or None for ``16 *
            workers``.

    Returns:
        One row per pair of groups of one stratum, by ``first`` and then
        ``second``: ``first``, ``second``, ``coupling``, ``null_mean``,
        ``null_sd`` (with ``permutations - 1`` degrees of freedom) and
        ``z_score``, NaN without permutations; and the average-linkage tree
        of :func:`coupling_linkage`.

    Raises:
        ValueError: If the pairs are invalid, a group holds no cell, no two
            groups share a stratum, a clone's cells lie in groups of two
            strata, a count is not a whole number, or
            ``normalized_covariance`` meets a stratum of fewer than two
            clones.
    """
    method_code = _checked_method(method)
    matrix = group_clone_matrix(pair_groups, pair_clones, pair_cells, n_groups)
    if not np.array_equal(matrix.data, np.round(matrix.data)):
        raise ValueError("pair_cells must hold whole numbers of cells")
    group_cells = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    if (group_cells == 0).any():
        raise ValueError("Every group needs a cell with a clone")
    if strata is not None and np.shape(strata) != (n_groups,):
        raise ValueError("strata must hold one stratum per group")
    group_strata = (
        np.zeros(n_groups, dtype=np.int64)
        if strata is None
        else np.unique(np.asarray(strata), return_inverse=True)[1].reshape(-1)
    ).astype(np.int64)
    first, second = _pairs_within(n_groups, None if strata is None else group_strata)
    if len(first) == 0:
        raise ValueError("No two groups share a stratum, so no pair can be compared")
    # Each clone's groups and cells, the clones by stratum and then by code.
    by_clone = matrix.T.tocsr()
    del matrix
    held = np.flatnonzero(np.diff(by_clone.indptr) > 0)
    clone_strata = group_strata[by_clone.indices[by_clone.indptr[held]]]
    ordered = by_clone[held[np.argsort(clone_strata, kind="stable")]]
    ordered.sort_indices()
    del by_clone, held
    clone_strata = np.sort(clone_strata)
    entries = np.diff(ordered.indptr)
    if (
        strata is not None
        and (group_strata[ordered.indices] != np.repeat(clone_strata, entries)).any()
    ):
        raise ValueError("A clone's cells lie in groups of two strata")
    n_strata = int(group_strata.max()) + 1
    stratum_clones = np.bincount(clone_strata, minlength=n_strata).astype(np.float64)
    if (
        method_code == _NORMALIZED_COVARIANCE
        and (stratum_clones[np.unique(group_strata[first])] < 2).any()
    ):
        raise ValueError(
            "normalized_covariance needs at least two clones in each stratum of "
            "compared groups"
        )
    # Each cell with a clone holds its group's label, the cells of each clone
    # together and its groups in ascending order.
    cells = ordered.data.astype(np.int64)
    labels = np.repeat(ordered.indices.astype(np.int32), cells)
    clone_bounds = np.zeros(len(entries) + 1, dtype=np.int64)
    np.cumsum(np.add.reduceat(cells, ordered.indptr[:-1]), out=clone_bounds[1:])
    stratum_bounds = clone_bounds[
        np.searchsorted(clone_strata, np.arange(n_strata + 1))
    ].astype(np.int64)
    del ordered, cells, entries, clone_strata
    n_pairs = len(first)
    scratch_workers = max(1, int(workers)) if permutations else 1

    def scratch(n: int) -> tuple[np.ndarray, ...]:
        return (
            np.zeros((n, n_groups), dtype=np.int64),
            np.zeros((n, n_groups), dtype=np.int64),
            np.zeros((n, n_groups), dtype=np.float64),
            np.zeros((n, n_groups, n_groups), dtype=np.float64),
            np.zeros((n, n_groups), dtype=np.float64),
        )

    counts, touched, values, dot, squares = scratch(scratch_workers)
    coupling = np.empty(n_pairs, dtype=np.float64)
    _arrangement_coupling(
        labels,
        clone_bounds,
        method_code,
        group_cells,
        group_strata,
        stratum_clones,
        first,
        second,
        counts[0],
        touched[0],
        values[0],
        dot[0],
        squares[0],
        coupling,
    )
    mean = np.full(n_pairs, np.nan)
    spread = np.full(n_pairs, np.nan)
    if permutations:
        import numba

        numba.set_num_threads(
            min(scratch_workers, int(getattr(numba.config, "NUMBA_NUM_THREADS")))
        )
        rows = 16 * scratch_workers if batch is None else int(batch)
        rows = max(min(rows, int(permutations)), 1)
        shuffled = np.empty((scratch_workers, len(labels)), dtype=np.int32)
        out = np.empty((rows, n_pairs), dtype=np.float64)
        mean[:] = 0.0
        m2 = np.zeros(n_pairs, dtype=np.float64)
        key = _seed_key(seed)
        count = 0
        for start in range(0, int(permutations), rows):
            n_rows = min(rows, int(permutations) - start)
            _permuted_couplings(
                labels,
                stratum_bounds,
                clone_bounds,
                method_code,
                group_cells,
                group_strata,
                stratum_clones,
                first,
                second,
                key,
                np.int64(start),
                shuffled,
                counts,
                touched,
                values,
                dot,
                squares,
                out[:n_rows],
            )
            count = _accumulate(out, n_rows, count, mean, m2)
        del shuffled, out
        spread = np.sqrt(m2 / (count - 1)) if count > 1 else spread
    del counts, touched, values, dot, squares, labels
    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = (coupling - mean) / spread
    tree = coupling_linkage(first, second, coupling, n_groups)
    pairs = pd.DataFrame(
        {
            "first": first,
            "second": second,
            "coupling": coupling,
            "null_mean": mean,
            "null_sd": spread,
            "z_score": z_score,
        }
    )
    return pairs, tree


def dominant_fates(cells: np.ndarray) -> np.ndarray:
    """The fate that holds the most of each row's cells, -1 for none or a tie.

    Args:
        cells: Cells by fate, one row per clone.

    Returns:
        Each row's dominant fate, a column position, or -1 when the row holds
        no cell or several fates hold its most cells.
    """
    cells = np.asarray(cells)
    if cells.ndim != 2:
        raise ValueError("cells must hold one row per clone and one column per fate")
    if cells.shape[1] == 0:
        return np.full(len(cells), -1, dtype=np.int64)
    best = cells.argmax(axis=1)
    top = cells[np.arange(len(cells)), best]
    single = np.count_nonzero(cells == top[:, None], axis=1) == 1
    return np.where((top > 0) & single, best, -1).astype(np.int64)


@dataclass(frozen=True, slots=True)
class FateBiasArrays:
    """The fates of the clones of early cells, as :func:`fate_bias` measures them.

    Attributes:
        clones: The codes of the clones with a late cell, ascending.
        clone_cells: Each of those clones' late cells in each fate.
        clone_early_cells: Each of those clones' early cells.
        p_values: For each of those clones and each fate of its stratum, the
            one-sided Fisher test of more late cells in the fate than chance;
            NaN for fates of another stratum.
        q_values: The p-values adjusted by Benjamini and Hochberg's method
            within each fate.
        fractions: Each early cell's share of its clone's late cells in each
            fate, NaN when its clone has no late cell.
        late_cells: Each early cell's clone's late cells, 0 for none.
        dominant: Each early cell's dominant fate, -1 for none or a tie.
        fate_cells: Each fate's late cells, with or without a clone.
    """

    clones: np.ndarray
    clone_cells: np.ndarray
    clone_early_cells: np.ndarray
    p_values: np.ndarray
    q_values: np.ndarray
    fractions: np.ndarray
    late_cells: np.ndarray
    dominant: np.ndarray
    fate_cells: np.ndarray


def fate_bias_bytes(n_early: int, n_tested: int, n_fates: int) -> int:
    """The bytes that :func:`fate_bias` and the tables of its result hold.

    Args:
        n_early: The early cells.
        n_tested: The clones with a late cell.
        n_fates: The fates.
    """
    return int(
        _BIAS_BYTES_PER_EARLY_CELL * n_early
        + _BIAS_BYTES_PER_FRACTION * n_early * n_fates
        + _BIAS_BYTES_PER_CLONE_FATE * n_tested * n_fates
        + _BIAS_BYTES_PER_FATE * n_fates
    )


def fate_bias(
    early_clones: np.ndarray,
    pair_fates: np.ndarray,
    pair_clones: np.ndarray,
    pair_cells: np.ndarray,
    fate_cells: np.ndarray,
    fate_strata: np.ndarray | None = None,
) -> FateBiasArrays:
    """Follow each early cell's clone to its late cells, and test each clone's fates.

    An early cell's fractions are the shares of its clone's late cells in each
    fate, the clonal fate map that CoSpar's ``fate_map`` computes with
    ``method="sum"`` from a transition map of clones alone; its dominant fate
    holds the most of them. Each clone with late cells is tested against all
    late cells of its stratum with a fate, those without a clone included:
    for a clone of ``N`` late cells, ``x`` of them in a fate of ``K`` late cells
    among ``M``, the p-value is the chance of at least ``x`` in a hypergeometric
    draw of ``N`` from ``M``, Fisher's exact test of enrichment, as CoSpar's
    ``clonal_fate_bias`` computes it with ``alternative="less"``. The p-values
    of each fate are adjusted over the clones of its stratum.

    Args:
        early_clones: Each early cell's clone code, -1 for none.
        pair_fates: The fate of each distinct pair of fate and clone among the
            late cells, as :func:`~cytearc.clones.groups.pair_counts` counts
            them.
        pair_clones: The clone code of each pair.
        pair_cells: The late cells of the clone in the fate.
        fate_cells: Each fate's late cells, with or without a clone; at least
            each pair's cells.
        fate_strata: Each fate's stratum, such as its labels in the columns
            that the clones keep apart, or None for one stratum.

    Raises:
        ValueError: If the arrays do not align, a fate holds fewer late cells
            than its pairs, or a clone's late cells lie in fates of two
            strata.
    """
    from scipy.stats import false_discovery_control, hypergeom

    early_clones = np.asarray(early_clones, dtype=np.int64)
    pair_fates = np.asarray(pair_fates, dtype=np.int64)
    pair_clones = np.asarray(pair_clones, dtype=np.int64)
    pair_cells = np.asarray(pair_cells, dtype=np.int64)
    fate_cells = np.asarray(fate_cells, dtype=np.int64)
    n_fates = len(fate_cells)
    if (
        early_clones.ndim != 1
        or fate_cells.ndim != 1
        or pair_fates.shape != pair_cells.shape
        or pair_clones.shape != pair_cells.shape
        or pair_cells.ndim != 1
    ):
        raise ValueError(
            "early_clones, fate_cells and the pairs must be one-dimensional, the "
            "pairs of one length"
        )
    if len(pair_cells) and (
        pair_fates.min() < 0 or pair_fates.max() >= n_fates or pair_clones.min() < 0
    ):
        raise ValueError("Each pair needs a fate below len(fate_cells) and a clone")
    if (pair_cells <= 0).any() or (
        np.bincount(pair_fates, weights=pair_cells, minlength=n_fates) > fate_cells
    ).any():
        raise ValueError("A fate holds fewer late cells than its pairs of clones")
    strata = (
        np.zeros(n_fates, dtype=np.int64)
        if fate_strata is None
        else np.unique(np.asarray(fate_strata), return_inverse=True)[1].reshape(-1)
    ).astype(np.int64)
    if len(strata) != n_fates:
        raise ValueError("fate_strata must hold one stratum per fate")
    clones, rows = np.unique(pair_clones, return_inverse=True)
    rows = rows.reshape(-1)
    n_tested = len(clones)
    clone_cells = np.zeros((n_tested, n_fates), dtype=np.int64)
    np.add.at(clone_cells, (rows, pair_fates), pair_cells)
    clone_late = clone_cells.sum(axis=1)
    lowest = np.full(n_tested, np.iinfo(np.int64).max, dtype=np.int64)
    highest = np.full(n_tested, -1, dtype=np.int64)
    np.minimum.at(lowest, rows, strata[pair_fates])
    np.maximum.at(highest, rows, strata[pair_fates])
    if (lowest != highest).any():
        raise ValueError("A clone's late cells lie in fates of two strata")
    clone_strata = highest
    population = np.bincount(strata, weights=fate_cells).astype(np.int64)
    tested = strata[None, :] == clone_strata[:, None]
    # A clone without late cells in a fate of its stratum has p = 1.
    p_values = np.where(tested, 1.0, np.nan)
    found_rows, found_fates = np.nonzero(clone_cells)
    p_values[found_rows, found_fates] = hypergeom.sf(
        clone_cells[found_rows, found_fates] - 1,
        population[clone_strata[found_rows]],
        fate_cells[found_fates],
        clone_late[found_rows],
    )
    del found_rows, found_fates
    q_values = np.full_like(p_values, np.nan)
    for fate in range(n_fates):
        column = tested[:, fate]
        if column.any():
            q_values[column, fate] = false_discovery_control(
                p_values[column, fate], method="bh"
            )
    del tested
    at = np.searchsorted(clones, early_clones)
    found = (early_clones >= 0) & (at < n_tested)
    found[found] = clones[at[found]] == early_clones[found]
    positions = at[found]
    # Each clone's fractions once, and a row of NaN for the early cells
    # without late clone-mates, gathered into one array of the early cells.
    shares = np.full((n_tested + 1, n_fates), np.nan)
    np.divide(clone_cells, clone_late[:, None], out=shares[:n_tested])
    fractions = shares[np.where(found, at, n_tested)]
    del shares
    late_cells = np.zeros(len(early_clones), dtype=np.int64)
    late_cells[found] = clone_late[positions]
    dominant = np.full(len(early_clones), -1, dtype=np.int64)
    dominant[found] = dominant_fates(clone_cells)[positions]
    return FateBiasArrays(
        clones=clones.astype(np.int64),
        clone_cells=clone_cells,
        clone_early_cells=np.bincount(positions, minlength=n_tested).astype(np.int64),
        p_values=p_values,
        q_values=q_values,
        fractions=fractions,
        late_cells=late_cells,
        dominant=dominant,
        fate_cells=fate_cells,
    )


#: The columns of the cells table of fate bias besides one per fate, which no
#: fate may share.
FATE_CELL_COLUMNS = ("clone", "late_cells", "fate")


def fate_bias_tables(
    cell_ids: np.ndarray,
    *,
    codes: np.ndarray,
    dominant: np.ndarray,
    late_cells: np.ndarray,
    fractions: np.ndarray,
    clone_ids: np.ndarray,
    clones: np.ndarray,
    clone_cells: np.ndarray,
    clone_early_cells: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
    fates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The tables of early cells and of clones with late cells.

    Args:
        cell_ids: The early cells' IDs.
        codes: Each early cell's clone code, -1 for none.
        dominant: Each early cell's dominant fate, -1 for none.
        late_cells: Each early cell's clone's late cells.
        fractions: Each early cell's share of its clone's late cells in each
            fate.
        clone_ids: The label of each clone code.
        clones: The codes of the clones with a late cell.
        clone_cells: Their late cells in each fate.
        clone_early_cells: Their early cells.
        p_values: Their p-values in each fate, NaN for fates not tested.
        q_values: Their adjusted p-values.
        fates: One row per fate: ``group``, its name, and its key columns.

    Returns:
        One row per early cell, indexed by ID: ``clone``, ``late_cells``,
        ``fate`` and one column per fate of its share; and one row per clone
        and fate of its stratum, by clone code and then fate: ``clone``,
        ``fate``, the key columns, ``early_cells``, ``late_cells``, ``cells``,
        ``fraction``, ``p_value`` and ``q_value``.
    """
    names = fates["group"].to_numpy(dtype=object)
    labels = pd.Index(np.asarray(clone_ids, dtype=object))
    fate_names = pd.Index(names)
    cell_columns: dict[str, Any] = {
        "clone": pd.Categorical.from_codes(np.asarray(codes, dtype=np.int64), labels),
        "late_cells": np.asarray(late_cells, dtype=np.int64),
        "fate": pd.Categorical.from_codes(
            np.asarray(dominant, dtype=np.int64), fate_names
        ),
    }
    shares = np.asarray(fractions, dtype=np.float64)
    for position, name in enumerate(names.tolist()):
        cell_columns[str(name)] = shares[:, position]
    cells = pd.DataFrame(cell_columns, index=pd.Index(np.asarray(cell_ids), name="ids"))
    rows, columns = np.nonzero(~np.isnan(np.asarray(p_values)))
    clone_late = np.asarray(clone_cells).sum(axis=1)
    table = fates.take(columns).reset_index(drop=True)
    table = table.rename(columns={"group": "fate"})
    table["fate"] = pd.Categorical.from_codes(columns, fate_names)
    table.insert(
        0,
        "clone",
        pd.Categorical.from_codes(np.asarray(clones, dtype=np.int64)[rows], labels),
    )
    found = np.asarray(clone_cells)[rows, columns]
    table["early_cells"] = np.asarray(clone_early_cells, dtype=np.int64)[rows]
    table["late_cells"] = clone_late[rows]
    table["cells"] = found
    table["fraction"] = found / clone_late[rows]
    table["p_value"] = np.asarray(p_values)[rows, columns]
    table["q_value"] = np.asarray(q_values)[rows, columns]
    return cells, table


@dataclass(frozen=True, slots=True)
class CloneCoupling:
    """How strongly groups of cells share clones, against shuffled labels.

    Attributes:
        group_sizes: One row per group: ``group``, the key columns, ``cells``
            with a clone and ``clones``.
        pairs: One row per pair of groups compared: ``group_1``, ``group_2``,
            ``coupling``, ``null_mean`` and ``null_sd`` of the permutations, and
            ``z_score``, ``(coupling - null_mean) / null_sd``.
        order: The groups' names in the leaf order of the average-linkage tree
            of the coupling.
        linkage: That tree, as SciPy's linkage matrix of the groups' positions
            in ``group_sizes``.
        method: The coupling method, one of ``COUPLING_METHODS``.
        permutations: The permutations of the null.
        seed: The seed of the permutations.
        clones: The clone artifact, such as a ``clones`` artifact.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
    """

    group_sizes: pd.DataFrame = field(repr=False)
    pairs: pd.DataFrame = field(repr=False)
    order: tuple[str, ...]
    linkage: np.ndarray = field(repr=False)
    method: str
    permutations: int
    seed: int
    clones: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None


@dataclass(frozen=True, slots=True)
class CloneFateBias:
    """The fates that the late cells of each early cell's clone took.

    Attributes:
        cells: One row per early cell, indexed by cell ID: ``clone``,
            ``late_cells`` (its clone's late cells with a fate), ``fate`` (the
            fate that holds the most of them, missing for none or a tie), and
            one column per fate, named by the fate, of the share of the
            clone's late cells in it, NaN without late cells.
        clone_fates: One row per clone with late cells and fate of its
            stratum, by clone and then fate: ``clone``, ``fate``, the fate's
            key columns,
            ``early_cells``, ``late_cells``, ``cells`` (its late cells in the
            fate), ``fraction``, ``p_value`` of the one-sided Fisher test and
            ``q_value``, adjusted by Benjamini and Hochberg's method within
            the fate.
        fates: One row per fate: ``fate``, the key columns, and ``late_cells``,
            those without a clone included unless the clones keep labels
            apart.
        clones: The clone artifact, such as a ``clones`` artifact.
        early: The selection of the early cells.
        late: The selection of the late cells.
        group_by: The cell metadata columns of the fates.
        groups: The label artifact of the fates, or None.
    """

    cells: pd.DataFrame = field(repr=False)
    clone_fates: pd.DataFrame = field(repr=False)
    fates: pd.DataFrame = field(repr=False)
    clones: ArtifactRef
    early: ArtifactRef
    late: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
