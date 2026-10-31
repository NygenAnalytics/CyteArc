"""E-distances between perturbed and control cells in a coordinate space.

The E-distance of Peidli et al. (2024, scPerturb) measures how far the cells
of one perturbation sit from the control cells in a space such as a PCA:

    E = 2 * mean ||x - y||^2 - mean ||x - x'||^2 - mean ||y - y'||^2

over perturbed cells x, control cells y, and distinct pairs within each group,
with squared Euclidean distances. With group means and sample covariances
the three means reduce exactly to

    E = 2 * ||mean_P - mean_C||^2 - 2 * trace(S_P) / n_P - 2 * trace(S_C) / n_C,

so each distance needs only the sums and squared norms of the cells, never a
matrix of pairwise distances.

A permutation test draws, many times, a group of the perturbed group's size
from the pooled perturbed and control cells and counts the E-distances that
reach the observed one; the p-value is (count + 1) / (permutations + 1),
adjusted across rows by Benjamini-Hochberg. Each draw takes a uniform subset of
the pool with Floyd's algorithm, of the smaller group's size, the rest of the
pool forming the other group, and scales uniform numbers in [0, 1) from the
generator to the positions it picks. A permuted distance that falls short of the
observed one by less than 1e-9 times the pooled cells' mean squared norm still
reaches it, so rounding cannot break the ties that equal groups give. Each row
draws from its own seed, derived from the given seed and the row, so a row's
p-value does not depend on the other rows.

These functions read coordinates that a caller passes and never touch storage.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from cytearc.perturbation.efficiency import rows_setting
from cytearc.utils.arguments import integer_argument
from cytearc.utils.numba import njit

__all__ = [
    "EDISTANCE_STATUSES",
    "EDISTANCE_TESTED",
    "EDISTANCE_TOO_FEW_CELLS",
    "EDistanceArrays",
    "PerturbationEDistance",
    "edistance_arrays",
    "edistance_from_sums",
    "edistance_memory_bytes",
    "edistance_settings",
]

#: Row statuses, stored as positions in this tuple.
EDISTANCE_STATUSES = ("tested", "too_few_cells")
#: The E-distance was computed.
EDISTANCE_TESTED = 0
#: Fewer than two perturbed cells, so the within-group mean is undefined.
EDISTANCE_TOO_FEW_CELLS = 1
#: The most permutations of one row's test.
MAX_PERMUTATIONS = 1_000_000

# The most bytes of uniform numbers that a thread draws at once. The generator
# yields the same numbers however they are split, so the p-values do not
# depend on it.
_DRAW_BYTES = 1 << 23
# The bytes of Python objects that a thread holds while it tests rows, such
# as its state and the generator of its row; about 8 kB in Python 3.13.
_THREAD_BYTES = 1 << 16
# A permuted distance that falls short of the observed one by less than this
# share of the pooled cells' mean squared norm reaches it.
_TIE_TOLERANCE = 1e-9


@njit(cache=True, nogil=True)
def edistance_from_sums(
    sum_p: np.ndarray,
    squares_p: float,
    n_p: int,
    sum_c: np.ndarray,
    squares_c: float,
    n_c: int,
) -> float:
    """Return the E-distance of two groups from their sums.

    Args:
        sum_p: The sum of the perturbed cells' coordinates.
        squares_p: The sum of their squared norms.
        n_p: The number of perturbed cells, at least 2.
        sum_c: The sum of the control cells' coordinates.
        squares_c: The sum of their squared norms.
        n_c: The number of control cells, at least 2.
    """
    between = 0.0
    norm_p = 0.0
    norm_c = 0.0
    for dim in range(sum_p.shape[0]):
        mean_p = sum_p[dim] / n_p
        mean_c = sum_c[dim] / n_c
        between += (mean_p - mean_c) * (mean_p - mean_c)
        norm_p += mean_p * mean_p
        norm_c += mean_c * mean_c
    trace_p = (squares_p - n_p * norm_p) / (n_p - 1)
    trace_c = (squares_c - n_c * norm_c) / (n_c - 1)
    return 2.0 * between - 2.0 * trace_p / n_p - 2.0 * trace_c / n_c


@njit(cache=True, nogil=True)
def _count_reached(
    values: np.ndarray,
    squares: np.ndarray,
    n_control: int,
    start: int,
    n_perturbed: int,
    uniforms: np.ndarray,
    pooled_sum: np.ndarray,
    pooled_squares: float,
    threshold: float,
) -> int:
    """Count the permuted groups whose E-distance reaches ``threshold``.

    The pool holds the control cells, rows 0 to ``n_control`` of ``values``,
    and the perturbed cells, ``n_perturbed`` rows from ``start``; pool
    position p is row p for a control cell and row ``start + p - n_control``
    otherwise. Each row of ``uniforms`` picks a uniform subset of the pool by
    Floyd's algorithm: its k-th number, scaled to a position from 0 to
    ``last = n_pool - n_drawn + k``, is taken unless the subset holds it, in
    which case ``last`` is. The subset is the permuted perturbed group when it
    has the perturbed group's size, and the permuted control group otherwise.
    """
    n_pool = n_control + n_perturbed
    n_drawn = uniforms.shape[1]
    n_dims = values.shape[1]
    taken = np.zeros(n_pool, dtype=np.bool_)
    chosen = np.empty(n_drawn, dtype=np.int64)
    drawn_sum = np.empty(n_dims)
    rest_sum = np.empty(n_dims)
    reached = 0
    for permutation in range(uniforms.shape[0]):
        drawn_sum[:] = 0.0
        drawn_squares = 0.0
        for k in range(n_drawn):
            last = n_pool - n_drawn + k
            position = min(int(uniforms[permutation, k] * (last + 1)), last)
            if taken[position]:
                position = last
            taken[position] = True
            chosen[k] = position
            cell = position if position < n_control else start + position - n_control
            for dim in range(n_dims):
                drawn_sum[dim] += values[cell, dim]
            drawn_squares += squares[cell]
        for k in range(n_drawn):
            taken[chosen[k]] = False
        for dim in range(n_dims):
            rest_sum[dim] = pooled_sum[dim] - drawn_sum[dim]
        rest_squares = pooled_squares - drawn_squares
        if n_drawn == n_perturbed:
            distance = edistance_from_sums(
                drawn_sum, drawn_squares, n_perturbed, rest_sum, rest_squares, n_control
            )
        else:
            distance = edistance_from_sums(
                rest_sum, rest_squares, n_perturbed, drawn_sum, drawn_squares, n_control
            )
        if distance >= threshold:
            reached += 1
    return reached


def edistance_settings(
    by: object, n_permutations: object, random_seed: object
) -> tuple[int, int]:
    """Check the settings of E-distances, as ``edistance`` takes them.

    The pipeline recipe checks the same settings before its run starts.

    Returns:
        The permutations of each test and the seed.

    Raises:
        TypeError: If a number is not an integer.
        ValueError: If ``by`` is unknown, or a number is out of range.
    """
    rows_setting(by)
    permutations = integer_argument(
        n_permutations, "n_permutations", minimum=0, maximum=MAX_PERMUTATIONS
    )
    seed = integer_argument(random_seed, "random_seed", minimum=0)
    return permutations, seed


def edistance_memory_bytes(
    group_sizes: np.ndarray,
    n_control: int,
    n_dims: int,
    *,
    n_permutations: int,
    nthreads: int = 1,
) -> int:
    """Return the bytes that :func:`edistance_arrays` holds.

    The count includes the float64 coordinates, int64 groups and boolean
    control flags that the caller passes.

    Args:
        group_sizes: The perturbed cells of each row, tested or not.
        n_control: The control cells.
        n_dims: The dimensions of the cells' coordinates.
        n_permutations: The permutations of each test.
        nthreads: The threads that may test rows at once.
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    n_cells = n_control + int(sizes.sum())
    # The coordinates as passed and as sorted take 16 bytes per cell and
    # dimension. The groups and flags as passed, the squared norms and the
    # positions that sort the cells by row take at most 41 bytes per cell,
    # which also leaves room for a flag per perturbed cell of the rows that
    # the threads test. Each row's bounds, results and adjustment take at
    # most 128 bytes. Centring the coordinates, and in numpy before 2.3 also
    # summing them, passes them through a buffer of at most np.getbufsize()
    # numbers.
    held = n_cells * (16 * n_dims + 41) + 128 * len(sizes) + 8 * np.getbufsize()
    tested = sizes[sizes >= 2]
    # A thread tests one row at a time, and no more threads run than rows
    # are tested.
    threads = min(nthreads, len(tested))
    if not threads:
        return held
    # A row draws blocks of uniform numbers of at most _DRAW_BYTES, or of
    # one permutation of the smaller group when that is larger, and never
    # more than its permutations at once, which the largest group bounds.
    drawn = min(int(tested.max()), n_control)
    draws = min(8 * drawn * n_permutations, max(_DRAW_BYTES, 8 * drawn))
    # Each thread holds one block of uniform numbers, its kernel a flag per
    # control cell, the position of each drawn cell and the sums of the two
    # groups, and its own Python objects.
    per_thread = draws + 9 * n_control + 32 * n_dims + _THREAD_BYTES
    return held + threads * per_thread


@dataclass(frozen=True, slots=True)
class EDistanceArrays:
    """One entry per target or guide.

    Attributes:
        status: The position of each row's status in
            :data:`EDISTANCE_STATUSES`.
        n_cells: The perturbed cells with coordinates.
        n_control_cells: The control cells with coordinates.
        edistance: The E-distance to the control cells, NaN without a test.
        p_value: The permutation p-value, NaN without permutations.
        adjusted_p_value: The Benjamini-Hochberg adjustment across the rows
            with a p-value.
    """

    status: np.ndarray
    n_cells: np.ndarray
    n_control_cells: np.ndarray
    edistance: np.ndarray
    p_value: np.ndarray
    adjusted_p_value: np.ndarray


def edistance_arrays(
    coordinates: np.ndarray,
    groups: np.ndarray,
    control: np.ndarray,
    n_rows: int,
    *,
    n_permutations: int,
    random_seed: int,
    nthreads: int = 1,
) -> EDistanceArrays:
    """Compute each row's E-distance to the control cells and its p-value.

    Args:
        coordinates: The coordinates of the compared cells, cells x dimensions.
        groups: The row each cell carries alone, or -1.
        control: Whether each cell is a control cell, which carries no row.
        n_rows: The number of rows.
        n_permutations: The permutations of each test; 0 skips the tests.
        random_seed: The seed from which each row's permutations draw.
        nthreads: The threads that test rows at once. The results do not
            depend on it.

    Returns:
        The status, cells, E-distance and p-values of every row.

    Raises:
        ValueError: If the arrays do not align, a cell carries a row out of
            range or is a control cell that carries a row, or there are fewer
            than two control cells.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    control = np.asarray(control, dtype=bool)
    if values.ndim != 2 or len(groups) != len(values) or len(control) != len(values):
        raise ValueError(
            "coordinates must be cells x dimensions, with one group and one "
            "control flag per cell"
        )
    if np.any(groups >= n_rows) or np.any(control & (groups >= 0)):
        raise ValueError(
            "groups must hold rows below n_rows, and control cells carry no row"
        )
    controls = np.flatnonzero(control)
    n_c = len(controls)
    if n_c < 2:
        raise ValueError("E-distances need at least two control cells")
    grouped = np.flatnonzero(groups >= 0)
    members = grouped[np.argsort(groups[grouped], kind="stable")]
    # The control cells come first and then the cells of each row in turn,
    # centred on the control mean, which keeps the sums small and limits
    # cancellation in the traces.
    values = values[np.concatenate([controls, members])]
    values -= values[:n_c].mean(axis=0)
    squares = np.einsum("ij,ij->i", values, values)
    sum_c = values[:n_c].sum(axis=0)
    squares_c = float(squares[:n_c].sum())
    bounds = n_c + np.searchsorted(groups[members], np.arange(n_rows + 1))
    sizes = np.diff(bounds).astype(np.int64)
    status = np.full(n_rows, EDISTANCE_TOO_FEW_CELLS, dtype=np.int8)
    distance = np.full(n_rows, np.nan)
    p_values = np.full(n_rows, np.nan)

    def measure(row: int) -> None:
        start, n_p = int(bounds[row]), int(sizes[row])
        sum_p = values[start : start + n_p].sum(axis=0)
        squares_p = float(squares[start : start + n_p].sum())
        observed = float(
            edistance_from_sums(sum_p, squares_p, n_p, sum_c, squares_c, n_c)
        )
        distance[row] = observed
        if not n_permutations:
            return
        n_pool = n_c + n_p
        pooled_squares = squares_c + squares_p
        threshold = observed - _TIE_TOLERANCE * pooled_squares / n_pool
        n_drawn = min(n_p, n_c)
        block = max(1, _DRAW_BYTES // (8 * n_drawn))
        rng = np.random.default_rng([random_seed, row])
        # One block of uniform numbers, filled again for each block of
        # permutations, so that a thread never holds two.
        uniforms = np.empty((min(block, n_permutations), n_drawn))
        reached = 0
        for first in range(0, n_permutations, block):
            reached += int(
                _count_reached(
                    values,
                    squares,
                    n_c,
                    start,
                    n_p,
                    rng.random(out=uniforms[: n_permutations - first]),
                    sum_c + sum_p,
                    pooled_squares,
                    threshold,
                )
            )
        p_values[row] = (reached + 1) / (n_permutations + 1)

    tested_rows = np.flatnonzero(sizes >= 2)
    status[tested_rows] = EDISTANCE_TESTED
    threads = min(nthreads, len(tested_rows))
    if threads > 1:
        # Each thread measures the next row until none is left, so that no
        # object is kept for each row. The kernel releases the GIL, and each
        # row draws from its own seed.
        next_rows = iter(tested_rows)
        lock = Lock()

        def measure_rows() -> None:
            while True:
                with lock:
                    row = next(next_rows, None)
                if row is None:
                    return
                measure(int(row))

        with ThreadPoolExecutor(max_workers=threads) as pool:
            for running in [pool.submit(measure_rows) for _ in range(threads)]:
                running.result()
    else:
        for row in tested_rows:
            measure(int(row))
    adjusted = np.full(n_rows, np.nan)
    tested = np.isfinite(p_values)
    if tested.any():
        adjusted[tested] = false_discovery_control(p_values[tested], method="bh")
    return EDistanceArrays(
        status=status,
        n_cells=sizes,
        n_control_cells=np.full(n_rows, n_c, dtype=np.int64),
        edistance=distance,
        p_value=p_values,
        adjusted_p_value=adjusted,
    )


@dataclass(frozen=True, slots=True)
class PerturbationEDistance:
    """The E-distance of each perturbation to the control cells.

    Attributes:
        table: One row per target or guide, with ``target``, ``guide`` (for
            rows by guide), ``status``, ``n_cells``, ``n_control_cells``,
            ``edistance``, ``p_value`` and ``adjusted_p_value``.
        by: ``"target"`` or ``"guide"``.
        from_assay: The assay of the coordinates.
        n_permutations: The permutations of each test.
        random_seed: The seed from which each row's permutations drew.
    """

    table: pd.DataFrame = field(repr=False)
    by: str
    from_assay: str
    n_permutations: int
    random_seed: int
