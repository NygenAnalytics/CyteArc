"""Peak memory of a Harmony2 fit, for admission before any coordinate is read.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

import math
from collections.abc import Sequence

import numpy as np

from ...utils.arguments import integer_argument
from .parameters import check_harmony_options, harmony_work_sizes


def _column_levels(column_levels: object, n_cells: int) -> list[int]:
    if isinstance(column_levels, str | bytes) or not isinstance(
        column_levels, Sequence | np.ndarray
    ):
        raise TypeError(
            "column_levels must be a sequence of level counts, one per batch column"
        )
    levels: list[int] = []
    for count in column_levels:
        if isinstance(count, bool | np.bool_) or not isinstance(
            count, int | np.integer
        ):
            raise TypeError(
                "column_levels must be a sequence of level counts, one per batch column"
            )
        levels.append(int(count))
    if not levels or any(not 1 <= count <= n_cells for count in levels):
        raise ValueError(
            "column_levels must hold at least one level count, each between 1 and "
            "n_cells"
        )
    return levels


def harmony_peak_bytes(
    n_cells: int,
    dims: int,
    n_clusters: int,
    column_levels: Sequence[int],
    *,
    n_combinations: int | None = None,
    block_size: float = 0.05,
) -> int:
    """Return the bytes that a Harmony2 fit holds at its peak, with its float64 input.

    The bound covers the caller's ``(dims, cells)`` float64 input, which the
    DataStore fills before the fit, and the fit's own arrays at the largest of
    its phases: encoding the batch design, k-means++ seeding, clustering, and
    correction. Per cell, the dominant terms are ``24 dims + 8 n_clusters`` bytes:
    the input, original and corrected coordinates, and the assignments. An
    eighth of the total is added for small temporaries, plus 64 KiB for Python
    objects.

    Args:
        n_cells: The number of cells.
        dims: The number of dimensions.
        n_clusters: The cluster count of the fit.
        column_levels: The number of levels of each batch column.
        n_combinations: The number of distinct level combinations among the
            cells; by default the largest it can be.
        block_size: The fit's ``block_size``.

    Returns:
        The peak in bytes.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    d = integer_argument(dims, "dims", minimum=1)
    k = integer_argument(n_clusters, "n_clusters", minimum=1)
    if k > cells:
        raise ValueError(
            f"n_clusters must be between one and the cell count ({cells}); got {k}"
        )
    levels = _column_levels(column_levels, cells)
    c = len(levels)
    b = sum(levels)
    largest = max(levels)
    most = min(cells, math.prod(levels))
    if n_combinations is None:
        g = b if c == 1 else most
    else:
        g = integer_argument(n_combinations, "n_combinations")
        if not largest <= g <= most:
            raise ValueError(
                "n_combinations must lie between the largest level count and "
                "min(n_cells, product of level counts)"
            )
    _block_cells, m = harmony_work_sizes(
        cells, check_harmony_options(block_size=block_size)["block_size"]
    )
    # The input, original and corrected coordinates, the inverse norms, the
    # level codes, the update order and the cells sorted by combination, the
    # combination offsets and levels, the level counts, proportions and theta,
    # and the norm scratch.
    base = (
        24 * cells * d
        + 8 * cells
        + 4 * c * cells
        + 16 * cells
        + 8 * (g + 1)
        + 4 * g * c
        + 24 * b
        + 8 * m
    )
    # The input with the level codes, the factorize codes and the combination keys.
    encoding = 8 * cells * d + 4 * c * cells + (56 + 24 * (c - 1)) * cells
    # Five float64 seeding vectors and a boolean mask.
    seeding = 41 * cells
    # Chunk buffers, three (levels, clusters) tables and four (dims, clusters) arrays.
    workspace = 8 * m * (k + 2 * d + 1) + 24 * b * k + 32 * d * k
    clustering = 8 * cells * k + workspace
    # Two dense Schur complements over the intercept and the levels outside the
    # largest column.
    schur = 0 if c == 1 else 16 * (1 + b - largest) ** 2
    # Combination masses, the (levels, clusters, dims) sums, the intercept sums
    # and terms, the activity mask, the Schur complements, the sparse coupling,
    # and the per-cluster gathers of the largest column's sums.
    correction = (
        8 * cells * k
        + workspace
        + 8 * g * k
        + 8 * b * k * d
        + 16 * k * d
        + b * k
        + schur
        + 48 * (g + b)
        + 24 * largest * d
    )
    raw = max(encoding, base + max(seeding, clustering, correction))
    return raw + raw // 8 + 65536
