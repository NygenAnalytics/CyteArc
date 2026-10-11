import numpy as np
import zarr

from ..storage.virtual_arrays import read_selection


def _read_block(
    zarr_arr: zarr.Array,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Read ``zarr_arr[row_idx, col_idx]`` returning rows/cols in index order.

    A basic slice is used only for an index run that is provably contiguous
    (consecutive ascending integers), so it selects exactly the requested
    positions and never includes neighbouring rows or columns. Any other
    selection falls back to orthogonal (fancy) indexing, which preserves the
    order of the index arrays. This centralizes the read path so callers never
    hand-roll ``slice(idx[0], idx[-1] + 1)``. It is ``read_selection``, kept on
    the assay facade so tests can intercept assay reads.
    """
    return read_selection(zarr_arr, row_idx, col_idx)
