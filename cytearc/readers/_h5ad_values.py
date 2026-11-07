"""Read and scan the values of an H5AD count matrix.

Scan processes import this module alone, so that they start without the
rest of the H5AD reader.
"""

from typing import Any

import h5py
import numpy as np

from ..utils.count_values import (
    CountScan,
    IndexOrder,
    compressed_count_ranges,
    dense_count_ranges,
    new_count_ranges,
)


def matrix_values(node: h5py.Dataset, selection: slice, dtype: Any) -> np.ndarray:
    """Read the values of a dense matrix's rows in ``dtype``."""
    if node.dtype == dtype:
        return np.asarray(node[selection])
    return np.asarray(node.astype(dtype)[selection])


def scan_window(
    node: Any,
    orientation: str,
    sourceDtype: Any,
    minorSize: int,
    start: int,
    stop: int,
    maxBytes: int,
    featureGroups: np.ndarray | None,
) -> CountScan:
    """Scan the values of the compressed vectors ``[start, stop)`` of a matrix.

    The vectors are the rows of a dense or CSR matrix and the columns of a
    CSC matrix. A scan of every vector reads the pointers as it goes; a
    window reads its own pointers first.

    Args:
        node: The matrix's dataset, or the group of a sparse matrix.
        orientation: ``"dense"``, ``"csr"``, or ``"csc"``.
        sourceDtype: Dtype in which a dense matrix's values are read.
        minorSize: Length of a sparse matrix's minor axis.
        start: First vector to scan.
        stop: Vector after the last one to scan.
        maxBytes: Memory available to the scan.
        featureGroups: Group, numbered from zero, of each feature. None puts
            every feature in one group.
    """
    if orientation == "dense":
        row_nonzeros = np.empty(stop - start, dtype=np.int64)
        value_ranges = dense_count_ranges(
            lambda first, last: matrix_values(
                node, slice(start + first, start + last), sourceDtype
            ),
            stop - start,
            int(node.shape[1]),
            maxBytes=maxBytes,
            groups=featureGroups,
            rowNonzeros=row_nonzeros,
        )
        return CountScan(start, stop, value_ranges, rowNonzeros=row_nonzeros)
    csr = orientation == "csr"
    whole = start == 0 and stop == int(node["indptr"].shape[0]) - 1
    index_order = IndexOrder()
    value_ranges = compressed_count_ranges(
        node["indptr"] if whole else np.asarray(node["indptr"][start : stop + 1]),
        node["indices"],
        node["data"],
        minorSize=minorSize,
        maxBytes=maxBytes,
        # Features are the minor axis of CSR and the vectors of CSC.
        groups=(
            featureGroups if csr or featureGroups is None else featureGroups[start:stop]
        ),
        groupAxis=1 if csr else 0,
        # A window of CSC columns may hold no feature of some groups, and
        # still has a range for each.
        valueRanges=new_count_ranges(featureGroups),
        indexOrder=index_order,
    )
    return CountScan(start, stop, value_ranges, canonical=index_order.canonical)


def scan_file_window(
    path: str,
    matrixKey: str,
    orientation: str,
    sourceDtype: Any,
    minorSize: int,
    start: int,
    stop: int,
    maxBytes: int,
    featureGroups: np.ndarray | None,
) -> CountScan:
    """Scan one window of the matrix ``matrixKey`` of an H5AD file.

    A scan process runs this; the arguments after ``matrixKey`` are those
    of :func:`scan_window`.
    """
    with h5py.File(path, mode="r") as h5:
        return scan_window(
            h5[matrixKey],
            orientation,
            sourceDtype,
            minorSize,
            start,
            stop,
            maxBytes,
            featureGroups,
        )
