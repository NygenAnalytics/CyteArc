import numpy as np
from scipy.sparse import csr_matrix

from ..utils.arrays import sparse_matrix_bytes
from ..utils.memory_budget import MemoryBudgetError
from ..utils.numba import njit

_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT64_SIZE = int(np.dtype(np.float64).itemsize)


def inverse_degree_diagonal(degree: np.ndarray) -> csr_matrix:
    """Return a sparse diagonal of reciprocal degrees; zero degrees stay zero."""
    inverse_degree = np.array(degree, copy=True)
    nonzero = inverse_degree != 0
    inverse_degree[nonzero] = 1 / inverse_degree[nonzero]
    n_cells = inverse_degree.shape[0]
    return csr_matrix(
        (
            inverse_degree,
            (range(n_cells), range(n_cells)),
        ),
        shape=[n_cells, n_cells],
    )


def _row_degree_diagonal(graph: csr_matrix) -> csr_matrix:
    return inverse_degree_diagonal(np.ravel(graph.sum(axis=1)))


def transition_bytes(n_cells: int, n_entries: int) -> int:
    """Return the most bytes that :func:`transition_matrix` holds beside the graph.

    The reciprocal degrees are a diagonal CSR matrix, a float64 value and two
    int32 indices a cell, and the product holds its float64 values and int32
    indices for each of the graph's ``n_entries`` entries, its row pointers,
    and a float64 value and two int32 links a cell while it is formed.
    """
    cells, entries = int(n_cells), int(n_entries)
    return 12 * entries + 36 * cells + 8


def transition_matrix(graph: csr_matrix) -> csr_matrix:
    """Return the row-normalized graph, one step of graph diffusion.

    Each row with edges sums to one and rows without edges stay empty. The
    result has the entries of ``graph``, so it needs about as much memory as
    the graph itself.
    """
    return _row_degree_diagonal(graph).dot(graph)


@njit(cache=True)
def _product_nnz_bound(
    left_indptr: np.ndarray,
    left_indices: np.ndarray,
    right_row_nnz: np.ndarray,
    n_cols: int,
) -> int:
    """Sum per-row upper bounds on the entries of a sparse product."""
    total = 0
    for row in range(left_indptr.shape[0] - 1):
        count = 0
        for offset in range(left_indptr[row], left_indptr[row + 1]):
            count += right_row_nnz[left_indices[offset]]
        total += min(count, n_cols)
    return total


@njit(cache=True)
def _product_nnz(
    left_indptr: np.ndarray,
    left_indices: np.ndarray,
    right_indptr: np.ndarray,
    right_indices: np.ndarray,
    n_cols: int,
) -> int:
    """Count the structural entries of a sparse product without forming it."""
    last_row = np.full(n_cols, -1, dtype=np.int64)
    total = 0
    for row in range(left_indptr.shape[0] - 1):
        for offset in range(left_indptr[row], left_indptr[row + 1]):
            middle = left_indices[offset]
            for inner in range(right_indptr[middle], right_indptr[middle + 1]):
                col = right_indices[inner]
                if last_row[col] != row:
                    last_row[col] = row
                    total += 1
    return total


def _product_bytes(left: csr_matrix, right: csr_matrix, nnz: int) -> int:
    """Bytes SciPy allocates to multiply two CSR matrices with ``nnz`` outputs."""
    n_rows, n_cols = int(left.shape[0]), int(right.shape[1])
    narrow_inputs = all(
        np.dtype(array.dtype).itemsize <= 4
        for matrix in (left, right)
        for array in (matrix.indptr, matrix.indices)
    )
    wide = max(n_rows, n_cols, nnz) > _INT32_MAX or not narrow_inputs
    index_size = 8 if wide else 4
    output = nnz * (_FLOAT64_SIZE + index_size) + (n_rows + 1) * index_size
    # The product keeps one value and two links per output column.
    scratch = n_cols * (_FLOAT64_SIZE + 2 * index_size)
    if wide and narrow_inputs:
        # Operand index arrays are widened before the product.
        scratch += 8 * (int(left.nnz) + int(right.nnz) + n_rows + n_cols + 2)
    return output + scratch


def bounded_diffusion_operator(
    graph: csr_matrix,
    power: int,
    *,
    memory_bytes: int,
) -> csr_matrix:
    """Return ``transition_matrix(graph)`` raised to ``power`` within a budget.

    The products mirror ``scipy.sparse.linalg.matrix_power``, so the entries
    and their storage order are those of the unbounded power. Before each
    product, its output entries and the operands still held, including
    ``graph``, are checked against ``memory_bytes``, and ``MemoryError`` is
    raised before a product that does not fit is allocated.
    """
    if isinstance(power, bool) or not isinstance(power, int | np.integer):
        raise TypeError("power must be a positive integer")
    if power < 1:
        raise ValueError("power must be a positive integer")

    def reserve(
        left: csr_matrix,
        right: csr_matrix,
        held: tuple[csr_matrix, ...],
    ) -> None:
        held_bytes = sparse_matrix_bytes(graph, *held)
        n_cols = int(right.shape[1])
        bound = int(
            _product_nnz_bound(
                left.indptr,
                left.indices,
                np.diff(right.indptr),
                n_cols,
            )
        )
        if held_bytes + _product_bytes(left, right, bound) < memory_bytes:
            return
        exact = int(
            _product_nnz(
                left.indptr,
                left.indices,
                right.indptr,
                right.indices,
                n_cols,
            )
        )
        needed = held_bytes + _product_bytes(left, right, exact)
        if needed >= memory_bytes:
            raise MemoryBudgetError(
                f"Diffusion step with {exact} operator entries needs about "
                f"{needed} bytes, but the memory budget is {memory_bytes} bytes.",
                needed_bytes=needed + 1,
                budget_bytes=int(memory_bytes),
                item=f"a diffusion step of {exact} operator entries",
                remedy="use a smaller diffusion power t",
            )

    diagonal = _row_degree_diagonal(graph)
    reserve(diagonal, graph, (diagonal,))
    del diagonal
    transition = transition_matrix(graph)

    # The products of scipy.sparse.linalg.matrix_power, in its order, so that
    # values match exactly: from the smallest power up, each squares the one
    # below it, after one more step for an odd power.
    exponents = []
    exponent = int(power)
    while exponent > 1:
        exponents.append(exponent)
        exponent //= 2
    current = transition
    for exponent in reversed(exponents):
        half = current
        if exponent % 2:
            reserve(transition, half, (transition, half))
            partial = transition @ half
            reserve(partial, half, (transition, half, partial))
            current = partial @ half
            del partial
        else:
            reserve(half, half, (transition, half))
            current = half @ half
    return csr_matrix(current)
