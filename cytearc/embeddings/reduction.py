from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..matrix import ChunkedArray
from ..utils.arguments import integer_argument
from ..utils.logging import logger
from ..utils.warnings import warn

type PcaSolver = Literal["gram", "randomized"]

# PCA is exact up to this many features. The Gram solver's p x p float64
# covariance then takes at most 128 MiB, no more than one block of a
# normalized matrix at the default chunk target, and the dense
# eigendecomposition, whose cost grows as p**3, takes seconds. Wider inputs
# use the randomized solver, whose memory and cost grow linearly with p.
_GRAM_PCA_MAX_FEATURES = 4096
# The randomized solver's block of starting vectors holds the fitted
# components and at least as many more, and never fewer than this many more.
_RANDOMIZED_PCA_MIN_OVERSAMPLES = 10
# The randomized solver stops once its leading subspace moves less than this
# many degrees between two passes, by when it is typically a third of that
# angle or less from the exact subspace, or after the last pass.
_RANDOMIZED_PCA_TOLERANCE_DEGREES = 0.1
_RANDOMIZED_PCA_MAX_PASSES = 24
# A new basis direction smaller than this, relative to the largest variance,
# is rounding error of the orthogonalization and is left out.
_RANDOMIZED_PCA_NEGLIGIBLE = 1e-10


@dataclass(frozen=True, slots=True)
class _PcaModel:
    """A fitted PCA with scikit-learn's attribute names.

    ``components_`` holds one principal axis per row, in decreasing order of
    explained variance. ``n_passes_`` counts the passes the fit read over its
    input rows.
    """

    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    mean_: np.ndarray
    n_components_: int
    n_features_in_: int
    n_samples_seen_: int
    solver_: PcaSolver
    n_passes_: int


def pca_solver(n_features: int) -> PcaSolver:
    """Return the solver that PCA uses for ``n_features`` features."""
    return "gram" if int(n_features) <= _GRAM_PCA_MAX_FEATURES else "randomized"


def _randomized_pca_width(n_features: int, dims: int) -> int:
    """Return the number of columns of each block of the randomized solver."""
    oversamples = max(_RANDOMIZED_PCA_MIN_OVERSAMPLES, int(dims))
    return min(int(n_features), int(dims) + oversamples)


def pca_fit_resident_bytes(n_features: int, block_rows: int, dims: int) -> int:
    """Return the bytes a PCA fit holds beside one streamed block of its input.

    Every block is held as two float64 copies at once while it is scaled or
    copied to the solver's memory order, and the fitted components as three.
    The Gram solver adds its p x p covariance and the eigensolver's
    workspace. The randomized solver adds its basis, allocated for the last
    pass, the projection of the covariance onto it with the eigensolver's copy,
    and a few blocks of vectors that orthogonalize the next block.
    """
    itemsize = np.dtype(np.float64).itemsize
    p, rows, dims = int(n_features), int(block_rows), int(dims)
    held = 2 * rows * p + 4 * p + 3 * p * dims
    if pca_solver(p) == "gram":
        return itemsize * (held + p * p + 32 * p)
    width = _randomized_pca_width(p, dims)
    capacity = min(p, width * _RANDOMIZED_PCA_MAX_PASSES)
    return itemsize * (
        held
        + rows * width
        + p * capacity
        + 2 * capacity * capacity
        + (32 + 3 * dims) * capacity
        + 7 * p * width
    )


class _FittedRows:
    """Stream the selected rows of the values that PCA fits as float64 blocks.

    Unscaled values keep their means, and products of values far from zero
    cancel when the means are removed afterwards. Unscaled blocks are
    therefore shifted by the first block's means, and a fit corrects its
    products for the small ``offset`` of the final means from that shift.
    Scaled values are already centered. The first pass also counts the rows
    and sums each column of the unshifted values. Blocks have the memory
    ``order`` that the solver's products read without a copy.
    """

    def __init__(
        self,
        data: ChunkedArray,
        *,
        row_mask: np.ndarray | None,
        scale: Callable[[np.ndarray], np.ndarray] | None,
        nthreads: int,
        resident_bytes: int,
        order: Literal["C", "F"],
    ) -> None:
        self.data = data
        self.row_mask = row_mask
        self.scale = scale
        self.nthreads = nthreads
        self.resident_bytes = resident_bytes
        self.order = order
        self.n_features = int(data.shape[1])
        self.column_sum = np.zeros(self.n_features, dtype=np.float64)
        self.n_rows = 0
        self.passes = 0
        self.shift: np.ndarray | None = None

    def blocks(self, message: str) -> Iterator[np.ndarray]:
        """Yield the blocks of one pass, read ahead within the memory budget."""
        first_pass = self.passes == 0
        self.passes += 1
        for block in self.data._stream_blocks(
            nthreads=self.nthreads,
            msg=message,
            prefetch=None,
            row_mask=self.row_mask,
            resident_bytes=self.resident_bytes,
        ):
            if self.scale is not None:
                values = np.asarray(
                    self.scale(block), dtype=np.float64, order=self.order
                )
            else:
                # A copy, because a NumPy-backed block can view its source.
                values = np.array(block, dtype=np.float64, order=self.order)
            del block
            if first_pass:
                self.column_sum += values.sum(axis=0, dtype=np.float64)
                self.n_rows += len(values)
            if self.scale is None:
                if self.shift is None:
                    self.shift = values.mean(axis=0)
                values -= self.shift
            yield values
            # Release the block before the stream reads the next one.
            del values

    @property
    def mean(self) -> np.ndarray:
        return self.column_sum / self.n_rows

    @property
    def offset(self) -> np.ndarray:
        """The final column means minus the shift of the streamed blocks."""
        return self.mean if self.shift is None else self.mean - self.shift


def _finite_total_variance(total_variance: float) -> float:
    """Return the total variance of finite values, raising for any other value.

    Constant values have no variance and give components of zero variance,
    and rounding of a variance near zero is clipped to it.
    """
    if not np.isfinite(total_variance):
        raise ValueError("PCA input must have finite variance")
    return max(total_variance, 0.0)


def _pca_model(
    rows: _FittedRows,
    *,
    vectors: np.ndarray,
    eigenvalues: np.ndarray,
    total_variance: float,
    solver: PcaSolver,
) -> _PcaModel:
    """Return the model of eigenpairs in decreasing order of eigenvalue."""
    from sklearn.utils.extmath import svd_flip

    variance = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    components = np.array(vectors.T, dtype=np.float64, copy=True, order="C")
    _, components = svd_flip(None, components, u_based_decision=False)
    return _PcaModel(
        components_=components,
        explained_variance_=variance,
        explained_variance_ratio_=(
            variance / total_variance if total_variance > 0 else np.zeros_like(variance)
        ),
        singular_values_=np.sqrt(variance * (rows.n_rows - 1)),
        mean_=rows.mean,
        n_components_=len(variance),
        n_features_in_=rows.n_features,
        n_samples_seen_=rows.n_rows,
        solver_=solver,
        n_passes_=rows.passes,
    )


def _fit_gram_pca(rows: _FittedRows, *, dims: int, nthreads: int) -> _PcaModel:
    """Fit exact PCA from the covariance that one pass accumulates."""
    from scipy.linalg import blas, eigh
    from threadpoolctl import ThreadpoolController

    n_features = rows.n_features
    controller = ThreadpoolController()
    gram = np.zeros(
        (n_features, n_features),
        dtype=np.float64,
        order="F",
    )
    for values in rows.blocks("Fitting PCA"):
        # The block stream clamps BLAS to one thread per reader for its
        # whole lifetime; the rank update runs here, between blocks.
        with controller.limit(limits=nthreads, user_api="blas"):
            gram = blas.dsyrk(
                1.0,
                values,
                beta=1.0,
                c=gram,
                trans=1,
                lower=0,
                overwrite_c=1,
            )
        del values

    gram = blas.dsyr(
        -float(rows.n_rows),
        rows.offset,
        lower=0,
        a=gram,
        overwrite_a=1,
    )
    gram /= rows.n_rows - 1
    total_variance = _finite_total_variance(float(np.trace(gram)))
    eigenvalues, eigenvectors = eigh(
        gram,
        lower=False,
        subset_by_index=(n_features - dims, n_features - 1),
        overwrite_a=True,
        check_finite=False,
        driver="evr",
    )
    return _pca_model(
        rows,
        vectors=eigenvectors[:, ::-1],
        eigenvalues=eigenvalues[::-1],
        total_variance=total_variance,
        solver="gram",
    )


def _fit_randomized_pca(
    rows: _FittedRows,
    *,
    dims: int,
    random_state: int,
    nthreads: int,
) -> _PcaModel:
    """Fit PCA with a streamed randomized block Krylov solver.

    The solver follows the randomized range finder of Halko, Martinsson and
    Tropp (2011, SIAM Review 53:217) on the covariance ``C`` of the fitted
    values, and keeps every power of ``C`` applied to its random starting
    block in its basis, a block Krylov space (Rokhlin, Szlam and Tygert 2009;
    Musco and Musco 2015), which converges in far fewer passes than subspace
    iteration when the trailing eigenvalues are close. Each pass multiplies
    the newest block of the basis by ``C`` while it streams the rows, so the
    centered matrix is never formed, and the Rayleigh-Ritz projection of
    ``C`` onto the basis gives the components and their variances. The fit
    stops once the leading ``dims``-dimensional subspace moves less than
    ``_RANDOMIZED_PCA_TOLERANCE_DEGREES`` between two passes, or once the
    basis spans every feature or a subspace that ``C`` maps into itself, where
    the projection is exact, and warns when the last pass comes first.
    """
    from scipy.linalg import eigh, qr, subspace_angles
    from threadpoolctl import ThreadpoolController

    n_features = rows.n_features
    controller = ThreadpoolController()
    width = _randomized_pca_width(n_features, dims)
    capacity = min(n_features, width * _RANDOMIZED_PCA_MAX_PASSES)
    tolerance = np.radians(_RANDOMIZED_PCA_TOLERANCE_DEGREES)
    basis = np.empty((n_features, capacity), dtype=np.float64, order="F")
    starting = np.random.default_rng(random_state).standard_normal((n_features, width))
    basis[:, :width] = np.linalg.qr(starting)[0]
    del starting
    # The projection of the covariance onto the basis. A pass adds the
    # column block of its newest basis block and, by symmetry, the row block.
    projected = np.zeros((capacity, capacity), dtype=np.float64)
    start, stop = 0, width
    total_variance = 0.0
    previous: np.ndarray | None = None
    change = np.inf
    pass_number = 0
    # Every pass ends the loop or adds a block to the basis.
    while True:
        pass_number += 1
        newest = basis[:, start:stop]
        product = np.zeros((n_features, stop - start), dtype=np.float64)
        squares = 0.0
        for values in rows.blocks(f"Fitting PCA (pass {pass_number})"):
            # The block stream clamps BLAS to one thread per reader for its
            # whole lifetime; the products run here, between blocks.
            with controller.limit(limits=nthreads, user_api="blas"):
                product += values.T @ (values @ newest)
                if pass_number == 1:
                    flat = values.ravel(order="K")
                    squares += float(flat @ flat)
            del values
        n_rows = rows.n_rows
        offset = rows.offset
        if pass_number == 1:
            total_variance = _finite_total_variance(
                (squares - n_rows * float(offset @ offset)) / (n_rows - 1)
            )
        product -= n_rows * np.outer(offset, offset @ newest)
        product /= n_rows - 1

        projected[:stop, start:stop] = basis[:, :stop].T @ product
        diagonal = projected[start:stop, start:stop]
        projected[start:stop, start:stop] = (diagonal + diagonal.T) / 2
        projected[start:stop, :start] = projected[:start, start:stop].T
        # The leading Ritz pairs, in increasing order.
        ritz_values, ritz_vectors = eigh(
            projected[:stop, :stop], subset_by_index=(stop - dims, stop - 1)
        )
        leading = ritz_vectors[:, ::-1]
        if previous is not None:
            padded = np.zeros_like(leading)
            padded[: len(previous)] = previous
            change = float(subspace_angles(padded, leading)[0])
        if change <= tolerance or stop == n_features:
            break
        if pass_number == _RANDOMIZED_PCA_MAX_PASSES:
            warn(
                f"PCA's randomized solver stopped after {pass_number} passes "
                f"while its leading {dims} components still moved "
                f"{np.degrees(change):.2g} degrees in the last pass. Fit fewer "
                "dims or select fewer features."
            )
            break
        previous = leading
        # The next block is the product's part outside the basis,
        # orthogonalized twice. Where the product barely leaves the basis, that
        # part is rounding error and not orthogonal to the basis, so a pivoted
        # QR keeps only the larger directions, which are orthogonalized once
        # more after they are normalized.
        known = basis[:, :stop]
        residual = product - known @ (known.T @ product)
        residual -= known @ (known.T @ residual)
        directions, triangle, _ = qr(residual, mode="economic", pivoting=True)
        negligible = _RANDOMIZED_PCA_NEGLIGIBLE * float(ritz_values[-1])
        kept = int(np.count_nonzero(np.abs(np.diag(triangle)) > negligible))
        kept = min(kept, capacity - stop)
        if kept == 0:
            # The product lies in the basis, so the basis spans an invariant
            # subspace and its Rayleigh-Ritz pairs are exact.
            break
        directions = directions[:, :kept]
        directions -= known @ (known.T @ directions)
        basis[:, stop : stop + kept] = np.linalg.qr(directions)[0]
        start, stop = stop, stop + kept
        del newest, product, known, residual, directions, triangle

    logger.debug(
        f"Fitted PCA of {n_features} features with the randomized solver in "
        f"{rows.passes} passes over {n_rows} rows, a basis of {stop} vectors "
        f"in blocks of {width}, with a last change of "
        f"{np.degrees(change):.2g} degrees"
    )
    return _pca_model(
        rows,
        vectors=basis[:, :stop] @ leading,
        eigenvalues=ritz_values[::-1],
        total_variance=total_variance,
        solver="randomized",
    )


def fit_pca(
    data: ChunkedArray,
    *,
    dims: int,
    use_for_pca: np.ndarray,
    scale: Callable[[np.ndarray], np.ndarray] | None,
    random_state: int,
    nthreads: int,
) -> tuple[np.ndarray, _PcaModel]:
    """Fit PCA over streamed row blocks and return loadings with the model.

    ``use_for_pca`` is a boolean vector over the rows of ``data`` that
    selects at least ``dims + 1`` rows, and ``data`` has at least
    ``dims + 1`` columns. The reduction operation checks both before it fits.

    Up to 4,096 features the fit is exact: one pass accumulates the float64
    covariance of the selected rows, whose leading eigenvectors are the
    loadings. Wider inputs use the randomized block Krylov solver, which
    reads the selected rows once per pass, for up to 24 passes, from vectors
    seeded by ``random_state``. Neither result depends on the block size
    beyond rounding. Constant values give components of zero variance, and
    values that are not finite raise ValueError.
    """
    from threadpoolctl import threadpool_limits

    selected_samples = int(np.count_nonzero(use_for_pca))
    row_mask = use_for_pca if selected_samples != data.shape[0] else None
    n_features = int(data.shape[1])
    block_rows = int(data.chunksize[0])
    solver = pca_solver(n_features)
    rows = _FittedRows(
        data,
        row_mask=row_mask,
        scale=scale,
        nthreads=nthreads,
        resident_bytes=pca_fit_resident_bytes(n_features, block_rows, dims),
        # The rank update reads Fortran-ordered blocks; the randomized
        # solver's products read scaled blocks as they come.
        order="F" if solver == "gram" else "C",
    )
    with threadpool_limits(limits=nthreads):
        if solver == "gram":
            accumulator_mib = n_features * n_features * 8 / (1024**2)
            logger.debug(
                "Fitting PCA with the Gram covariance solver "
                f"({n_features} features, {block_rows} rows per block, "
                f"{accumulator_mib:.0f} MiB accumulator)"
            )
            model = _fit_gram_pca(rows, dims=dims, nthreads=nthreads)
        else:
            logger.debug(
                "Fitting PCA with the randomized solver "
                f"({n_features} features, {block_rows} rows per block, "
                f"blocks of {_randomized_pca_width(n_features, dims)} vectors)"
            )
            model = _fit_randomized_pca(
                rows,
                dims=dims,
                random_state=random_state,
                nthreads=nthreads,
            )
            logger.info(
                f"Fitted PCA of {n_features} features with the randomized solver "
                f"in {model.n_passes_} passes over the data. At most "
                f"{_GRAM_PCA_MAX_FEATURES} features, such as highly variable "
                "genes, are fitted exactly in one pass."
            )
    return model.components_.T, model


def fit_lsi(
    data: ChunkedArray,
    *,
    dims: int,
    skip_first: bool,
    params: dict[str, Any],
    random_state: int,
    nthreads: int,
) -> np.ndarray:
    """Fit uncentered LSI loadings with a streamed or materialized solver.

    ``dims + skip_first`` must not exceed the smaller dimension of ``data``;
    the reduction operation checks the rank before it fits.
    """
    reserved = sorted({"n_components", "random_state"}.intersection(params))
    if reserved:
        raise ValueError(f"LSI parameters cannot set {', '.join(reserved)}")

    n_components = dims + int(skip_first)
    solver_params = dict(params)
    solver = solver_params.pop("solver", "streaming")
    if solver == "streaming":
        allowed = {"n_iter", "n_oversamples"}
        unsupported = sorted(set(solver_params) - allowed)
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"Streaming LSI does not support parameters: {joined}")
        return _fit_streaming_lsi(
            data,
            n_components=n_components,
            skip_first=skip_first,
            n_iter=solver_params.get("n_iter", 5),
            n_oversamples=solver_params.get("n_oversamples", 10),
            random_state=random_state,
            nthreads=nthreads,
        )
    if solver != "materialized":
        raise ValueError("LSI solver must be 'streaming' or 'materialized'")
    return _fit_materialized_lsi(
        data,
        n_components=n_components,
        skip_first=skip_first,
        params=solver_params,
        random_state=random_state,
        nthreads=nthreads,
    )


def _fit_materialized_lsi(
    data: ChunkedArray,
    *,
    n_components: int,
    skip_first: bool,
    params: dict[str, Any],
    random_state: int,
    nthreads: int,
) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD

    # Fill one preallocated matrix instead of holding every block and a
    # stacked copy at the same time.
    matrix = data.compute(nthreads=nthreads, msg="Fitting materialized LSI model")
    model = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
        **params,
    )
    model.fit(matrix)
    components = model.components_.T
    if skip_first:
        return np.asarray(components[:, 1:])
    return np.asarray(components)


def _materialized_lsi_bytes(
    *,
    n_rows: int,
    n_features: int,
    itemsize: int,
    n_components: int,
    n_oversamples: int,
) -> int:
    """Estimate peak bytes held by the materialized LSI solver.

    The dense input is held once and scikit-learn's explained-variance step
    allocates a matrix-sized temporary. The randomized SVD keeps a few cell-
    and feature-length bases with ``n_components + n_oversamples`` columns.
    """
    matrix_bytes = int(n_rows) * int(n_features) * int(itemsize)
    width = min(int(n_rows), int(n_features), int(n_components) + int(n_oversamples))
    float_bytes = np.dtype(np.float64).itemsize
    basis_bytes = (
        4 * (int(n_rows) + int(n_features)) * width * float_bytes
        + 2 * width * width * float_bytes
    )
    return 2 * matrix_bytes + basis_bytes


def require_materialized_lsi_budget(
    *,
    n_rows: int,
    n_features: int,
    itemsize: int,
    n_components: int,
    n_oversamples: int,
    memory_bytes: int,
) -> None:
    """Raise MemoryError when the materialized LSI solver exceeds a budget."""
    required = _materialized_lsi_bytes(
        n_rows=n_rows,
        n_features=n_features,
        itemsize=itemsize,
        n_components=n_components,
        n_oversamples=n_oversamples,
    )
    if required > memory_bytes:
        raise MemoryError(
            f"Materialized LSI needs about {required} bytes to hold the "
            f"{n_rows} x {n_features} matrix and its SVD workspace, but the "
            f"operation limit is {memory_bytes} bytes. Use solver='streaming', "
            "which reads the matrix in bounded blocks."
        )


def _streaming_lsi_accumulator_bytes(n_features: int, width: int) -> int:
    itemsize = np.dtype(np.float64).itemsize
    return 3 * n_features * width * itemsize + 2 * width * width * itemsize


def _streaming_lsi_resident_bytes(data: ChunkedArray, width: int) -> int:
    block_rows = min(int(data.chunksize[0]), int(data.shape[0]))
    block_temporaries = (
        block_rows * (int(data.shape[1]) + width) * np.dtype(np.float64).itemsize
    )
    return (
        _streaming_lsi_accumulator_bytes(int(data.shape[1]), width) + block_temporaries
    )


def _stream_lsi_gram_action(
    data: ChunkedArray,
    basis: np.ndarray,
    *,
    nthreads: int,
    message: str,
) -> np.ndarray:
    result = np.zeros_like(basis, dtype=np.float64)
    resident_bytes = _streaming_lsi_resident_bytes(data, basis.shape[1])
    for block in data._stream_blocks(
        nthreads=nthreads,
        msg=message,
        prefetch=1,
        row_mask=None,
        resident_bytes=resident_bytes,
    ):
        values = np.asarray(block)
        projected = values @ basis
        result += values.T @ projected
        del block, values, projected
    if not np.isfinite(result).all():
        raise ValueError("LSI input must contain only finite values")
    return result


def _fit_streaming_lsi(
    data: ChunkedArray,
    *,
    n_components: int,
    skip_first: bool,
    n_iter: Any,
    n_oversamples: Any,
    random_state: int,
    nthreads: int,
) -> np.ndarray:
    iterations = integer_argument(n_iter, "n_iter", minimum=0)
    oversamples = integer_argument(n_oversamples, "n_oversamples", minimum=0)
    width = min(min(data.shape), n_components + oversamples)
    rng = np.random.default_rng(random_state)
    basis = rng.standard_normal((data.shape[1], width), dtype=np.float64)
    basis, _ = np.linalg.qr(basis, mode="reduced")

    for iteration in range(iterations + 1):
        basis = _stream_lsi_gram_action(
            data,
            basis,
            nthreads=nthreads,
            message=f"Fitting streaming LSI model ({iteration + 1}/{iterations + 2})",
        )
        basis, _ = np.linalg.qr(basis, mode="reduced")

    projected_gram = np.zeros((width, width), dtype=np.float64)
    resident_bytes = _streaming_lsi_resident_bytes(data, width)
    # The power iterations above have rejected non-finite input, so this
    # projection of the same rows onto an orthonormal basis stays finite.
    for block in data._stream_blocks(
        nthreads=nthreads,
        msg=f"Fitting streaming LSI model ({iterations + 2}/{iterations + 2})",
        prefetch=1,
        row_mask=None,
        resident_bytes=resident_bytes,
    ):
        projected = np.asarray(block) @ basis
        projected_gram += projected.T @ projected
        del block, projected

    eigenvalues, rotations = np.linalg.eigh(projected_gram)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    loadings = np.asarray(basis @ rotations[:, order], dtype=np.float64)
    largest = np.argmax(np.abs(loadings), axis=0)
    signs = np.sign(loadings[largest, np.arange(loadings.shape[1])])
    signs[signs == 0] = 1
    loadings *= signs
    if skip_first:
        return loadings[:, 1:]
    return loadings
