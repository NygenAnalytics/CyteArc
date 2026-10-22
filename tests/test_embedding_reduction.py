import tracemalloc

import numpy as np
import pytest
import scipy.linalg
from scipy.linalg import subspace_angles
from sklearn.decomposition import TruncatedSVD

import cytearc.embeddings.reduction as reduction_module
from cytearc.embeddings.reduction import (
    _FittedRows,
    _streaming_lsi_resident_bytes,
    fit_lsi,
    fit_pca,
    pca_fit_resident_bytes,
    pca_solver,
)
from cytearc.matrix import ChunkedArray
from cytearc.storage.budget import ResourceBudget


class _ReadTrackingArray(ChunkedArray):
    def __init__(self, values: np.ndarray, block_size: int) -> None:
        super().__init__(values, block_size=block_size, nthreads=1, is_numpy=True)
        self.read_ranges: list[tuple[int, int]] = []

    def _materialize_range(self, start: int, end: int) -> np.ndarray:
        self.read_ranges.append((start, end))
        return super()._materialize_range(start, end)


class _ResidentTrackingArray(ChunkedArray):
    def __init__(
        self,
        values: np.ndarray,
        *,
        block_size: int,
        resources: ResourceBudget,
    ) -> None:
        super().__init__(
            values,
            block_size=block_size,
            nthreads=resources.workers,
            resources=resources,
            is_numpy=True,
        )
        self.resident_bytes: list[int] = []

    def _stream_blocks(
        self,
        *,
        nthreads,
        msg,
        prefetch,
        row_mask,
        resident_bytes=0,
    ):
        self.resident_bytes.append(resident_bytes)
        yield from super()._stream_blocks(
            nthreads=nthreads,
            msg=msg,
            prefetch=prefetch,
            row_mask=row_mask,
            resident_bytes=resident_bytes,
        )


# More features than the exact Gram solver takes.
_WIDE_FEATURES = 4100


def _spectrum_values(
    n_rows: int,
    n_features: int,
    seed: int,
    *,
    n_signal: int = 12,
) -> np.ndarray:
    """Return float32 values whose variances decay toward a noise floor.

    Signal axes with geometrically decaying spreads lie under unit noise, and
    each feature has its own mean and scale, as normalized expression does.
    """
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n_rows, n_signal)) * np.geomspace(10.0, 3.0, n_signal)
    axes = np.linalg.qr(rng.standard_normal((n_features, n_signal)))[0]
    values = scores @ axes.T + rng.standard_normal((n_rows, n_features))
    values += rng.uniform(0.0, 3.0, n_features)
    values *= rng.uniform(0.5, 2.0, n_features)
    return values.astype(np.float32)


@pytest.fixture(scope="module")
def wide_values() -> np.ndarray:
    return _spectrum_values(200, _WIDE_FEATURES, 11)


def _standardizer(values: np.ndarray):
    """Return the feature scaling that a PCA operation applies to ``values``."""
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)

    def scale(block: np.ndarray) -> np.ndarray:
        return (block - mean) / std

    return scale


def _exact_pca(values: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading principal axes and variance ratios of a dense SVD."""
    widened = np.asarray(values, dtype=np.float64)
    centered = widened - widened.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular_values)
    return axes[:dims].T, variance[:dims] / variance.sum()


def _max_angle_degrees(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.degrees(subspace_angles(observed, expected)).max())


def _fit(
    values: np.ndarray,
    *,
    block_size: int,
    dims: int,
    selected: np.ndarray | None = None,
    scale=None,
    random_state: int = 4466,
    nthreads: int = 1,
    data: ChunkedArray | None = None,
):
    if data is None:
        data = ChunkedArray.from_numpy(values, block_size=block_size)
    return fit_pca(
        data,
        dims=dims,
        use_for_pca=(
            np.ones(len(values), dtype=bool) if selected is None else selected
        ),
        scale=scale,
        random_state=random_state,
        nthreads=nthreads,
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_fitted_rows_copy_shared_views_before_shifting(order: str) -> None:
    source = np.arange(24, dtype=np.float64).reshape(6, 4) + 100.0
    original = source.copy()
    rows = _FittedRows(
        ChunkedArray.from_numpy(source, block_size=3),
        row_mask=None,
        scale=None,
        nthreads=1,
        resident_bytes=0,
        order=order,
    )

    blocks = list(rows.blocks("Fitting PCA"))

    assert all(block.flags.owndata for block in blocks)
    assert all(
        block.flags.c_contiguous if order == "C" else block.flags.f_contiguous
        for block in blocks
    )
    # Every block is shifted by the first block's means; the source is not.
    shift = original[:3].mean(axis=0)
    np.testing.assert_array_equal(np.vstack(blocks), original - shift)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(rows.mean, original.mean(axis=0))
    np.testing.assert_array_equal(rows.offset, original.mean(axis=0) - shift)
    assert (rows.n_rows, rows.passes) == (6, 1)
    # A later pass keeps the shift and the counts of the first.
    np.testing.assert_array_equal(
        np.vstack(list(rows.blocks("Fitting PCA"))), original - shift
    )
    assert (rows.n_rows, rows.passes) == (6, 2)


def test_pca_is_deterministic_across_chunked_input():
    values = np.random.default_rng(42).normal(size=(24, 6))
    original_values = values.copy()
    data = ChunkedArray.from_numpy(values, block_size=8)
    selected = np.ones(values.shape[0], dtype=bool)

    first, first_model = fit_pca(
        data,
        dims=3,
        use_for_pca=selected,
        scale=None,
        random_state=4466,
        nthreads=1,
    )
    second, second_model = fit_pca(
        data,
        dims=3,
        use_for_pca=selected,
        scale=None,
        random_state=4466,
        nthreads=1,
    )

    assert first.shape == (6, 3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first_model.explained_variance_ratio_,
        second_model.explained_variance_ratio_,
    )
    np.testing.assert_array_equal(values, original_values)
    # The loadings are the leading principal axes, up to sign.
    _, _, axes = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose(np.abs(first.T @ axes[:3].T), np.eye(3), atol=1e-8)


def test_gram_pca_matches_full_svd_and_preserves_model_contract():
    values = np.random.default_rng(11).normal(size=(40, 7))
    data = ChunkedArray.from_numpy(values, block_size=10)
    selected = np.ones(values.shape[0], dtype=bool)

    loadings, model = fit_pca(
        data,
        dims=3,
        use_for_pca=selected,
        scale=None,
        random_state=4466,
        nthreads=1,
    )

    centered = values - values.mean(axis=0)
    _, singular_values, expected_components = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    expected_components = expected_components[:3]
    expected_variance = np.square(singular_values[:3]) / (len(values) - 1)
    total_variance = values.var(axis=0, ddof=1).sum()

    assert loadings.shape == (values.shape[1], 3)
    assert model.components_.shape == (3, values.shape[1])
    assert model.explained_variance_ratio_.shape == (3,)
    np.testing.assert_array_equal(loadings, model.components_.T)
    np.testing.assert_allclose(
        np.abs(model.components_ @ expected_components.T),
        np.eye(3),
        atol=1e-10,
    )
    np.testing.assert_allclose(model.explained_variance_, expected_variance)
    np.testing.assert_allclose(
        model.explained_variance_ratio_,
        expected_variance / total_variance,
    )
    np.testing.assert_allclose(model.singular_values_, singular_values[:3])
    np.testing.assert_allclose(model.mean_, values.mean(axis=0))
    assert model.n_components_ == 3
    assert model.n_features_in_ == values.shape[1]
    assert model.n_samples_seen_ == values.shape[0]
    assert (model.solver_, model.n_passes_) == ("gram", 1)
    # Signs follow svd_flip: the largest loading of each component is positive.
    largest = np.argmax(np.abs(model.components_), axis=1)
    assert np.all(model.components_[np.arange(3), largest] > 0)


def test_unscaled_gram_pca_does_not_cancel_large_means():
    rng = np.random.default_rng(29)
    spread = rng.normal(size=(600, 6)) * np.linspace(1.0, 3.0, 6)
    # Means of a million beside unit spreads: the Gram matrix minus
    # n * mean * mean^T loses about twelve digits of the covariance.
    values = (1e6 + spread).astype(np.float32)
    original = values.copy()
    data = ChunkedArray.from_numpy(values, block_size=100)

    _, model = fit_pca(
        data,
        dims=2,
        use_for_pca=np.ones(len(values), dtype=bool),
        scale=None,
        random_state=4466,
        nthreads=1,
    )

    widened = values.astype(np.float64)
    covariance = np.cov(widened, rowvar=False)
    expected = np.sort(np.linalg.eigvalsh(covariance))[::-1][:2]
    np.testing.assert_allclose(model.explained_variance_, expected, rtol=1e-9)
    np.testing.assert_allclose(
        model.explained_variance_ratio_, expected / np.trace(covariance), rtol=1e-9
    )
    np.testing.assert_array_equal(model.mean_, widened.mean(axis=0))
    # The source matrix is read, never shifted in place.
    np.testing.assert_array_equal(values, original)
    # Nor is a float64 source, whose last block of one row is a view of it.
    rows = widened[:201].copy()
    fit_pca(
        ChunkedArray.from_numpy(rows, block_size=100),
        dims=2,
        use_for_pca=np.ones(len(rows), dtype=bool),
        scale=None,
        random_state=4466,
        nthreads=1,
    )
    np.testing.assert_array_equal(rows, widened[:201])


def test_unscaled_randomized_pca_does_not_cancel_large_means(monkeypatch):
    # The randomized solver shifts unscaled blocks as the Gram solver does.
    monkeypatch.setattr(reduction_module, "_GRAM_PCA_MAX_FEATURES", 2)
    rng = np.random.default_rng(29)
    spread = rng.normal(size=(600, 20)) * np.linspace(1.0, 3.0, 20)
    values = (1e6 + spread).astype(np.float32)
    original = values.copy()

    _, model = _fit(values, block_size=100, dims=2)

    covariance = np.cov(values.astype(np.float64), rowvar=False)
    expected = np.sort(np.linalg.eigvalsh(covariance))[::-1][:2]
    assert model.solver_ == "randomized"
    np.testing.assert_allclose(model.explained_variance_, expected, rtol=1e-9)
    np.testing.assert_allclose(
        model.explained_variance_ratio_, expected / np.trace(covariance), rtol=1e-9
    )
    np.testing.assert_array_equal(values, original)


def test_gram_pca_matches_selected_scaled_rows():
    values = np.random.default_rng(19).normal(size=(36, 5))
    data = ChunkedArray.from_numpy(values, block_size=6)
    selected = np.zeros(values.shape[0], dtype=bool)
    selected[::2] = True
    offset = np.linspace(-0.5, 0.5, values.shape[1])
    divisor = np.linspace(1.0, 2.0, values.shape[1])

    def scale(block: np.ndarray) -> np.ndarray:
        return (block - offset) / divisor

    loadings, model = fit_pca(
        data,
        dims=2,
        use_for_pca=selected,
        scale=scale,
        random_state=4466,
        nthreads=1,
    )

    expected_values = scale(values[selected])
    centered = expected_values - expected_values.mean(axis=0)
    _, _, expected_components = np.linalg.svd(centered, full_matrices=False)
    expected_components = expected_components[:2]

    np.testing.assert_allclose(
        np.abs(model.components_ @ expected_components.T),
        np.eye(2),
        atol=1e-10,
    )
    np.testing.assert_allclose(model.mean_, expected_values.mean(axis=0))
    np.testing.assert_array_equal(loadings, model.components_.T)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize(
    ("block_size", "masked"),
    [
        # One block of every row.
        (240, False),
        # Several blocks, each with more rows than features.
        (80, False),
        # Blocks with fewer rows than features, which IncrementalPCA fitted
        # before, with trailing components tens of degrees off.
        (25, False),
        (25, True),
    ],
)
def test_gram_pca_is_exact_for_every_block_layout(
    scaled: bool, block_size: int, masked: bool
) -> None:
    values = _spectrum_values(240, 60, 5)
    selected = np.ones(240, dtype=bool)
    if masked:
        selected[1::3] = False
    scale = _standardizer(values) if scaled else None

    loadings, model = _fit(
        values, block_size=block_size, dims=12, selected=selected, scale=scale
    )

    fitted = values[selected] if scale is None else scale(values[selected])
    axes, ratios = _exact_pca(fitted, 12)
    assert (model.solver_, model.n_passes_) == ("gram", 1)
    assert _max_angle_degrees(loadings, axes) < 1e-8
    # Each component, not just their span, matches the exact axis.
    np.testing.assert_allclose(
        np.abs(np.sum(loadings * axes, axis=0)), np.ones(12), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(model.explained_variance_ratio_, ratios, rtol=1e-10)
    np.testing.assert_allclose(
        model.mean_, fitted.mean(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-12
    )


@pytest.mark.slow
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize(
    ("block_size", "masked"),
    [(200, False), (64, False), (64, True)],
)
def test_randomized_pca_matches_the_exact_subspace(
    wide_values: np.ndarray, scaled: bool, block_size: int, masked: bool
) -> None:
    selected = np.ones(len(wide_values), dtype=bool)
    if masked:
        selected[1::3] = False
    scale = _standardizer(wide_values) if scaled else None

    loadings, model = _fit(
        wide_values, block_size=block_size, dims=10, selected=selected, scale=scale
    )

    fitted = wide_values[selected] if scale is None else scale(wide_values[selected])
    axes, ratios = _exact_pca(fitted, 10)
    assert model.solver_ == "randomized"
    assert 3 <= model.n_passes_ < reduction_module._RANDOMIZED_PCA_MAX_PASSES
    assert _max_angle_degrees(loadings, axes) < 0.5
    np.testing.assert_allclose(model.explained_variance_ratio_, ratios, rtol=1e-6)
    np.testing.assert_allclose(loadings.T @ loadings, np.eye(10), atol=1e-12)
    np.testing.assert_allclose(
        model.mean_, fitted.mean(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    ("n_features", "dims", "passes"),
    [
        # Blocks of twelve vectors span all twelve features at once.
        (12, 2, 1),
        # Blocks of eighteen vectors: the second block completes thirty.
        (30, 8, 2),
    ],
)
def test_randomized_pca_is_exact_once_its_basis_spans_every_feature(
    monkeypatch, n_features: int, dims: int, passes: int
) -> None:
    monkeypatch.setattr(reduction_module, "_GRAM_PCA_MAX_FEATURES", 4)
    values = _spectrum_values(60, n_features, 5, n_signal=6)

    loadings, model = _fit(values, block_size=16, dims=dims)

    axes, ratios = _exact_pca(values, dims)
    assert (model.solver_, model.n_passes_) == ("randomized", passes)
    assert _max_angle_degrees(loadings, axes) < 1e-8
    np.testing.assert_allclose(model.explained_variance_ratio_, ratios, rtol=1e-10)


def test_randomized_pca_warns_when_it_reaches_the_last_pass(
    monkeypatch, wide_values: np.ndarray
) -> None:
    monkeypatch.setattr(reduction_module, "_RANDOMIZED_PCA_MAX_PASSES", 3)

    with pytest.warns(
        UserWarning,
        match=r"stopped after 3 passes while its leading 10 components still moved",
    ):
        loadings, model = _fit(wide_values, block_size=64, dims=10)

    assert model.n_passes_ == 3
    assert loadings.shape == (_WIDE_FEATURES, 10)
    np.testing.assert_allclose(loadings.T @ loadings, np.eye(10), atol=1e-12)


@pytest.mark.slow
def test_randomized_pca_is_deterministic_for_a_fixed_seed(
    wide_values: np.ndarray,
) -> None:
    first, first_model = _fit(wide_values, block_size=64, dims=10, random_state=7)
    again, again_model = _fit(wide_values, block_size=64, dims=10, random_state=7)
    other, _ = _fit(wide_values, block_size=64, dims=10, random_state=8)
    one_block, _ = _fit(wide_values, block_size=200, dims=10, random_state=7)

    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(
        first_model.explained_variance_, again_model.explained_variance_
    )
    assert first_model.n_passes_ == again_model.n_passes_
    # The block size changes only the rounding of the sums.
    np.testing.assert_allclose(one_block, first, rtol=0, atol=1e-10)
    # Another seed starts from other vectors and reaches the same subspace.
    assert not np.array_equal(first, other)
    assert _max_angle_degrees(first, other) < 0.5


@pytest.mark.slow
@pytest.mark.parametrize("n_features", [60, _WIDE_FEATURES])
def test_pca_agrees_across_thread_counts(
    wide_values: np.ndarray, n_features: int
) -> None:
    values = (
        wide_values if n_features == _WIDE_FEATURES else _spectrum_values(240, 60, 5)
    )
    scale = _standardizer(values)

    fits = [
        _fit(values, block_size=64, dims=10, scale=scale, nthreads=threads)
        for threads in (1, 2, 4)
    ]

    loadings, model = fits[0]
    for other_loadings, other_model in fits[1:]:
        assert other_model.n_passes_ == model.n_passes_
        np.testing.assert_allclose(other_loadings, loadings, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            other_model.explained_variance_, model.explained_variance_, rtol=1e-12
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_features", "block_size", "scaled"),
    [
        # Arrays large enough that the fit's own allocations dominate.
        (600, 2000, True),
        (600, 500, False),
        (_WIDE_FEATURES, 200, True),
        (_WIDE_FEATURES, 64, False),
    ],
)
def test_pca_fit_resident_bytes_bound_the_traced_peak(
    wide_values: np.ndarray, n_features: int, block_size: int, scaled: bool
) -> None:
    values = (
        wide_values
        if n_features == _WIDE_FEATURES
        else _spectrum_values(2000, n_features, 5)
    )
    scale = _standardizer(values) if scaled else None
    # A first fit imports the solvers, so the traced fit allocates only arrays.
    _fit(values, block_size=block_size, dims=10, scale=scale)
    data = ChunkedArray.from_numpy(values, block_size=block_size)

    tracemalloc.start()
    try:
        _fit(values, block_size=block_size, dims=10, scale=scale, data=data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    bound = (
        pca_fit_resident_bytes(n_features, int(data.chunksize[0]), 10)
        + data._block_task_bytes()
    )
    assert peak <= bound < 2 * peak


def test_pca_solver_depends_only_on_the_feature_count():
    assert pca_solver(1) == "gram"
    assert pca_solver(4096) == "gram"
    assert pca_solver(4097) == "randomized"
    assert pca_solver(np.int64(30_000)) == "randomized"


def test_pca_logs_selected_solver(monkeypatch):
    messages: list[tuple[str, str]] = []

    class RecordingLogger:
        @staticmethod
        def debug(message: str) -> None:
            messages.append(("debug", message))

        @staticmethod
        def info(message: str) -> None:
            messages.append(("info", message))

    monkeypatch.setattr(reduction_module, "logger", RecordingLogger())
    rng = np.random.default_rng(21)
    # Blocks narrower than the features, and a single block, fit exactly.
    for n_features, block_size in ((6, 8), (10, 8), (10, 24)):
        _fit(rng.normal(size=(24, n_features)), block_size=block_size, dims=2)
        assert messages.pop() == (
            "debug",
            "Fitting PCA with the Gram covariance solver "
            f"({n_features} features, {block_size} rows per block, 0 MiB accumulator)",
        )
    assert messages == []

    _, model = _fit(rng.normal(size=(24, 4097)), block_size=8, dims=2)

    selection, summary, note = messages
    assert selection == (
        "debug",
        "Fitting PCA with the randomized solver "
        "(4097 features, 8 rows per block, blocks of 12 vectors)",
    )
    assert summary[0] == "debug"
    assert summary[1].startswith(
        "Fitted PCA of 4097 features with the randomized solver in "
        f"{model.n_passes_} passes over 24 rows"
    )
    # Wide fits name their passes at the default level, with the remedy.
    assert note == (
        "info",
        f"Fitted PCA of 4097 features with the randomized solver in "
        f"{model.n_passes_} passes over the data. At most 4096 features, such as "
        "highly variable genes, are fitted exactly in one pass.",
    )


@pytest.mark.parametrize("n_features", [6, 4097])
def test_pca_skips_blocks_without_selected_rows(n_features: int) -> None:
    values = np.random.default_rng(3).normal(size=(24, n_features))
    data = _ReadTrackingArray(values, block_size=6)
    selected = np.zeros(values.shape[0], dtype=bool)
    selected[:6] = True
    selected[-6:] = True

    _, model = fit_pca(
        data,
        dims=2,
        use_for_pca=selected,
        scale=None,
        random_state=4466,
        nthreads=1,
    )

    # Every pass reads only the blocks that hold selected rows.
    assert model.n_samples_seen_ == 12
    assert data.read_ranges == [(0, 6), (18, 24)] * model.n_passes_


@pytest.mark.parametrize("n_features", [10, 4097])
def test_pca_propagates_a_failed_eigensolve(monkeypatch, n_features: int) -> None:
    values = np.random.default_rng(8).normal(size=(24, n_features))

    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("eigensolve failed")

    monkeypatch.setattr(scipy.linalg, "eigh", fail)

    with pytest.raises(np.linalg.LinAlgError, match="eigensolve failed"):
        _fit(values, block_size=8, dims=2)


@pytest.mark.parametrize("n_features", [10, 4097])
def test_pca_reads_no_block_after_a_failed_one(n_features: int) -> None:
    values = np.random.default_rng(8).normal(size=(24, n_features))
    data = _ReadTrackingArray(values, block_size=8)
    calls = 0

    def scale(block: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise np.linalg.LinAlgError("scaling failed")
        return block

    with pytest.raises(np.linalg.LinAlgError, match="scaling failed"):
        _fit(values, block_size=8, dims=2, scale=scale, data=data)

    assert calls == 2
    assert data.read_ranges == [(0, 8), (8, 16)]


def test_lsi_returns_requested_components():
    values = np.random.default_rng(7).uniform(size=(18, 5))
    data = ChunkedArray.from_numpy(values, block_size=6)
    params = {"n_iter": 4}

    loadings = fit_lsi(
        data,
        dims=2,
        skip_first=True,
        params=params,
        random_state=3,
        nthreads=1,
    )

    assert loadings.shape == (5, 2)
    assert params == {"n_iter": 4}
    # Uncentered LSI keeps the second and third right singular vectors.
    _, _, right_vectors = np.linalg.svd(values, full_matrices=False)
    np.testing.assert_allclose(
        np.abs(loadings.T @ right_vectors[1:3].T), np.eye(2), atol=1e-8
    )


@pytest.mark.parametrize("reserved", ["n_components", "random_state"])
def test_lsi_rejects_reserved_solver_parameters(reserved):
    data = ChunkedArray.from_numpy(np.ones((6, 3)), block_size=3)
    params = {"n_iter": 1, reserved: 2}

    with pytest.raises(ValueError, match=f"cannot set {reserved}"):
        fit_lsi(
            data,
            dims=1,
            skip_first=False,
            params=params,
            random_state=0,
            nthreads=1,
        )
    assert params == {"n_iter": 1, reserved: 2}


def test_streaming_lsi_matches_exact_singular_subspace_and_is_deterministic():
    rng = np.random.default_rng(19)
    left, _ = np.linalg.qr(rng.normal(size=(60, 5)))
    right, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    values = left @ np.diag([20.0, 12.0, 7.0, 3.0, 1.0]) @ right.T
    values += 0.05 * rng.normal(size=values.shape)
    data = ChunkedArray.from_numpy(values, block_size=7)
    params = {
        "solver": "streaming",
        "n_iter": 3,
        "n_oversamples": 4,
    }

    observed = fit_lsi(
        data,
        dims=4,
        skip_first=False,
        params=params,
        random_state=11,
        nthreads=1,
    )
    repeated = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=9),
        dims=4,
        skip_first=False,
        params=params.copy(),
        random_state=11,
        nthreads=1,
    )
    _, _, right_vectors = np.linalg.svd(values, full_matrices=False)
    expected = right_vectors[:4].T

    np.testing.assert_allclose(
        observed @ observed.T,
        expected @ expected.T,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(observed, repeated, rtol=1e-10, atol=1e-10)
    assert data.numblocks[0] > 1


def test_streaming_lsi_never_stacks_all_input_blocks(monkeypatch):
    values = np.random.default_rng(23).normal(size=(30, 12))

    def fail_vstack(*_args, **_kwargs):
        pytest.fail("Streaming LSI must not stack the full input matrix")

    monkeypatch.setattr(np, "vstack", fail_vstack)
    loadings = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=4),
        dims=3,
        skip_first=True,
        params={"n_iter": 1},
        random_state=3,
        nthreads=1,
    )

    assert loadings.shape == (values.shape[1], 3)


def test_streaming_lsi_reserves_accumulators_and_block_temporaries():
    values = np.random.default_rng(31).normal(size=(30, 12))
    resources = ResourceBudget(memoryBytes=10_000, workers=4)
    data = _ResidentTrackingArray(
        values,
        block_size=4,
        resources=resources,
    )

    fit_lsi(
        data,
        dims=3,
        skip_first=True,
        params={
            "solver": "streaming",
            "n_iter": 1,
            "n_oversamples": 2,
        },
        random_state=5,
        nthreads=4,
    )

    expected = _streaming_lsi_resident_bytes(data, width=6)
    assert data.resident_bytes == [expected, expected, expected]
    assert expected + data._block_owned_bytes() <= resources.memoryBytes


def test_materialized_lsi_preserves_sklearn_compatibility():
    values = np.random.default_rng(29).normal(size=(28, 9))
    expected_model = TruncatedSVD(
        n_components=4,
        n_iter=3,
        n_oversamples=2,
        random_state=7,
    )
    expected_model.fit(values)

    observed = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=5),
        dims=3,
        skip_first=True,
        params={
            "solver": "materialized",
            "n_iter": 3,
            "n_oversamples": 2,
        },
        random_state=7,
        nthreads=1,
    )

    np.testing.assert_allclose(
        observed,
        expected_model.components_.T[:, 1:],
        rtol=1e-12,
        atol=1e-12,
    )


def test_materialized_lsi_does_not_depend_on_block_size():
    values = np.random.default_rng(37).normal(size=(28, 9))
    expected = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=28),
        dims=3,
        skip_first=True,
        params={"solver": "materialized", "n_iter": 3, "n_oversamples": 2},
        random_state=7,
        nthreads=1,
    )
    observed = fit_lsi(
        ChunkedArray.from_numpy(values, block_size=5),
        dims=3,
        skip_first=True,
        params={"solver": "materialized", "n_iter": 3, "n_oversamples": 2},
        random_state=7,
        nthreads=1,
    )

    np.testing.assert_array_equal(observed, expected)


def test_materialized_lsi_budget_counts_matrix_copies_and_svd_bases():
    required = reduction_module._materialized_lsi_bytes(
        n_rows=100,
        n_features=20,
        itemsize=4,
        n_components=4,
        n_oversamples=6,
    )

    assert required == 2 * 100 * 20 * 4 + 4 * 120 * 10 * 8 + 2 * 10 * 10 * 8
    reduction_module.require_materialized_lsi_budget(
        n_rows=100,
        n_features=20,
        itemsize=4,
        n_components=4,
        n_oversamples=6,
        memory_bytes=required,
    )
    with pytest.raises(MemoryError, match="solver='streaming'"):
        reduction_module.require_materialized_lsi_budget(
            n_rows=100,
            n_features=20,
            itemsize=4,
            n_components=4,
            n_oversamples=6,
            memory_bytes=required - 1,
        )
