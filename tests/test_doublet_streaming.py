import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cytearc import DataStore
from cytearc.mapping.features import AlignedFeatureStream
from cytearc.mapping.models import ScaledPCAProjectionModel
from cytearc.mapping.symphony import project_pca
from cytearc.matrix import ChunkedArray
from cytearc.quality_control import doublets
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.count_matrix import CountMatrixPolicy

from .doublet_fixtures import write_doublet_target_zarr

# An index this deep finds the exact neighbors of the few points of a test.
_EXACT_INDEX = doublets.DoubletIndexSettings(
    metric="l2", m=16, ef_construction=200, ef=200, parallel=False
)


@pytest.mark.parametrize(
    "save_k,error,message",
    [
        (0, ValueError, "n_neighbors must be at least 1"),
        (-1, ValueError, "n_neighbors must be at least 1"),
        (True, TypeError, "n_neighbors must be an integer"),
        (1.5, TypeError, "n_neighbors must be an integer"),
    ],
)
def test_doublet_scoring_rejects_invalid_k_before_reading_counts(
    save_k, error, message
):
    # Without an assay or a reference, any read would fail differently.
    with pytest.raises(error, match=message):
        doublets.score_synthetic_doublets(
            None,
            None,
            np.arange(2),
            np.array([0, 1]),
            np.zeros(2, dtype=np.int64),
            np.arange(3),
            None,
            index=_EXACT_INDEX,
            cluster_sample_fraction=1,
            max_cells_per_cluster=None,
            simulation_ratio=1,
            heterotypic_fraction=0.8,
            n_neighbors=save_k,
            expected_doublet_rate=0.05,
            threshold=None,
            random_seed=91,
            resources=ResourceBudget(1_000_000, 1),
        )


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("detected", [[0, 9, 10, 11], [9, 10, 10, 11], [10, 10]])
def test_simulated_doublets_keep_every_pair_and_add_parent_libraries(
    monkeypatch, detected, subset
):
    # Doublets of parents with few detected features were once left out, and
    # parents with ten features each raised. Every pair is now simulated,
    # whatever the batches it is computed in.
    counts = np.zeros((len(detected), 14), dtype=np.int16)
    for row, n in enumerate(detected):
        counts[row, :n] = row + 1
    features = np.array([0, 3, 5, 9, 12])
    rng = np.random.default_rng(17)
    model = ScaledPCAProjectionModel(
        feature_means=rng.normal(size=len(features)),
        feature_scales=rng.uniform(0.5, 2.0, size=len(features)),
        center=rng.normal(size=len(features)),
        loadings=rng.normal(size=(len(features), 3)),
    )
    reference = SimpleNamespace(
        model=model,
        normalization_parameters={
            "size_factor": 100.0,
            "log_transform": True,
            "renormalize_subset": subset,
        },
    )
    assay = SimpleNamespace(rawData=ChunkedArray.from_numpy(counts), name="RNA")
    # Library sizes count features outside the selection too.
    totals = None if subset else counts.sum(axis=1, dtype=np.float64) + 7
    members = np.arange(len(counts))
    (plan,) = doublets.plan_doublet_samples(
        [len(counts)], simulation_ratio=3.0, n_neighbors=1
    )
    original = doublets._doublet_batch_rows
    results = []
    for rows in (1, 3, 100):
        monkeypatch.setattr(
            doublets,
            "_doublet_batch_rows",
            lambda *args, rows=rows, **kwargs: min(rows, original(*args, **kwargs)),
        )
        results.append(
            doublets._simulate_sample(
                assay,
                reference,
                np.random.default_rng(5),
                members,
                plan,
                active_indices=members,
                labels=np.zeros(len(counts), dtype=np.int64),
                feature_indices=features,
                cell_totals=totals,
                cluster_sample_fraction=1.0,
                max_cells_per_cluster=None,
                heterotypic_fraction=0.0,
                resources=ResourceBudget(1_000_000, 1),
                resident_bytes=0,
            )
        )
    coordinates, parents, fraction = results[0]
    assert coordinates.shape == (plan.simulated, 3) == (3 * len(counts), 3)
    assert coordinates.dtype == np.float32
    assert fraction == 0.0
    left, right = parents.T
    summed = (counts[left][:, features] + counts[right][:, features]).astype(float)
    library = summed.sum(axis=1) if subset else totals[left] + totals[right]
    # A doublet without counts normalizes to zero.
    library[library == 0] = 1
    expected = (
        (np.log1p(100.0 * summed / library[:, None]) - model.feature_means)
        / model.feature_scales
        - model.center
    ) @ model.loadings
    np.testing.assert_allclose(coordinates, expected, rtol=1e-5, atol=1e-5)
    for repeated, repeated_parents, _ in results[1:]:
        np.testing.assert_array_equal(repeated, coordinates)
        np.testing.assert_array_equal(repeated_parents, parents)


def test_doublet_parent_pool_is_read_in_admitted_blocks(monkeypatch):
    counts = np.arange(20 * 19, dtype=np.uint16).reshape(20, 19)
    raw = ChunkedArray.from_numpy(counts)
    rows = np.array([1, 3, 4, 7, 10, 11, 12, 16])
    sizes = []
    original = doublets.csr_matrix

    def observe(values):
        sizes.append(len(values))
        return original(values)

    monkeypatch.setattr(doublets, "csr_matrix", observe)
    every = np.arange(counts.shape[1])
    result = doublets._load_parent_counts(raw, rows, every, ResourceBudget(4_096, 1), 0)
    np.testing.assert_array_equal(result.toarray(), counts[rows])
    assert len(sizes) > 1
    assert max(sizes) < len(rows)
    # Only the selected features of each parent are kept.
    columns = np.array([2, 5, 11, 18])
    subset = doublets._load_parent_counts(
        raw, rows, columns, ResourceBudget(4_096, 1), 0
    )
    np.testing.assert_array_equal(subset.toarray(), counts[np.ix_(rows, columns)])
    with pytest.raises(MemoryError, match="operation limit is 100 bytes"):
        doublets._load_parent_counts(raw, rows, every, ResourceBudget(100, 1), 0)


@pytest.mark.parametrize("power", [1, 2, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_doublet_smoothing_matches_powered_diffusion_with_hub_and_isolate(power, dtype):
    dense = np.zeros((101, 101), dtype=dtype)
    dense[0, 1:100] = dense[1:100, 0] = 0.5
    graph = csr_matrix(dense)
    scores = np.linspace(0, 1, len(dense))
    inverse_degree = np.ravel(graph.sum(axis=1))
    inverse_degree[inverse_degree != 0] = 1 / inverse_degree[inverse_degree != 0]
    diagonal = csr_matrix(
        (inverse_degree, (range(len(dense)), range(len(dense)))),
        shape=[len(dense), len(dense)],
    )
    # Smoothing formerly stepped this operator in COO form. Its values must
    # not change, so doublet scores and their artifacts stay identical.
    step = diagonal.dot(graph).tocoo()
    previous = scores
    for _ in range(power):
        previous = np.asarray(step.dot(previous), dtype=np.float64)
    expected = (diagonal.dot(graph) ** power).dot(scores)
    actual = doublets.smooth_doublet_scores(graph, scores, power=power, normalize=False)
    np.testing.assert_array_equal(actual, previous)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert actual[-1] == 0
    constant = doublets.smooth_doublet_scores(
        graph,
        np.zeros(len(dense)),
        power=power,
        normalize=True,
    )
    np.testing.assert_array_equal(constant, 0)


@pytest.mark.parametrize(
    "power,error", [(0, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_doublet_smoothing_rejects_invalid_power(power, error):
    with pytest.raises(error, match="positive integer"):
        doublets.smooth_doublet_scores(
            csr_matrix(np.eye(2)), np.ones(2), power=power, normalize=True
        )


def _write_counts(path: Path, counts) -> None:
    ids = np.array([f"g{i}" for i in range(counts.shape[1])])
    write_doublet_target_zarr(
        str(path),
        "RNA",
        csr_matrix(counts),
        ids,
        ids,
        dtype=str(counts.dtype),
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=16_384, chunkBytes=1024),
    )


# The scorer's settings that the stores below simulate doublets for.
_SEED = 91
_RATIO = 2.0
_HETEROTYPIC = 0.8


@dataclass(frozen=True)
class _DoubletStores:
    """A reference store and a store holding the doublets it simulates."""

    counts: np.ndarray
    store: DataStore
    query: DataStore
    simulated: csr_matrix
    parents: np.ndarray


def _doublet_stores(directory: Path, variant: str) -> _DoubletStores:
    rng = np.random.default_rng(73)
    counts = (
        np.full((30, 32), 5, dtype=np.uint16)
        if variant == "constant"
        else rng.integers(0, 200, (30, 24), dtype=np.uint16)
    )
    if variant == "filtered":
        counts[:15, 4:] = 0
    _write_counts(directory / "reference.zarr", counts)
    store = DataStore(
        str(directory / "reference.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    # The simulation below repeats the one inside score_synthetic_doublets:
    # one sample, every cell a parent, and the sample's child generator.
    labels = np.arange(len(counts)) % 3
    rng = np.random.default_rng(np.random.SeedSequence(_SEED).spawn(1)[0])
    left, right = doublets.simulate_doublet_pairs(
        labels, int(round(_RATIO * len(counts))), _HETEROTYPIC, rng
    )
    simulated = doublets.sum_doublet_pairs(csr_matrix(counts), left, right)
    _write_counts(directory / "query.zarr", simulated)
    query = DataStore(str(directory / "query.zarr"), default_assay="RNA", nthreads=1)
    return _DoubletStores(
        counts, store, query, simulated, np.column_stack((left, right))
    )


@pytest.fixture(scope="module")
def doublet_stores(tmp_path_factory):
    """Build each count variant's stores once; normalizations share them."""
    built: dict[str, _DoubletStores] = {}

    def get(variant: str) -> _DoubletStores:
        if variant not in built:
            built[variant] = _doublet_stores(tmp_path_factory.mktemp(variant), variant)
        return built[variant]

    return get


def _likelihood(counts: np.ndarray, neighbors: int, ratio: float) -> np.ndarray:
    """Scrublet's doublet probability, written out for the tests."""
    rate = 0.05
    fraction = (counts + 1) / (neighbors + 2)
    doublet = fraction * rate / ratio
    return doublet / (doublet + (1 - fraction) * (1 - rate))


def _exact_simulated_neighbors(
    points: np.ndarray, n_observed: int, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bound the simulated doublets among each point's k nearest other points.

    The bounds are equal unless the k-th distance ties others, to within the
    float32 rounding of hnswlib's distances; a tie may then go either way.
    """
    values = points.astype(np.float64)
    distances = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    kth = np.sort(distances, axis=1)[:, k - 1 : k]
    tolerance = 1e-5 * np.maximum(kth, 1e-12)
    certain = distances < kth - tolerance
    tied = np.abs(distances - kth) <= tolerance
    simulated = np.arange(len(points)) >= n_observed
    known = np.count_nonzero(certain & simulated, axis=1)
    free = k - np.count_nonzero(certain, axis=1)
    tied_simulated = np.count_nonzero(tied & simulated, axis=1)
    tied_observed = np.count_nonzero(tied & ~simulated, axis=1)
    low = known + np.maximum(0, free - tied_observed)
    high = known + np.minimum(tied_simulated, free)
    return low, high


def _assert_within(counts: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    low, high = bounds
    assert np.all((low <= counts) & (counts <= high))


@pytest.mark.parametrize(
    "log,subset,constant,filtered",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_streamed_doublets_match_materialized_doublets(
    doublet_stores, monkeypatch, log, subset, constant, filtered
):
    variant = "constant" if constant else "filtered" if filtered else "random"
    stores = doublet_stores(variant)
    counts, store, query = stores.counts, stores.store, stores.query
    cells = store.snapshot_cell_selection("I")
    feature_indices = np.arange(0, counts.shape[1], 2)
    features = store.features.snapshot(feature_indexes=feature_indices)
    normalized = store.features.normalize(
        cells, features, log_transform=log, renormalize_subset=subset
    )
    pca = store.reduction.pca(normalized, dims=3, local_cache=False)
    ann = store.graph.ann_index(pca)
    neighbors = store.graph.neighbors(ann, coordinates=pca, k=4)
    reference = store.mapping.load_reference(store.mapping.build_reference(neighbors))
    labels = np.arange(len(counts)) % 3
    rows = np.arange(len(counts))
    totals = None if subset else store.RNA._cell_count_totals(rows)
    options = dict(
        index=_EXACT_INDEX,
        cluster_sample_fraction=1.0,
        max_cells_per_cluster=None,
        simulation_ratio=_RATIO,
        heterotypic_fraction=_HETEROTYPIC,
        n_neighbors=4,
        expected_doublet_rate=0.05,
        threshold=None,
        random_seed=_SEED,
        resources=ResourceBudget(128 * 1024**2, 1),
    )
    if filtered:
        # Doublets with ten or fewer features were once left unscored.
        detected = np.asarray((stores.simulated > 0).sum(axis=1)).ravel()
        assert np.median(detected) > 10
        assert np.any(detected <= 10)
    captured: list[np.ndarray] = []
    counted: list[np.ndarray] = []
    count_neighbors = doublets.count_simulated_neighbors

    def capture(points, *args, **kwargs):
        captured.append(np.array(points))
        counted.append(count_neighbors(points, *args, **kwargs))
        return counted[-1]

    monkeypatch.setattr(doublets, "count_simulated_neighbors", capture)
    actual = doublets.score_synthetic_doublets(
        store.RNA,
        reference,
        rows,
        labels,
        np.zeros(len(rows), dtype=np.int64),
        feature_indices,
        totals,
        **options,
    )
    (points,) = captured
    n_cells = len(counts)
    n_simulated = len(stores.parents)
    assert points.shape == (n_cells + n_simulated, 3)
    np.testing.assert_array_equal(actual.simulated_parents, stores.parents)
    # The observed rows are the reference PCA of the cells.
    observed = np.asarray(artifact_group(store.zw, reference.reduction)["data"][:])
    np.testing.assert_array_equal(points[:n_cells], observed)
    # The streamed doublets are the materialized doublets, normalized by the
    # mapping stream and projected onto the reference PCA.
    stream = AlignedFeatureStream(
        query.RNA,
        np.arange(query.cells.N),
        reference.feature_ids,
        np.zeros(len(reference.feature_ids)),
        reference.normalization_parameters,
        "error",
        ResourceBudget(128 * 1024**2, 1),
    )
    expected = project_pca(
        np.vstack([block.values for block in stream.iter_blocks()]), reference.model
    )
    np.testing.assert_allclose(points[n_cells:], expected, rtol=1e-5, atol=1e-5)
    k = round(4 * (1 + n_simulated / n_cells))
    assert actual.samples[0].plan == doublets.DoubletSamplePlan(
        cells=n_cells, simulated=n_simulated, neighbors=4, adjusted_neighbors=k
    )
    (counts_used,) = counted
    bounds = _exact_simulated_neighbors(points, n_cells, k)
    _assert_within(counts_used, bounds)
    if constant:
        # Doublets of identical cells project onto the cells themselves, so
        # every neighbor is a tie.
        np.testing.assert_allclose(
            points[n_cells:], np.broadcast_to(points[0], (n_simulated, 3)), atol=1e-6
        )
    else:
        # Away from ties, the counts are those of the exact neighbors.
        assert np.mean(bounds[0] == bounds[1]) > 0.9
    np.testing.assert_array_equal(actual.raw_scores, counts_used[:n_cells] / k)
    np.testing.assert_allclose(
        actual.scores, _likelihood(counts_used[:n_cells], k, _RATIO), rtol=1e-12
    )
    np.testing.assert_allclose(
        actual.simulated_scores,
        _likelihood(counts_used[n_cells:], k, _RATIO),
        rtol=1e-12,
    )
    (sample,) = actual.samples
    np.testing.assert_array_equal(actual.predicted, actual.scores > sample.threshold)

    # Smaller batches stream the same doublets.
    original = doublets._doublet_batch_rows

    def smaller_batches(*args, **kwargs):
        return min(1 if filtered else 2, original(*args, **kwargs))

    monkeypatch.setattr(doublets, "_doublet_batch_rows", smaller_batches)
    repeated = doublets.score_synthetic_doublets(
        store.RNA,
        reference,
        rows,
        labels,
        np.zeros(len(rows), dtype=np.int64),
        feature_indices,
        totals,
        **options,
    )
    np.testing.assert_allclose(captured[1], points, rtol=1e-6, atol=1e-6)
    _assert_within(counted[1], _exact_simulated_neighbors(captured[1], n_cells, k))
    untied = bounds[0] == bounds[1]
    np.testing.assert_array_equal(counted[1][untied], counts_used[untied])
    np.testing.assert_array_equal(repeated.simulated_parents, actual.simulated_parents)
    # A budget far below the cells' coordinates and their index refuses.
    with pytest.raises(MemoryError, match="operation limit is 1000 bytes"):
        doublets.score_synthetic_doublets(
            store.RNA,
            reference,
            rows,
            labels,
            np.zeros(len(rows), dtype=np.int64),
            feature_indices,
            totals,
            **(options | {"resources": ResourceBudget(1_000, 1)}),
        )

    # Pair generation is admitted before any pair is drawn.
    def unadmitted(*_args, **_kwargs):
        raise AssertionError("doublet pairs were drawn before their admission")

    monkeypatch.setattr(doublets, "simulate_doublet_pairs", unadmitted)
    many = {"simulation_ratio": 1e4, "resources": ResourceBudget(1_000_000, 1)}
    with pytest.raises(MemoryError, match="operation limit is 1000000 bytes"):
        doublets.score_synthetic_doublets(
            store.RNA,
            reference,
            rows,
            labels,
            np.zeros(len(rows), dtype=np.int64),
            feature_indices,
            totals,
            **(options | many),
        )


def test_doublets_without_selected_feature_counts_are_scored(tmp_path, monkeypatch):
    from cytearc.mapping.features import normalize_reference_counts

    counts = np.random.default_rng(29).integers(1, 200, (30, 24), dtype=np.uint16)
    labels = np.arange(len(counts)) % 3
    feature_indices = np.arange(0, counts.shape[1], 2)
    # Cluster 0 has counts only outside the selected features.
    counts[np.ix_(labels == 0, feature_indices)] = 0
    ids = np.array([f"g{i}" for i in range(counts.shape[1])])
    path = str(tmp_path / "reference.zarr")
    write_doublet_target_zarr(
        path,
        "RNA",
        csr_matrix(counts),
        ids,
        ids,
        dtype=str(counts.dtype),
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=16_384, chunkBytes=1024),
    )
    store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)
    features = store.features.snapshot(feature_indexes=feature_indices)
    normalized = store.features.normalize(
        store.snapshot_cell_selection("I"), features, renormalize_subset=False
    )
    pca = store.reduction.pca(normalized, dims=3, local_cache=False)
    neighbors = store.graph.neighbors(store.graph.ann_index(pca), coordinates=pca, k=4)
    reference = store.mapping.load_reference(store.mapping.build_reference(neighbors))
    rng = np.random.default_rng(np.random.SeedSequence(91).spawn(1)[0])
    left, right = doublets.simulate_doublet_pairs(labels, 60, 0.0, rng)
    simulated = doublets.sum_doublet_pairs(csr_matrix(counts), left, right)
    empty = np.asarray(simulated[:, feature_indices].sum(axis=1)).ravel() == 0
    assert empty.any()
    queried = []
    count_neighbors = doublets.count_simulated_neighbors

    def recording_count(points, *args, **kwargs):
        queried.append(np.array(points))
        return count_neighbors(points, *args, **kwargs)

    monkeypatch.setattr(doublets, "count_simulated_neighbors", recording_count)
    rows = np.arange(len(counts))
    scores = doublets.score_synthetic_doublets(
        store.RNA,
        reference,
        rows,
        labels,
        np.zeros(len(rows), dtype=np.int64),
        feature_indices,
        store.RNA._cell_count_totals(rows),
        index=_EXACT_INDEX,
        cluster_sample_fraction=1.0,
        max_cells_per_cluster=None,
        simulation_ratio=2.0,
        heterotypic_fraction=0.0,
        n_neighbors=4,
        expected_doublet_rate=0.05,
        threshold=None,
        random_seed=91,
        resources=ResourceBudget(128 * 1024**2, 1),
    )

    # Every simulated doublet is a point of the sample's index, including
    # those empty in the selected features, which sit at the projection of an
    # empty profile.
    (points,) = queried
    simulated_points = points[len(counts) :]
    assert len(simulated_points) == simulated.shape[0]
    np.testing.assert_array_equal(
        scores.simulated_parents, np.column_stack((left, right))
    )
    parameters = reference.normalization_parameters
    empty_projection = project_pca(
        normalize_reference_counts(
            np.zeros((1, len(feature_indices))),
            size_factor=parameters["size_factor"],
            log_transform=parameters["log_transform"],
            source="The doublet simulation of RNA",
        ),
        reference.model,
    )
    assert np.linalg.norm(empty_projection) > 0
    at_empty = np.isclose(simulated_points, empty_projection, rtol=0, atol=1e-6).all(
        axis=1
    )
    np.testing.assert_array_equal(at_empty, empty)
    assert np.all(np.isfinite(scores.scores)) and np.any(scores.scores > 0)
    assert np.all(np.isfinite(scores.simulated_scores))
    # The scored cells must be the reference's rows.
    with pytest.raises(ValueError, match="Reference coordinates do not match"):
        doublets.score_synthetic_doublets(
            store.RNA,
            reference,
            rows[:-1],
            labels[:-1],
            np.zeros(len(rows) - 1, dtype=np.int64),
            feature_indices,
            store.RNA._cell_count_totals(rows[:-1]),
            index=_EXACT_INDEX,
            cluster_sample_fraction=1.0,
            max_cells_per_cluster=None,
            simulation_ratio=2.0,
            heterotypic_fraction=0.0,
            n_neighbors=4,
            expected_doublet_rate=0.05,
            threshold=None,
            random_seed=91,
            resources=ResourceBudget(128 * 1024**2, 1),
        )


@dataclass(frozen=True)
class _Samples:
    """A reference store of four samples and what scoring them takes."""

    store: DataStore
    reference: object
    labels: np.ndarray
    codes: np.ndarray
    features: np.ndarray
    totals: np.ndarray


_SAMPLE_CELLS = 60


@pytest.fixture(scope="module")
def four_samples(tmp_path_factory) -> _Samples:
    """Four samples of 60 cells, each of two of four cell types."""
    rng = np.random.default_rng(57)
    profiles = rng.gamma(1.0, 2.0, size=(4, 24))
    half = _SAMPLE_CELLS // 2
    types = np.concatenate(
        [np.repeat([sample, (sample + 1) % 4], half) for sample in range(4)]
    )
    counts = rng.poisson(3 * profiles[types]).astype(np.uint16)
    path = tmp_path_factory.mktemp("four_samples") / "reference.zarr"
    _write_counts(path, counts)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    features = np.arange(counts.shape[1])
    normalized = store.features.normalize(
        store.snapshot_cell_selection("I"),
        store.features.snapshot(feature_indexes=features),
        renormalize_subset=False,
    )
    pca = store.reduction.pca(normalized, dims=4, local_cache=False)
    neighbors = store.graph.neighbors(store.graph.ann_index(pca), coordinates=pca, k=5)
    reference = store.mapping.load_reference(store.mapping.build_reference(neighbors))
    return _Samples(
        store,
        reference,
        types,
        np.repeat(np.arange(4), _SAMPLE_CELLS),
        features,
        store.RNA._cell_count_totals(np.arange(len(counts))),
    )


def _score_samples(
    samples: _Samples,
    resources: ResourceBudget,
    codes: np.ndarray | None = None,
    n_neighbors: int = 5,
) -> doublets.DoubletScores:
    return doublets.score_synthetic_doublets(
        samples.store.RNA,
        samples.reference,
        np.arange(len(samples.codes)),
        samples.labels,
        samples.codes if codes is None else codes,
        samples.features,
        samples.totals,
        index=_EXACT_INDEX,
        cluster_sample_fraction=1.0,
        max_cells_per_cluster=None,
        simulation_ratio=2.0,
        heterotypic_fraction=0.0,
        n_neighbors=n_neighbors,
        expected_doublet_rate=0.05,
        threshold=None,
        random_seed=_SEED,
        resources=resources,
    )


def _assert_same_scores(
    actual: doublets.DoubletScores, expected: doublets.DoubletScores
) -> None:
    for name in (
        "scores",
        "raw_scores",
        "predicted",
        "simulated_scores",
        "simulated_parents",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    # The thresholds, calls and detectable fractions of each sample, exactly.
    assert actual.samples == expected.samples


_LARGE_BUDGET = 256 * 1024**2


def _scoring_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("cytearc-doublets")
    ]


def test_samples_scored_at_once_have_the_scores_of_samples_scored_one_at_a_time(
    four_samples, monkeypatch
):
    one_at_a_time = _score_samples(four_samples, ResourceBudget(_LARGE_BUDGET, 1))
    build_threads: list[int] = []
    count_neighbors = doublets.count_simulated_neighbors
    instantiate = doublets.instantiate_knn_index
    # Every sample waits here for the other three, which only samples scored
    # at once can pass.
    meeting = threading.Barrier(4, timeout=60)

    def meet(*args, **kwargs):
        meeting.wait()
        return count_neighbors(*args, **kwargs)

    def record_build_threads(*args):
        build_threads.append(args[-1])
        return instantiate(*args)

    monkeypatch.setattr(doublets, "count_simulated_neighbors", meet)
    monkeypatch.setattr(doublets, "instantiate_knn_index", record_build_threads)
    at_once = _score_samples(four_samples, ResourceBudget(_LARGE_BUDGET, 4))

    # Each index is built on one thread, so it is the index of one at a time.
    assert build_threads == [1, 1, 1, 1]
    _assert_same_scores(at_once, one_at_a_time)
    assert len(at_once.samples) == 4
    assert _scoring_threads() == []


def test_samples_scored_at_once_are_as_many_as_the_budget_holds(
    four_samples, monkeypatch
):
    observed: dict[str, object] = {}
    plan_samples = doublets._samples_in_flight

    def observe(plans, dims, settings, resources, resident_bytes):
        observed.update(plans=plans, dims=dims, settings=settings)
        observed["resident"] = resident_bytes
        return plan_samples(plans, dims, settings, resources, resident_bytes)

    monkeypatch.setattr(doublets, "_samples_in_flight", observe)
    # Each cell counts its 59 other cells, so a sample's query block, which
    # its charge holds with its points and index, is larger than the batches
    # of a simulation, and a simulation fits beside one sample being scored.
    one_at_a_time = _score_samples(
        four_samples, ResourceBudget(_LARGE_BUDGET, 1), n_neighbors=59
    )
    # Each sample charges its points, its index and its query block.
    charge = max(
        doublets._sample_charge(plan, observed["dims"], observed["settings"], 2)
        for plan in observed["plans"]
    )
    holds_two = observed["resident"] + 2 * charge
    count_neighbors = doublets.count_simulated_neighbors
    lock = threading.Lock()
    scoring = [0, 0]

    def track(meeting: threading.Barrier | None):
        def counted(*args, **kwargs):
            with lock:
                scoring[0] += 1
                scoring[1] = max(scoring)
            try:
                if meeting is not None:
                    meeting.wait()
                return count_neighbors(*args, **kwargs)
            finally:
                with lock:
                    scoring[0] -= 1

        return counted

    # Two samples fit, so they are scored in pairs that meet, never three at
    # once, and the simulations wait for the memory that the pairs hold.
    monkeypatch.setattr(
        doublets, "count_simulated_neighbors", track(threading.Barrier(2, timeout=60))
    )
    in_pairs = _score_samples(
        four_samples, ResourceBudget(holds_two, 4), n_neighbors=59
    )
    assert scoring == [0, 2]
    _assert_same_scores(in_pairs, one_at_a_time)

    # A byte less holds one sample, so the samples are scored one at a time.
    scoring[:] = [0, 0]
    monkeypatch.setattr(doublets, "count_simulated_neighbors", track(None))
    alone = _score_samples(
        four_samples, ResourceBudget(holds_two - 1, 4), n_neighbors=59
    )
    assert scoring == [0, 1]
    _assert_same_scores(alone, one_at_a_time)


def test_one_sample_is_scored_alone_on_every_worker(four_samples, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("one sample was scored beside others")

    workers: list[int] = []
    count_neighbors = doublets.count_simulated_neighbors

    def record_workers(*args, **kwargs):
        workers.append(kwargs["workers"])
        return count_neighbors(*args, **kwargs)

    monkeypatch.setattr(doublets, "_score_samples_at_once", forbidden)
    monkeypatch.setattr(doublets, "count_simulated_neighbors", record_workers)
    codes = np.zeros(len(four_samples.codes), dtype=np.int64)

    scores = _score_samples(four_samples, ResourceBudget(_LARGE_BUDGET, 4), codes)

    assert workers == [4]
    assert len(scores.samples) == 1


def test_a_failed_sample_stops_the_samples_scored_at_once(four_samples, monkeypatch):
    count_neighbors = doublets.count_simulated_neighbors
    simulate = doublets._simulate_sample
    lock = threading.Lock()
    calls = [0]

    def fail_second(*args, **kwargs):
        with lock:
            calls[0] += 1
            second = calls[0] == 2
        if second:
            raise RuntimeError("injected neighbor failure")
        return count_neighbors(*args, **kwargs)

    monkeypatch.setattr(doublets, "count_simulated_neighbors", fail_second)
    with pytest.raises(RuntimeError, match="injected neighbor failure"):
        _score_samples(four_samples, ResourceBudget(_LARGE_BUDGET, 4))
    assert _scoring_threads() == []

    # A simulation that fails while other samples are scored ends the call
    # once they finish.
    monkeypatch.setattr(doublets, "count_simulated_neighbors", count_neighbors)

    def fail_third(assay, reference, rng, members, plan, **kwargs):
        if members[0] == 2 * _SAMPLE_CELLS:
            raise RuntimeError("injected simulation failure")
        return simulate(assay, reference, rng, members, plan, **kwargs)

    monkeypatch.setattr(doublets, "_simulate_sample", fail_third)
    with pytest.raises(RuntimeError, match="injected simulation failure"):
        _score_samples(four_samples, ResourceBudget(_LARGE_BUDGET, 4))
    assert _scoring_threads() == []


def test_samples_scored_at_once_refuse_where_one_at_a_time_refuses(
    four_samples, monkeypatch
):
    from cytearc.utils.memory_budget import MemoryBudgetError

    def outcome(budget: int) -> doublets.DoubletScores | tuple[str, int]:
        try:
            return _score_samples(four_samples, ResourceBudget(budget, 4))
        except MemoryBudgetError as error:
            return str(error), error.needed_bytes

    # From budgets that refuse the first simulation to budgets that score two
    # and four samples at once.
    budgets = [5_000, 40_000, 120_000, 600_000, 1_000_000, 4_000_000, _LARGE_BUDGET]
    at_once = [outcome(budget) for budget in budgets]
    monkeypatch.setattr(doublets, "_samples_in_flight", lambda *_args: (1, 4))
    one_at_a_time = [outcome(budget) for budget in budgets]

    refused = 0
    for actual, expected in zip(at_once, one_at_a_time, strict=True):
        if isinstance(expected, tuple):
            refused += 1
            assert actual == expected
        else:
            assert isinstance(actual, doublets.DoubletScores)
            _assert_same_scores(actual, expected)
    assert 0 < refused < len(budgets)


class _RecordedQueries:
    """An hnswlib index that records the points of each query."""

    def __init__(self, index, queried: list[int]) -> None:
        self._index = index
        self._queried = queried

    def add_items(self, *args, **kwargs):
        return self._index.add_items(*args, **kwargs)

    def set_num_threads(self, threads):
        return self._index.set_num_threads(threads)

    def knn_query(self, values, k):
        self._queried.append(len(values))
        return self._index.knn_query(values, k=k)


def test_a_sample_charge_holds_its_index_and_a_bounded_query_block(monkeypatch):
    plan = doublets.DoubletSamplePlan(
        cells=2_000, simulated=4_000, neighbors=10, adjusted_neighbors=30
    )
    points = np.random.default_rng(3).normal(size=(6_000, 3)).astype(np.float32)
    queried: list[int] = []
    instantiate = doublets.instantiate_knn_index
    monkeypatch.setattr(
        doublets,
        "instantiate_knn_index",
        lambda *args: _RecordedQueries(instantiate(*args), queried),
    )
    settings = doublets.DoubletIndexSettings("l2", 16, 50, 50, False)
    charge = doublets._sample_charge(plan, 3, settings, 2)
    # The scoring holds the cells' int64 positions beside the charge's other
    # parts, and the call's own resident bytes beside them.
    resident = 1_000 + 8 * plan.cells

    def count(budget: int) -> np.ndarray:
        return doublets.count_simulated_neighbors(
            points,
            plan.cells,
            plan.adjusted_neighbors,
            settings,
            random_seed=1,
            workers=2,
            resources=ResourceBudget(budget, 2),
            resident_bytes=resident,
        )

    counts = count(1_000 + charge)
    assert queried == [4_096, 6_000 - 4_096]
    # A byte less holds a block of one point fewer.
    queried.clear()
    np.testing.assert_array_equal(count(1_000 + charge - 1), counts)
    assert queried == [4_095, 6_000 - 4_095]
