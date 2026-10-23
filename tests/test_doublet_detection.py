"""Doublet scores, thresholds, and calls from simulated doublets per sample.

The unit tests check the likelihood, the thresholds, the per-sample plan, and
the neighbor counts against hand computations and exact neighbors. The store
tests inject heterotypic and homotypic doublets into synthetic counts of two
samples and check what ``qc.doublets`` recovers, that each sample simulates
and scores doublets alone, and that the scores are stable across seeds.
"""

import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import cytearc.storage.operation_revisions as operation_revisions
from cytearc import DataStore
from cytearc.datastore._pipeline_recipe import resolve_pipeline_recipe
from cytearc.quality_control import doublets
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.selections import read_stored_selection_mask
from cytearc.utils.logging import logger
from tests.storage_helpers import insert_nullable_cell_column, write_count_store

_INDEX = doublets.DoubletIndexSettings(
    metric="l2", m=16, ef_construction=100, ef=50, parallel=False
)


def _likelihood(counts, neighbors: int, rate: float, ratio: float) -> np.ndarray:
    """Bayes' rule of the docstring, written out independently."""
    fraction = (np.asarray(counts, dtype=float) + 1) / (neighbors + 2)
    odds = fraction / (1 - fraction) / ratio * rate / (1 - rate)
    return odds / (1 + odds)


def test_plan_doublet_samples_scales_and_caps_the_neighbors_of_each_sample():
    plans = doublets.plan_doublet_samples(
        [100, 10, 2, 7], simulation_ratio=2.0, n_neighbors=30
    )

    assert plans == (
        doublets.DoubletSamplePlan(100, 200, 30, 90),
        # A sample counts at most its other cells.
        doublets.DoubletSamplePlan(10, 20, 9, 27),
        doublets.DoubletSamplePlan(2, 4, 1, 3),
        doublets.DoubletSamplePlan(7, 14, 6, 18),
    )
    # One simulated doublet at least, and ties round to the even integer.
    (few,) = doublets.plan_doublet_samples([4], simulation_ratio=0.01, n_neighbors=2)
    assert few == doublets.DoubletSamplePlan(4, 1, 2, 2)
    (tie,) = doublets.plan_doublet_samples([4], simulation_ratio=0.5, n_neighbors=1)
    # 1 * (1 + 2 / 4) = 1.5 rounds to 2, 0.5 * 5 = 2.5 rounds to 2.
    assert tie == doublets.DoubletSamplePlan(4, 2, 1, 2)
    (half,) = doublets.plan_doublet_samples([5], simulation_ratio=0.5, n_neighbors=1)
    assert half.simulated == 2


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"cell_counts": [5, 1]}, ValueError, "A sample of 1 selected cells"),
        ({"cell_counts": [0]}, ValueError, "A sample of 0 selected cells"),
        ({"cell_counts": [1.5]}, TypeError, "cell count must be an integer"),
        ({"simulation_ratio": 0}, ValueError, "simulation_ratio must be"),
        ({"simulation_ratio": True}, TypeError, "simulation_ratio must be a real"),
        ({"n_neighbors": 0}, ValueError, "n_neighbors must be at least 1"),
    ],
)
def test_plan_doublet_samples_rejects_what_cannot_pair(arguments, error, message):
    options = {"cell_counts": [5], "simulation_ratio": 1.0, "n_neighbors": 3}
    with pytest.raises(error, match=message):
        doublets.plan_doublet_samples(**(options | arguments))


def test_doublet_likelihood_is_bayes_rule_on_the_smoothed_neighbor_fraction():
    counts = np.array([0, 1, 5, 10])

    actual = doublets.doublet_likelihood(
        counts, 10, expected_doublet_rate=0.1, simulation_ratio=2.0
    )

    # q = 1/12: q * 0.1 / 2 / (q * 0.1 / 2 + (1 - q) * 0.9) = 1 / 199.
    assert actual[0] == pytest.approx(1 / 199, rel=1e-12)
    # q = 11/12: (11/240) / (11/240 + 0.075) = 11 / 29.
    assert actual[3] == pytest.approx(11 / 29, rel=1e-12)
    np.testing.assert_allclose(actual, _likelihood(counts, 10, 0.1, 2.0), rtol=1e-12)
    assert np.all(np.diff(actual) > 0)
    # Twice the simulated doublets per cell halve the evidence of each one.
    np.testing.assert_allclose(
        doublets.doublet_likelihood(
            counts, 10, expected_doublet_rate=0.1, simulation_ratio=4.0
        ),
        _likelihood(counts, 10, 0.1, 4.0),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"simulated_neighbors": np.array([0.5])}, TypeError, "integer counts"),
        ({"simulated_neighbors": np.array([11])}, ValueError, "from 0 to"),
        ({"simulated_neighbors": np.array([-1])}, ValueError, "from 0 to"),
        ({"adjusted_neighbors": 0}, ValueError, "adjusted_neighbors must be"),
        ({"expected_doublet_rate": 0.0}, ValueError, "above 0 and below 1"),
        ({"expected_doublet_rate": 1.0}, ValueError, "above 0 and below 1"),
        ({"simulation_ratio": -1.0}, ValueError, "simulation_ratio must be"),
        ({"simulation_ratio": math.inf}, ValueError, "simulation_ratio must be"),
    ],
)
def test_doublet_likelihood_rejects_impossible_counts_and_rates(
    arguments, error, message
):
    options = {
        "simulated_neighbors": np.array([0, 3]),
        "adjusted_neighbors": 10,
        "expected_doublet_rate": 0.05,
        "simulation_ratio": 2.0,
    } | arguments
    counts = options.pop("simulated_neighbors")
    with pytest.raises(error, match=message):
        doublets.doublet_likelihood(counts, **options)
    # No counts are no evidence of anything.
    assert doublets.doublet_likelihood(
        np.array([], dtype=np.int64),
        1,
        expected_doublet_rate=0.05,
        simulation_ratio=1.0,
    ).shape == (0,)


def _bin_values(heights: list[int], bins: int) -> np.ndarray:
    """Values whose equal-width histogram over [0, bins] has these heights."""
    values = [
        index + 0.5 for index, height in enumerate(heights) for _ in range(height)
    ]
    # The two ends pin the histogram range to [0, bins].
    return np.asarray([0.0, *values, float(bins)])


def test_histogram_minimum_threshold_splits_two_modes_at_their_valley():
    # Bin centers are at 0.5, 1.5, ...; the ends add one value to the first
    # and last bins. Two clean modes need no smoothing.
    heights = [9, 6, 2, 1, 3, 8, 4, 0, 0]
    values = _bin_values(heights, len(heights))

    assert doublets.histogram_minimum_threshold(values, bins=len(heights)) == 3.5
    # The same values satisfy the minimum and Otsu's split differently.
    assert doublets.doublet_threshold(values)[1] == "minimum"


def test_histogram_minimum_threshold_counts_plateaus_and_smooths_noise():
    # A plateau is one maximum, at its middle bin; the first lowest bin between
    # the two maxima is the threshold.
    plateau = _bin_values([1, 7, 7, 7, 2, 2, 5, 1], 8)
    assert doublets.histogram_minimum_threshold(plateau, bins=8) == 4.5
    # Noise makes four maxima; smoothing leaves the two real ones.
    noisy = _bin_values([10, 12, 11, 13, 3, 1, 0, 2, 9, 11, 10, 12], 12)
    threshold = doublets.histogram_minimum_threshold(noisy, bins=12)
    assert threshold is not None and 4 < threshold < 8
    # Without smoothing steps the noise is never resolved.
    assert doublets.histogram_minimum_threshold(noisy, bins=12, max_steps=0) is None


def test_histogram_minimum_threshold_finds_no_valley_in_one_mode():
    unimodal = _bin_values([1, 3, 6, 9, 6, 3, 1], 7)
    assert doublets.histogram_minimum_threshold(unimodal, bins=7) is None
    assert doublets.histogram_minimum_threshold(np.full(10, 0.3)) is None


@pytest.mark.parametrize(
    ("values", "options", "error", "message"),
    [
        (np.array([]), {}, ValueError, "non-empty one-dimensional"),
        (np.zeros((2, 2)), {}, ValueError, "non-empty one-dimensional"),
        (np.array([0.1, np.nan]), {}, ValueError, "must be finite"),
        (np.array([0.1, 0.2]), {"bins": 2}, ValueError, "bins must be at least 3"),
        (np.array([0.1, 0.2]), {"max_steps": -1}, ValueError, "max_steps"),
    ],
)
def test_histogram_minimum_threshold_validates_its_input(
    values, options, error, message
):
    with pytest.raises(error, match=message):
        doublets.histogram_minimum_threshold(values, **options)


def test_otsu_threshold_maximizes_the_variance_between_two_classes():
    # Two groups at the ends of a four-bin histogram over [0, 4].
    values = np.array([0.0, 0.5, 0.5, 0.5, 3.5, 3.5, 4.0])
    assert doublets.otsu_threshold(values, bins=4) == 0.5
    # With one bin per class the split is between them.
    assert doublets.otsu_threshold(np.array([0.0, 1.0]), bins=2) == 0.25
    # Equal values have no split.
    assert doublets.otsu_threshold(np.full(5, 0.7)) == 0.7
    with pytest.raises(ValueError, match="bins must be at least 2"):
        doublets.otsu_threshold(values, bins=1)


def test_doublet_threshold_falls_back_to_otsu_without_two_modes():
    # One peak over the 256 bins of the automatic threshold: heights rise by
    # one bin by bin to a plateau of two bins and fall back.
    bins = doublets.THRESHOLD_BINS
    unimodal = _bin_values([min(i + 1, bins - i) for i in range(bins)], bins)
    assert doublets.histogram_minimum_threshold(unimodal) is None
    threshold, method = doublets.doublet_threshold(unimodal)
    assert method == "otsu"
    assert threshold == doublets.otsu_threshold(unimodal)
    # A symmetric peak splits at its middle.
    assert threshold == bins / 2 - 0.5


def test_doublet_index_settings_are_validated():
    with pytest.raises(ValueError, match="'l2' or 'cosine'"):
        doublets.DoubletIndexSettings("ip", 16, 50, 50, False)
    with pytest.raises(ValueError, match="m must be at least 2"):
        doublets.DoubletIndexSettings("l2", 1, 50, 50, False)
    with pytest.raises(TypeError, match="ef_construction must be an integer"):
        doublets.DoubletIndexSettings("l2", 16, 5.0, 50, False)
    with pytest.raises(ValueError, match="ef must be at least 1"):
        doublets.DoubletIndexSettings("l2", 16, 50, 0, False)
    with pytest.raises(TypeError, match="parallel must be a boolean"):
        doublets.DoubletIndexSettings("l2", 16, 50, 50, 1)


def test_estimated_doublet_rate_divides_calls_by_detectable_doublets():
    plan = doublets.DoubletSamplePlan(
        cells=200, simulated=400, neighbors=5, adjusted_neighbors=15
    )
    result = doublets.DoubletSampleResult(
        plan=plan,
        heterotypic_fraction=0.0,
        threshold=0.2,
        threshold_method="minimum",
        predicted=12,
        detectable_fraction=0.6,
    )
    assert result.estimated_doublet_rate == pytest.approx(0.1)
    undetectable = doublets.DoubletSampleResult(
        plan=plan,
        heterotypic_fraction=0.0,
        threshold=0.99,
        threshold_method="manual",
        predicted=0,
        detectable_fraction=0.0,
    )
    assert undetectable.estimated_doublet_rate is None


def _exact_counts(points: np.ndarray, n_observed: int, k: int) -> np.ndarray:
    values = points.astype(np.float64)
    distances = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return np.count_nonzero(nearest >= n_observed, axis=1)


@pytest.mark.parametrize("parallel", [False, True])
def test_count_simulated_neighbors_counts_the_exact_nearest_doublets(parallel):
    rng = np.random.default_rng(5)
    observed = rng.normal(size=(40, 4))
    # Doublets lie between two observed cells, as added profiles do.
    simulated = (
        observed[rng.integers(0, 40, 60)] + observed[rng.integers(0, 40, 60)]
    ) / 2
    simulated += rng.normal(scale=0.05, size=simulated.shape)
    points = np.vstack([observed, simulated]).astype(np.float32)
    settings = doublets.DoubletIndexSettings("l2", 16, 200, 200, parallel)

    counts = doublets.count_simulated_neighbors(
        points,
        40,
        12,
        settings,
        random_seed=3,
        workers=2,
        resources=ResourceBudget(256 * 1024**2, 2),
    )

    np.testing.assert_array_equal(counts, _exact_counts(points, 40, 12))
    if not parallel:
        # A one-thread index is the same on every run.
        np.testing.assert_array_equal(
            counts,
            doublets.count_simulated_neighbors(
                points,
                40,
                12,
                settings,
                random_seed=3,
                workers=1,
                resources=ResourceBudget(256 * 1024**2, 1),
            ),
        )


@pytest.mark.parametrize(
    ("points", "n_observed", "k", "message"),
    [
        (np.zeros((1, 2)), 1, 1, "at least two rows"),
        (np.zeros(4), 1, 1, "at least two rows"),
        (np.zeros((4, 2)), 4, 1, "must hold a simulated doublet"),
        (np.zeros((4, 2)), 0, 1, "n_observed must be at least 1"),
        (np.zeros((4, 2)), 2, 4, "below the number of points"),
    ],
)
def test_count_simulated_neighbors_validates_points(points, n_observed, k, message):
    with pytest.raises(ValueError, match=message):
        doublets.count_simulated_neighbors(
            points,
            n_observed,
            k,
            _INDEX,
            random_seed=0,
            workers=1,
            resources=ResourceBudget(256 * 1024**2, 1),
        )


def test_count_simulated_neighbors_admits_its_index_before_building_it(monkeypatch):
    def unbuilt(*_args, **_kwargs):
        raise AssertionError("the index was built before its admission")

    monkeypatch.setattr(doublets, "instantiate_knn_index", unbuilt)
    with pytest.raises(MemoryError, match="operation limit is 1000000 bytes"):
        doublets.count_simulated_neighbors(
            np.zeros((10, 2), dtype=np.float32),
            5,
            3,
            _INDEX,
            random_seed=0,
            workers=1,
            resources=ResourceBudget(1_000_000, 1),
        )


def test_sample_cluster_pool_without_a_cap_keeps_each_cluster_fraction():
    clusters = np.array([0, 0, 0, 0, 1, 1, 2])
    pool = doublets.sample_cluster_pool(clusters, 0.5, None, np.random.default_rng(1))
    counts = {int(c): int((clusters[pool] == c).sum()) for c in np.unique(clusters)}
    # ceil(0.5 * size) cells of every cluster: abundance is kept.
    assert counts == {0: 2, 1: 1, 2: 1}
    every = doublets.sample_cluster_pool(clusters, 1.0, None, np.random.default_rng(1))
    np.testing.assert_array_equal(every, np.arange(len(clusters)))


@pytest.mark.parametrize(
    ("codes", "labels", "totals", "threshold", "message"),
    [
        (np.array([0.0, 0.0]), np.zeros(2), None, None, "one integer per selected"),
        (np.zeros(3, dtype=int), np.zeros(2), None, None, "one integer per selected"),
        (np.zeros(2, dtype=int), np.zeros(3), None, None, "one label per selected"),
        (np.array([0, -1]), np.zeros(2), None, None, "must be non-negative"),
        (np.array([0, 2]), np.zeros(2), None, None, "up to the largest"),
        (np.zeros(2, dtype=int), np.zeros(2), None, math.nan, "threshold must be"),
        (np.zeros(2, dtype=int), np.zeros(2), np.ones(2), None, "needed exactly"),
    ],
)
def test_score_synthetic_doublets_validates_samples_before_reading(
    codes, labels, totals, threshold, message
):
    # The reference renormalizes its features, so it takes no cell totals.
    reference = SimpleNamespace(normalization_parameters={"renormalize_subset": True})
    with pytest.raises(ValueError, match=message):
        doublets.score_synthetic_doublets(
            None,
            reference,
            np.arange(2),
            labels,
            codes,
            np.arange(3),
            totals,
            index=_INDEX,
            cluster_sample_fraction=1.0,
            max_cells_per_cluster=None,
            simulation_ratio=2.0,
            heterotypic_fraction=0.0,
            n_neighbors=5,
            expected_doublet_rate=0.05,
            threshold=threshold,
            random_seed=0,
            resources=ResourceBudget(1_000_000, 1),
        )
    with pytest.raises(ValueError, match="expected_doublet_rate must be above 0"):
        doublets.score_synthetic_doublets(
            None,
            reference,
            np.arange(2),
            np.zeros(2),
            np.zeros(2, dtype=int),
            np.arange(3),
            None,
            index=_INDEX,
            cluster_sample_fraction=1.0,
            max_cells_per_cluster=None,
            simulation_ratio=2.0,
            heterotypic_fraction=0.0,
            n_neighbors=5,
            expected_doublet_rate=1.5,
            threshold=None,
            random_seed=0,
            resources=ResourceBudget(1_000_000, 1),
        )
    library = SimpleNamespace(normalization_parameters={"renormalize_subset": False})
    with pytest.raises(ValueError, match="one value per selected cell"):
        doublets.score_synthetic_doublets(
            None,
            library,
            np.arange(2),
            np.zeros(2),
            np.zeros(2, dtype=int),
            np.arange(3),
            np.ones(3),
            index=_INDEX,
            cluster_sample_fraction=1.0,
            max_cells_per_cluster=None,
            simulation_ratio=2.0,
            heterotypic_fraction=0.0,
            n_neighbors=5,
            expected_doublet_rate=0.05,
            threshold=None,
            random_seed=0,
            resources=ResourceBudget(1_000_000, 1),
        )


# Synthetic counts: cell types with their own marker genes, a gene-wise batch
# effect on every capture after the first, and doublets added within each
# capture from two cells of different types (heterotypic, truth 1) or of one
# type (homotypic, truth 2).
_ONE_CAPTURE = {
    "A": {
        "types": (0, 1, 2, 3, 4),
        "sizes": (300, 150, 100, 80, 60),
        "het": 60,
        "hom": 15,
    },
}
_TWO_CAPTURES = {
    "A": {"types": (0, 1, 2), "sizes": (300, 150, 90), "het": 45, "hom": 10},
    "B": {"types": (1, 2, 3, 4), "sizes": (120, 110, 70, 50), "het": 40, "hom": 10},
}


@dataclass(frozen=True)
class _SyntheticDoublets:
    path: Path
    truth: np.ndarray
    samples: np.ndarray
    types: np.ndarray


def _synthetic_counts(
    seed: int,
    design: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_genes, n_types = 120, 5
    profiles = np.tile(rng.gamma(0.6, 0.5, size=n_genes), (n_types, 1))
    for cell_type in range(n_types):
        markers = rng.choice(n_genes, 12, replace=False)
        profiles[cell_type, markers] += rng.uniform(2.0, 6.0, size=12)
    counts, truth, samples, types = [], [], [], []
    for position, (sample, capture) in enumerate(design.items()):
        batch = np.ones(n_genes) if position == 0 else rng.uniform(0.7, 1.4, n_genes)
        sample_types = np.repeat(capture["types"], capture["sizes"])
        depth = rng.lognormal(0.0, 0.25, size=len(sample_types))
        singlets = rng.poisson(profiles[sample_types] * batch * depth[:, None] * 3)
        n = len(sample_types)
        left = rng.integers(0, n, capture["het"] + capture["hom"])
        right = np.empty_like(left)
        for index, first in enumerate(left):
            same = sample_types == sample_types[first]
            partners = (
                ~same if index < capture["het"] else same & (np.arange(n) != first)
            )
            right[index] = rng.choice(np.flatnonzero(partners))
        counts += [singlets, singlets[left] + singlets[right]]
        truth += [
            np.zeros(n, dtype=int),
            np.repeat([1, 2], [capture["het"], capture["hom"]]),
        ]
        samples.append(np.full(n + len(left), sample))
        types += [sample_types, sample_types[left]]
    return (
        np.vstack(counts),
        np.concatenate(truth),
        np.concatenate(samples),
        np.concatenate(types),
    )


def _build_synthetic(
    path: Path, design: dict[str, dict[str, Any]]
) -> _SyntheticDoublets:
    counts, truth, samples, types = _synthetic_counts(3, design)
    write_count_store(str(path), {"RNA": counts}, "uint32")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=2
    )
    store.cells.insert("sample", samples)
    store.cells.insert("cell_type", types.astype(str))
    cells = store.snapshot_cell_selection("I")
    normalized = store.features.normalize(cells, store.features.universe())
    pca = store.reduction.pca(normalized, dims=10, local_cache=False)
    neighbors = store.graph.neighbors(store.graph.ann_index(pca), coordinates=pca, k=10)
    store.graph.connectivity(neighbors)
    store.snapshot_cluster_labels("cell_type", cell_selection=cells)
    return _SyntheticDoublets(path, truth, samples, types)


@pytest.fixture(scope="module")
def one_capture(tmp_path_factory) -> _SyntheticDoublets:
    """One capture, so every cell is one sample."""
    path = tmp_path_factory.mktemp("one_capture") / "data.zarr"
    return _build_synthetic(path, _ONE_CAPTURE)


@pytest.fixture(scope="module")
def two_captures(tmp_path_factory) -> _SyntheticDoublets:
    """Two captures of different cell types with a batch effect between them."""
    path = tmp_path_factory.mktemp("two_captures") / "data.zarr"
    return _build_synthetic(path, _TWO_CAPTURES)


def _open_synthetic(synthetic: _SyntheticDoublets, tmp_path: Path) -> SimpleNamespace:
    target = tmp_path / "data.zarr"
    shutil.copytree(synthetic.path, target)
    store = DataStore(
        str(target), default_assay="RNA", min_features_per_cell=0, nthreads=2
    )
    (graph,) = store.artifacts.list(kind="connectivity_map", from_assay="RNA")
    (clusters,) = store.artifacts.list(kind="cluster_labels", scope="datastore")
    return SimpleNamespace(store=store, graph=graph, clusters=clusters)


def _scored(
    synthetic: _SyntheticDoublets, directory: Path, *requests: dict[str, Any]
) -> _SyntheticDoublets:
    """Copy a synthetic store with the doublets of each request scored.

    Tests that score the same doublets in their copies reuse these results
    instead of computing them again.
    """
    opened = _open_synthetic(synthetic, directory)
    for options in requests:
        opened.store.qc.doublets(opened.clusters, opened.graph, **options)
    return replace(synthetic, path=directory / "data.zarr")


@pytest.fixture(scope="module")
def one_capture_scored(one_capture, tmp_path_factory) -> _SyntheticDoublets:
    """One capture with its doublets scored with the default options."""
    return _scored(one_capture, tmp_path_factory.mktemp("one_capture_scored"), {})


@pytest.fixture(scope="module")
def two_captures_scored(two_captures, tmp_path_factory) -> _SyntheticDoublets:
    """Two captures with doublets scored per sample and pooled, by default."""
    return _scored(
        two_captures,
        tmp_path_factory.mktemp("two_captures_scored"),
        {"sample_column": "sample"},
        {},
    )


def _payload(store: DataStore, ref: ArtifactRef) -> Any:
    return artifact_group(store.zw, ref)


def _graph_pca(store: DataStore, graph: ArtifactRef) -> ArtifactRef:
    neighbors = store.artifacts.inspect(graph).input_ref("neighbors")
    return store.artifacts.inspect(neighbors).input_ref("coordinates")


def _assert_recovered(payload: Any, truth: np.ndarray, samples: np.ndarray) -> None:
    """Check the scores, calls, and thresholds against the injected doublets."""
    scores = payload["values"][:]
    calls = payload["predicted_doublets"][:]
    heterotypic, singlets = truth == 1, truth == 0
    # Heterotypic doublets rank above singlets almost perfectly.
    assert roc_auc_score(heterotypic[truth != 2], scores[truth != 2]) >= 0.98
    # The automatic threshold calls most of them and few singlets.
    assert calls[heterotypic].mean() >= 0.8
    assert calls[singlets].mean() <= 0.02
    # Doublets of one type look like singlets and are mostly missed.
    assert calls[truth == 2].mean() <= 0.5
    cutoffs = np.empty(len(scores))
    for sample in payload.attrs["samples"]:
        members = (
            samples == sample["labels"]["sample"]
            if sample["labels"]
            else np.ones(len(scores), dtype=bool)
        )
        cutoffs[members] = sample["threshold"]
        assert sample["thresholdMethod"] == "minimum"
        assert sample["predictedDoublets"] == int(calls[members].sum())
        # The attributes keep the floats exactly, so the estimate is exactly
        # Scrublet's ratio of the recorded calls and detectable fraction.
        assert sample["estimatedDoubletRate"] == (
            sample["predictedDoublets"]
            / sample["cells"]
            / sample["detectableDoubletFraction"]
        )
        # Scrublet's estimate is near the injected doublet rate.
        injected = np.mean(truth[members] != 0)
        assert abs(sample["estimatedDoubletRate"] - injected) <= 0.04
    # The calls are the scores above each sample's threshold.
    np.testing.assert_array_equal(calls, scores > cutoffs)


@pytest.mark.slow
def test_doublets_recover_injected_heterotypic_doublets_of_one_capture(
    one_capture_scored, tmp_path
):
    opened = _open_synthetic(one_capture_scored, tmp_path)

    # The scores that the template computed with these options are reused.
    ref = opened.store.qc.doublets(opened.clusters, opened.graph)

    payload = _payload(opened.store, ref)
    _assert_recovered(payload, one_capture_scored.truth, one_capture_scored.samples)
    assert payload.attrs["samples"][0]["labels"] == {}
    raw = payload["raw_scores"][:]
    assert np.all((raw >= 0) & (raw <= 1))
    # The score rises with the fraction of simulated doublets around a cell.
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(payload["values"][:][order]) >= 0)


@pytest.mark.slow
def test_per_sample_doublets_recover_injected_doublets_of_two_captures(
    two_captures_scored, tmp_path
):
    opened = _open_synthetic(two_captures_scored, tmp_path)
    store = opened.store

    # The template's scores, per sample and pooled, are reused.
    per_sample = _payload(
        store, store.qc.doublets(opened.clusters, opened.graph, sample_column="sample")
    )

    _assert_recovered(
        per_sample, two_captures_scored.truth, two_captures_scored.samples
    )
    # Pooled, doublets of two captures form where no cell can be, so they
    # score above the doublets of one capture and blur the threshold.
    pooled = _payload(store, store.qc.doublets(opened.clusters, opened.graph))
    parents = pooled["simulated_parents"][:]
    samples = two_captures_scored.samples
    across = samples[parents[:, 0]] != samples[parents[:, 1]]
    simulated = pooled["simulated_scores"][:]
    assert np.median(simulated[across]) > np.median(simulated[~across])
    heterotypic = two_captures_scored.truth == 1
    assert (
        per_sample["predicted_doublets"][:][heterotypic].mean()
        >= pooled["predicted_doublets"][:][heterotypic].mean()
    )


@pytest.mark.slow
def test_per_sample_doublets_pair_and_score_cells_of_one_sample_only(
    two_captures_scored, tmp_path, monkeypatch
):
    opened = _open_synthetic(two_captures_scored, tmp_path)
    store = opened.store
    captured: list[tuple[np.ndarray, int]] = []
    count_neighbors = doublets.count_simulated_neighbors

    def capture(points, n_observed, *args, **kwargs):
        captured.append((np.array(points), n_observed))
        return count_neighbors(points, n_observed, *args, **kwargs)

    monkeypatch.setattr(doublets, "count_simulated_neighbors", capture)
    # The template holds these scores; scoring them again shows each sample's
    # neighbor index.
    per_sample = store.qc.doublets(
        opened.clusters, opened.graph, sample_column="sample", invalidate_cache=True
    )

    payload = _payload(store, per_sample)
    parents = payload["simulated_parents"][:]
    samples = two_captures_scored.samples
    # Doublets never pair parents of two samples.
    np.testing.assert_array_equal(samples[parents[:, 0]], samples[parents[:, 1]])
    summaries = payload.attrs["samples"]
    assert [summary["labels"] for summary in summaries] == [
        {"sample": "A"},
        {"sample": "B"},
    ]
    sizes = [int(np.count_nonzero(samples == label)) for label in ("A", "B")]
    assert [summary["cells"] for summary in summaries] == sizes
    assert [summary["simulatedDoublets"] for summary in summaries] == [
        2 * size for size in sizes
    ]
    # The simulated doublets of A come first, then those of B.
    np.testing.assert_array_equal(
        samples[parents[:, 0]], np.repeat(["A", "B"], [2 * size for size in sizes])
    )
    # Each sample is scored against its own cells and doublets only: its
    # neighbor index holds the PCA rows of its cells and its doublets.
    coordinates = artifact_group(store.zw, _graph_pca(store, opened.graph))["data"][:]
    assert [n_observed for _, n_observed in captured] == sizes
    for (points, n_observed), label in zip(captured, ("A", "B"), strict=True):
        np.testing.assert_array_equal(
            points[:n_observed], coordinates[samples == label]
        )
        assert len(points) == 3 * n_observed
    status = store.artifacts.inspect(per_sample)
    assert status.parameters["sample_columns"] == ["sample"]
    assert isinstance(status.inputs["sample_assignments_fingerprint"], str)

    # Without a sample column, doublets pair cells of both samples.
    pooled = store.qc.doublets(opened.clusters, opened.graph)
    pooled_parents = _payload(store, pooled)["simulated_parents"][:]
    assert np.any(samples[pooled_parents[:, 0]] != samples[pooled_parents[:, 1]])
    pooled_status = store.artifacts.inspect(pooled)
    assert pooled_status.parameters["sample_columns"] == []
    assert pooled_status.inputs["sample_assignments_fingerprint"] is None


@pytest.mark.slow
def test_doublet_scores_are_stabler_across_seeds_than_with_a_capped_pool(
    one_capture_scored, tmp_path
):
    # The template holds the scores of the default seed 4444 and pool.
    opened = _open_synthetic(one_capture_scored, tmp_path)
    store = opened.store

    def scores(seed: int, **options: Any) -> np.ndarray:
        ref = store.qc.doublets(
            opened.clusters, opened.graph, random_seed=seed, **options
        )
        return _payload(store, ref)["values"][:]

    def agreement(**options: Any) -> list[float]:
        first = scores(4444, **options)
        return [float(spearmanr(first, scores(seed, **options))[0]) for seed in (1, 2)]

    every_cell = agreement()
    # The former default pool: 5% of each cluster, at most 100 cells, parent
    # every doublet, so each parent was reused dozens of times.
    capped = agreement(cluster_sample_fraction=0.05, max_cells_per_cluster=100)
    assert min(every_cell) >= 0.9
    assert min(every_cell) > max(capped) + 0.05


@pytest.mark.slow
def test_doublet_artifact_of_revision_1_is_recomputed_not_reused(
    one_capture, tmp_path, monkeypatch
):
    opened = _open_synthetic(one_capture, tmp_path)
    store = opened.store
    monkeypatch.setattr(
        operation_revisions, "OPERATION_REVISIONS", MappingProxyType({})
    )
    legacy = store.qc.doublets(opened.clusters, opened.graph)
    assert store.artifacts.inspect(legacy).revision == 1
    monkeypatch.undo()

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        current = store.qc.doublets(opened.clusters, opened.graph)
    finally:
        logger.remove(sink)

    assert current != legacy
    assert store.artifacts.inspect(current).revision == 2
    legacy_status = store.artifacts.inspect(legacy)
    assert not legacy_status.is_current
    assert [entry.revision for entry in legacy_status.superseded_by] == [2]
    (line,) = [message for message in messages if "Recomputing" in message]
    assert line.startswith(
        f"Recomputing run_doublet_detection: artifact {legacy.artifact_id[:12]} "
        "is revision 1, current 2: Doublets are simulated from every selected cell"
    )
    assert store.qc.doublets(opened.clusters, opened.graph) == current


@pytest.mark.slow
def test_save_k_is_capped_by_the_largest_sample_before_planning(
    two_captures, tmp_path, monkeypatch
):
    opened = _open_synthetic(two_captures, tmp_path)
    store = opened.store
    sizes = [int(np.count_nonzero(two_captures.samples == label)) for label in "AB"]
    largest = max(sizes)
    ref = store.qc.doublets(
        opened.clusters,
        opened.graph,
        sample_column="sample",
        simulation_ratio=0.1,
        save_k=largest + 10,
    )
    status = store.artifacts.inspect(ref)
    # The recorded neighbors are those used: the largest sample's other cells.
    assert status.parameters["save_k"] == largest - 1
    assert [
        sample["neighbors"] for sample in _payload(store, ref).attrs["samples"]
    ] == [size - 1 for size in sizes]

    def recomputed(*_args, **_kwargs):
        raise AssertionError("a request capped to the same neighbors was recomputed")

    monkeypatch.setattr(doublets, "score_synthetic_doublets", recomputed)
    # Any larger request is the same artifact, not a duplicate.
    assert (
        store.qc.doublets(
            opened.clusters,
            opened.graph,
            sample_column="sample",
            simulation_ratio=0.1,
            save_k=10 * largest,
        )
        == ref
    )


@pytest.mark.slow
def test_predicted_doublets_select_singlets_and_options_keep_calls(
    two_captures_scored, tmp_path
):
    opened = _open_synthetic(two_captures_scored, tmp_path)
    store = opened.store
    # The template's per-sample scores are reused.
    ref = store.qc.doublets(opened.clusters, opened.graph, sample_column="sample")
    payload = _payload(store, ref)
    scores = payload["values"][:]
    calls = payload["predicted_doublets"][:]

    def selected(selection: ArtifactRef) -> np.ndarray:
        return read_stored_selection_mask(
            store.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

    singlets = store.select_cells(ref, value="predicted_doublets", include=[False])
    np.testing.assert_array_equal(selected(singlets), ~calls)
    assert store.artifacts.inspect(singlets).parameters["value"] == "predicted_doublets"
    loaded = store.artifacts.load_values(ref, value="predicted_doublets")
    assert loaded.categorical
    np.testing.assert_array_equal(loaded.values, calls)
    # The canonical array is read and recorded as before.
    low = store.select_cells(ref, high=0.5)
    assert store.select_cells(ref, high=0.5, value="values") == low
    assert "value" not in store.artifacts.inspect(low).parameters
    raw = store.select_cells(ref, value="raw_scores", high=0.5)
    np.testing.assert_array_equal(selected(raw), payload["raw_scores"][:] < 0.5)
    with pytest.raises(ValueError, match="no per-cell array 'calls'"):
        store.select_cells(ref, value="calls", include=[False])
    with pytest.raises(TypeError, match="value must be a string or None"):
        store.select_cells(ref, value=1, include=[False])

    # A given threshold applies to every sample.
    manual = store.qc.doublets(
        opened.clusters, opened.graph, sample_column="sample", threshold=0.3
    )
    manual_payload = _payload(store, manual)
    np.testing.assert_array_equal(manual_payload["values"][:], scores)
    np.testing.assert_array_equal(manual_payload["predicted_doublets"][:], scores > 0.3)
    assert {
        (sample["threshold"], sample["thresholdMethod"])
        for sample in manual_payload.attrs["samples"]
    } == {(0.3, "manual")}
    # Scaling changes the scores alone.
    scaled = store.qc.doublets(
        opened.clusters, opened.graph, sample_column="sample", normalize_scores=True
    )
    scaled_payload = _payload(store, scaled)
    np.testing.assert_allclose(
        scaled_payload["values"][:],
        (scores - scores.min()) / (scores.max() - scores.min()),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(scaled_payload["predicted_doublets"][:], calls)
    np.testing.assert_array_equal(
        scaled_payload["raw_scores"][:], payload["raw_scores"][:]
    )


@pytest.mark.slow
def test_a_harmony_graph_scores_each_sample_on_the_pca_it_corrected(
    two_captures_scored, tmp_path
):
    # The template holds the per-sample scores of the uncorrected graph.
    opened = _open_synthetic(two_captures_scored, tmp_path)
    store = opened.store
    correction = store.reduction.harmony(_graph_pca(store, opened.graph), ["sample"])
    corrected_neighbors = store.graph.neighbors(
        store.graph.ann_index(correction), coordinates=correction, k=10
    )
    corrected = store.graph.connectivity(corrected_neighbors)

    with pytest.raises(ValueError, match="needs sample_column"):
        store.qc.doublets(opened.clusters, corrected)
    ref = store.qc.doublets(opened.clusters, corrected, sample_column="sample")
    uncorrected = store.qc.doublets(
        opened.clusters, opened.graph, sample_column="sample"
    )

    assert ref != uncorrected
    for name in ("values", "raw_scores", "predicted_doublets", "simulated_parents"):
        np.testing.assert_array_equal(
            _payload(store, ref)[name][:], _payload(store, uncorrected)[name][:]
        )


@pytest.mark.slow
def test_doublet_samples_need_labels_two_cells_and_pair_one_cluster_freely(
    two_captures, tmp_path
):
    opened = _open_synthetic(two_captures, tmp_path)
    store = opened.store
    samples = two_captures.samples
    missing = np.zeros(store.cells.N, dtype=bool)
    missing[[3, 7]] = True
    insert_nullable_cell_column(store, "partial_sample", samples, missing)
    with pytest.raises(
        ValueError, match=r"'partial_sample' has no label for 2 selected"
    ):
        store.qc.doublets(opened.clusters, opened.graph, sample_column="partial_sample")

    lonely = samples.copy()
    lonely[0] = "C"
    store.cells.insert("lonely_sample", lonely)
    with pytest.raises(
        ValueError, match=r"sample \(lonely_sample='C'\) has one selected cell"
    ):
        store.qc.doublets(opened.clusters, opened.graph, sample_column="lonely_sample")
    assert store.artifacts.list(kind="doublet_score", from_assay="RNA") == []

    # Sample A is one cluster: its doublets pair any two of its cells.
    one_type = np.where(samples == "A", "a", two_captures.types.astype(str))
    store.cells.insert("one_type_in_a", one_type)
    clusters = store.snapshot_cluster_labels(
        "one_type_in_a", cell_selection=store.snapshot_cell_selection("I")
    )
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        ref = store.qc.doublets(
            clusters, opened.graph, sample_column="sample", heterotypic_fraction=0.8
        )
    finally:
        logger.remove(sink)
    summaries = _payload(store, ref).attrs["samples"]
    assert [summary["heterotypicFraction"] for summary in summaries] == [0.0, 0.8]
    (warning,) = [message for message in messages if "same cluster label" in message]
    assert warning.startswith("Every parent cell of sample (sample='A')")
    parents = _payload(store, ref)["simulated_parents"][:]
    in_b = samples[parents[:, 0]] == "B"
    labels = one_type[parents]
    # Most of B's doublets are forced to pair two clusters.
    assert np.mean(labels[in_b, 0] != labels[in_b, 1]) >= 0.8


def test_doublet_samples_group_cells_by_every_column_in_label_order():
    from cytearc.datastore._operations.quality_control import (
        _doublet_sample_codes,
        _doublet_sample_columns,
    )

    assert _doublet_sample_columns(None) == ()
    assert _doublet_sample_columns("sample") == ("sample",)
    assert _doublet_sample_columns(["lane", "sample"]) == ("lane", "sample")
    for invalid in (3, [], ["lane", ""], [1], b"lane"):
        with pytest.raises(TypeError, match="a sequence of them, or None"):
            _doublet_sample_columns(invalid)
    with pytest.raises(ValueError, match="must not repeat a column"):
        _doublet_sample_columns(["lane", "lane"])

    codes, samples, fingerprint = _doublet_sample_codes(
        ("lane", "sample"),
        [np.array([2, 1, 1, 2]), np.array(["x", "y", "x", "x"], dtype=object)],
        4,
    )
    # Samples are numbered in the order of their labels, column by column.
    assert samples == [
        {"lane": "1", "sample": "x"},
        {"lane": "1", "sample": "y"},
        {"lane": "2", "sample": "x"},
    ]
    np.testing.assert_array_equal(codes, [2, 1, 0, 2])
    assert codes.dtype == np.int64
    # The fingerprint follows the samples of the cells, not their labels' type.
    same, _, same_fingerprint = _doublet_sample_codes(
        ("lane", "sample"),
        [np.array(["2", "1", "1", "2"]), np.array(["x", "y", "x", "x"])],
        4,
    )
    np.testing.assert_array_equal(same, codes)
    assert same_fingerprint == fingerprint
    _, _, swapped = _doublet_sample_codes(
        ("lane", "sample"),
        [np.array([1, 2, 1, 2]), np.array(["x", "y", "x", "x"], dtype=object)],
        4,
    )
    assert swapped != fingerprint
    pooled, pooled_samples, none = _doublet_sample_codes((), (), 3)
    np.testing.assert_array_equal(pooled, [0, 0, 0])
    assert (pooled_samples, none) == ([{}], None)
    with pytest.raises(ValueError, match="must hold one label per selected cell"):
        _doublet_sample_codes(("lane",), [np.array(["1"])], 2)
    with pytest.raises(ValueError, match="'lane' has no label for 2 selected cells"):
        _doublet_sample_codes(("lane",), [np.array(["1", None, " "], dtype=object)], 3)


@pytest.mark.slow
def test_internal_doublet_samples_need_one_label_array_per_column(
    one_capture, tmp_path
):
    from cytearc.graph.feature_projection import graph_cell_selection

    opened = _open_synthetic(one_capture, tmp_path)
    store = opened.store
    options = dict(
        source_assay=store.RNA,
        cell_selection=graph_cell_selection(store.zw, opened.graph),
        clusters=opened.clusters,
        cluster_values=one_capture.types,
        connectivity=opened.graph,
    )
    with pytest.raises(ValueError, match="one array of sample labels per sample"):
        store._run_doublet_detection_artifact(
            **options, sample_column="sample", sample_labels=None
        )
    with pytest.raises(ValueError, match="one array of sample labels per sample"):
        store._run_doublet_detection_artifact(
            **options, sample_column=None, sample_labels=[one_capture.samples]
        )
    assert store.artifacts.list(kind="doublet_score", from_assay="RNA") == []


def test_pipeline_resolves_doublet_samples_from_settings_and_harmony(
    datastore_zarr_root,
):
    store = DataStore(datastore_zarr_root, default_assay="RNA", zarr_mode="r")
    options = {
        "assay": None,
        "label": None,
        "cell_key": "I",
        "filtering": False,
        "harmony_batch_columns": None,
        "hvg_count": 1000,
        "pca_dims": 21,
        "neighbors_k": 11,
        "umap": True,
        "leiden": True,
        "cell_cycle": True,
        "paris": True,
        "doublets": True,
        "markers": True,
        "snapshot_columns": (),
    }

    def resolve(**overrides: Any):
        return resolve_pipeline_recipe(store, **(options | overrides))

    pooled = resolve()
    assert pooled.doublet_sample_columns == ()
    assert pooled.to_config()["doubletSampleColumns"] == []
    named = resolve(params={"doublets": {"sample_column": "ids"}})
    assert named.doublet_sample_columns == ("ids",)
    assert named.to_config()["doubletSampleColumns"] == ["ids"]
    # The recipe resolves the samples; the run records the setting as given.
    assert named.to_config()["params"]["doublets"] == {"sample_column": "ids"}
    # Doublet samples are frozen apart, so the cell snapshot, and the
    # identities of the stages that read it, do not change.
    assert named.cell_snapshot_columns == pooled.cell_snapshot_columns
    assert (
        resolve(params={"doublets": {"sample_column": None}}).doublet_sample_columns
        == ()
    )
    # With Harmony, doublets are scored within each batch by default.
    harmony = resolve(harmony_batch_columns=["ids", "names"])
    assert harmony.doublet_sample_columns == ("ids", "names")
    chosen = resolve(
        harmony_batch_columns=["ids"],
        params={"doublets": {"sample_column": "names"}},
    )
    assert chosen.doublet_sample_columns == ("names",)
    assert (
        resolve(harmony_batch_columns=["ids"], doublets=False).doublet_sample_columns
        == ()
    )
    with pytest.raises(ValueError, match="cannot be None"):
        resolve(
            harmony_batch_columns=["ids"],
            params={"doublets": {"sample_column": None}},
        )
    with pytest.raises(TypeError, match="non-empty string or None"):
        resolve(params={"doublets": {"sample_column": ""}})
    with pytest.raises(KeyError, match="Doublet sample column 'missing'"):
        resolve(params={"doublets": {"sample_column": "missing"}})
    with pytest.raises(ValueError, match="collide with reserved run fields"):
        resolve(snapshot_columns=("predicted_doublet",))
