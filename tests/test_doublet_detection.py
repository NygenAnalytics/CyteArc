"""Doublet scores, thresholds, and calls from simulated doublets per sample.

The unit tests check the likelihood, the thresholds, the per-sample plan, and
the neighbor counts against hand computations and exact neighbors. The store
tests inject heterotypic and homotypic doublets into synthetic counts of two
samples and check what ``qc.doublets`` recovers, that each sample simulates
and scores doublets alone, and that the scores are stable across seeds.
"""

import inspect
import math
import shutil
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import cytearc.storage.operation_revisions as operation_revisions
from cytearc import DataStore, DiagnosticWarning
from cytearc.assay import norm_clr, norm_lib_size, norm_pearson_residuals, norm_tf_idf
from cytearc.datastore._pipeline.recipes import rna as rna_recipe
from cytearc.datastore.pipeline_accessor import PipelineAccessor
from cytearc.mapping.models import ScaledPCAProjectionModel
from cytearc.matrix import ChunkedArray
from cytearc.quality_control import doublets
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.selections import read_stored_selection_mask
from cytearc.utils.logging import logger
from tests import full_path
from tests.storage_helpers import (
    by_cell_id,
    insert_nullable_cell_column,
    write_count_store,
)
from tests.test_reduction_graph_diagnostics import _warned

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


# The qc.doublets defaults of the settings that doublet_settings checks.
_DOUBLET_SETTINGS: dict[str, Any] = {
    "cluster_sample_fraction": 1.0,
    "max_cells_per_cluster": None,
    "simulation_ratio": 2.0,
    "heterotypic_fraction": 0.0,
    "save_k": 30,
    "expected_doublet_rate": 0.05,
    "threshold": None,
    "smoothing_t": 0,
    "normalize_scores": False,
    "random_seed": 4444,
}


def test_doublet_settings_take_their_bounds_as_python_values():
    settings = doublets.doublet_settings(
        cluster_sample_fraction=np.float32(1.0),
        max_cells_per_cluster=np.int64(1),
        simulation_ratio=1e-9,
        heterotypic_fraction=1,
        save_k=np.int32(1),
        expected_doublet_rate=0.999,
        threshold=0,
        smoothing_t=0,
        normalize_scores=np.True_,
        random_seed=0,
    )

    assert settings == doublets.DoubletSettings(
        1.0, 1, 1e-9, 1.0, 1, 0.999, 0.0, 0, True, 0
    )
    numbers = {
        float: (
            "cluster_sample_fraction",
            "simulation_ratio",
            "heterotypic_fraction",
            "expected_doublet_rate",
            "threshold",
        ),
        int: ("max_cells_per_cluster", "save_k", "smoothing_t", "random_seed"),
        bool: ("normalize_scores",),
    }
    for kind, names in numbers.items():
        for name in names:
            assert type(getattr(settings, name)) is kind, name
    assert doublets.doublet_settings(**_DOUBLET_SETTINGS) == doublets.DoubletSettings(
        **_DOUBLET_SETTINGS
    )


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("cluster_sample_fraction", 0.0, ValueError),
        ("cluster_sample_fraction", 1.0000001, ValueError),
        ("max_cells_per_cluster", 0, ValueError),
        ("simulation_ratio", 0.0, ValueError),
        ("simulation_ratio", np.inf, ValueError),
        ("simulation_ratio", np.True_, TypeError),
        ("heterotypic_fraction", -1e-9, ValueError),
        ("heterotypic_fraction", 1.0000001, ValueError),
        ("save_k", True, TypeError),
        ("expected_doublet_rate", 0.0, ValueError),
        ("expected_doublet_rate", 1.0, ValueError),
        ("expected_doublet_rate", np.nan, ValueError),
        ("threshold", -1e-9, ValueError),
        ("threshold", 1.0000001, ValueError),
        ("smoothing_t", -1, ValueError),
        ("normalize_scores", 0, TypeError),
        ("random_seed", -1, ValueError),
        ("random_seed", 1.0, TypeError),
    ],
)
def test_doublet_settings_refuse_values_past_their_bounds(name, value, error):
    with pytest.raises(error, match=f"^{name} must be"):
        doublets.doublet_settings(**{**_DOUBLET_SETTINGS, name: value})


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
    # The index of ten cells holds a few kilobytes, more than this budget.
    with pytest.raises(MemoryError, match="operation limit is 1000 bytes"):
        doublets.count_simulated_neighbors(
            np.zeros((10, 2), dtype=np.float32),
            5,
            3,
            _INDEX,
            random_seed=0,
            workers=1,
            resources=ResourceBudget(1_000, 1),
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
    # The projection renormalizes its features, so it takes no cell totals.
    # It has nothing else to read, so a check that read it would fail.
    renormalizing = SimpleNamespace(renormalize_subset=True)
    with pytest.raises(ValueError, match=message):
        doublets.score_synthetic_doublets(
            None,
            renormalizing,
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
            renormalizing,
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
    library = SimpleNamespace(renormalize_subset=False)
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
    assay: str = "RNA"


def _synthetic_counts(
    seed: int,
    design: dict[str, dict[str, Any]],
    *,
    peaks: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return counts of the design's singlets and doublets, and their truth.

    Genes have a shared expression profile and twelve markers of each cell
    type. With ``peaks``, each cell type opens 50 of 400 peaks, which it
    covers with a mean of one fragment against 0.03 elsewhere.
    """
    rng = np.random.default_rng(seed)
    n_types = 5
    if peaks:
        n_genes, scale = 400, 1.0
        profiles = np.full((n_types, n_genes), 0.03)
        for cell_type in range(n_types):
            profiles[cell_type, rng.choice(n_genes, 50, replace=False)] = 1.0
    else:
        n_genes, scale = 120, 3.0
        profiles = np.tile(rng.gamma(0.6, 0.5, size=n_genes), (n_types, 1))
        for cell_type in range(n_types):
            markers = rng.choice(n_genes, 12, replace=False)
            profiles[cell_type, markers] += rng.uniform(2.0, 6.0, size=12)
    counts, truth, samples, types = [], [], [], []
    for position, (sample, capture) in enumerate(design.items()):
        batch = np.ones(n_genes) if position == 0 else rng.uniform(0.7, 1.4, n_genes)
        sample_types = np.repeat(capture["types"], capture["sizes"])
        depth = rng.lognormal(0.0, 0.25, size=len(sample_types))
        singlets = rng.poisson(profiles[sample_types] * batch * depth[:, None] * scale)
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
    path: Path, design: dict[str, dict[str, Any]], *, assay: str = "RNA"
) -> _SyntheticDoublets:
    """Write the design's counts with a graph and the cell types as clusters.

    An RNA graph is built on a PCA of library-size values, and an ATAC graph
    on an LSI of TF-IDF values.
    """
    counts, truth, samples, types = _synthetic_counts(3, design, peaks=assay == "ATAC")
    write_count_store(str(path), {assay: counts}, "uint32")
    store = DataStore(
        str(path), default_assay=assay, min_features_per_cell=0, nthreads=2
    )
    store.cells.insert("sample", by_cell_id(store.cells, samples))
    store.cells.insert("cell_type", by_cell_id(store.cells, types.astype(str)))
    cells = store.snapshot_cell_selection("I")
    normalized = store.features.normalize(cells, store.features.universe())
    reduction = (
        store.reduction.lsi(normalized, dims=8, local_cache=False)
        if assay == "ATAC"
        else store.reduction.pca(normalized, dims=10, local_cache=False)
    )
    neighbors = store.graph.neighbors(
        store.graph.ann_index(reduction), coordinates=reduction, k=10
    )
    store.graph.connectivity(neighbors)
    store.snapshot_cluster_labels("cell_type", cell_selection=cells)
    return _SyntheticDoublets(path, truth, samples, types, assay)


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


# Two captures of a hundred cells each, for tests that need a graph and its
# clusters rather than the recovery of doublets.
_SMALL_CAPTURES = {
    "A": {"types": (0, 1), "sizes": (60, 40), "het": 6, "hom": 2},
    "B": {"types": (2, 3), "sizes": (60, 40), "het": 6, "hom": 2},
}


@pytest.fixture(scope="module")
def small_captures(tmp_path_factory) -> _SyntheticDoublets:
    """Two small captures, each of two cell types."""
    path = tmp_path_factory.mktemp("small_captures") / "data.zarr"
    return _build_synthetic(path, _SMALL_CAPTURES)


def _open_synthetic(synthetic: _SyntheticDoublets, tmp_path: Path) -> SimpleNamespace:
    target = tmp_path / "data.zarr"
    shutil.copytree(synthetic.path, target)
    store = DataStore(
        str(target), default_assay=synthetic.assay, min_features_per_cell=0, nthreads=2
    )
    (graph,) = store.artifacts.list(kind="connectivity_map", from_assay=synthetic.assay)
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


_LIBRARY_SIZE = {"module": "cytearc.assay", "qualname": "norm_lib_size"}


@pytest.mark.parametrize(
    ("recorded", "named"),
    [
        (
            {
                "external_hook": True,
                "module": "cytearc.assay",
                "qualname": "norm_lib_size_log",
            },
            "cytearc.assay.norm_lib_size_log",
        ),
        (
            {
                "external_hook": True,
                "module": "lab.norms",
                "qualname": "half",
                "code": "0f",
            },
            "lab.norms.half",
        ),
        (
            {"identity": "cytearc.assay.pearson_residuals:theta=50.0:clip=sqrt-n"},
            "cytearc.assay.pearson_residuals:theta=50.0:clip=sqrt-n",
        ),
        # Records that name no normalizer are quoted as they are.
        ({"external_hook": True, "module": 3}, "{'module': 3}"),
        (None, "None"),
    ],
)
def test_the_doublet_graph_check_names_the_recorded_normalizer(
    monkeypatch, recorded, named
):
    from cytearc.datastore._operations import quality_control

    statuses = {
        "normalized": SimpleNamespace(
            parameters=None if recorded is None else {"normalization_method": recorded}
        ),
        "reduction": SimpleNamespace(
            operation="run_pca", parameters={"feat_scaling": True}
        ),
    }
    monkeypatch.setattr(
        quality_control, "inspect_artifact", lambda _root, ref: statuses[ref]
    )
    lineage = SimpleNamespace(normalized="normalized", reduction="reduction")

    with pytest.raises(ValueError) as caught:
        quality_control._require_doublet_graph(None, lineage)

    assert str(caught.value).startswith(
        f"The graph was built on values normalized with {named}, but doublet "
        "scoring normalizes its simulated doublets by library size"
    )
    # norm_lib_size passes, recorded with or without its external_hook key.
    for library_size in (_LIBRARY_SIZE, {"external_hook": True, **_LIBRARY_SIZE}):
        statuses["normalized"].parameters = {"normalization_method": library_size}
        quality_control._require_doublet_graph(None, lineage)


def test_the_doublet_graph_check_needs_a_pca_of_normalized_values():
    from cytearc.datastore._operations import quality_control

    message = (
        "Doublet scoring projects its simulated doublets onto the PCA of the "
        "graph's normalized values, but the graph's lineage names no PCA or no "
        "normalized values; build the graph on reduction.pca"
    )
    # An explicit error, not an assertion, so it holds under python -O.
    for lineage in (
        SimpleNamespace(normalized="normalized", reduction=None),
        SimpleNamespace(normalized=None, reduction="reduction"),
    ):
        with pytest.raises(ValueError) as caught:
            quality_control._require_doublet_graph(None, lineage)
        assert str(caught.value) == message


@pytest.mark.slow
def test_doublets_refuse_a_graph_whose_pca_they_cannot_project_onto(
    small_captures, tmp_path
):
    opened = _open_synthetic(small_captures, tmp_path)
    store = opened.store
    cells = store.snapshot_cell_selection("I")
    features = store.features.universe()

    def graph_on(coordinates: ArtifactRef) -> ArtifactRef:
        ann_index = store.graph.ann_index(coordinates)
        return store.graph.connectivity(
            store.graph.neighbors(ann_index, coordinates=coordinates, k=10)
        )

    def pca(normalized: ArtifactRef, **options: Any) -> ArtifactRef:
        return store.reduction.pca(normalized, dims=10, local_cache=False, **options)

    library_size = store.features.normalize(cells, features)
    store.RNA.normMethod = norm_pearson_residuals
    residuals = store.features.normalize(cells, features)
    store.RNA.normMethod = norm_lib_size
    unscaled = pca(library_size, feat_scaling=False)
    loadings = np.random.default_rng(0).normal(size=(int(store.RNA.feats.N), 10))
    normalizer = (
        "The graph was built on values normalized with "
        "cytearc.assay.pearson_residuals:theta=100.0:clip=sqrt-n, but doublet "
        "scoring normalizes its simulated doublets by library size and needs a "
        "graph of norm_lib_size values; normalize with cytearc.assay.norm_lib_size "
        "and build the PCA and graph again"
    )
    scaling = (
        "The graph was built on a PCA fitted without feature scaling, but doublet "
        "scoring projects its simulated doublets onto the PCA through a mapping "
        "reference, which needs feature scaling; fit the PCA again with "
        "feat_scaling=True and build the graph on it"
    )
    cases = [
        # Residuals are standardized, so their PCA scales no gene; the
        # normalizer is the cause, so it is named first.
        (graph_on(pca(residuals, feat_scaling=False)), {}, normalizer),
        (graph_on(pca(residuals)), {}, normalizer),
        (graph_on(unscaled), {}, scaling),
        # A Harmony graph is scored on the PCA that it corrected.
        (
            graph_on(store.reduction.harmony(unscaled, ["sample"])),
            {"sample_column": "sample"},
            scaling,
        ),
        (
            graph_on(store.reduction.custom(loadings, library_size)),
            {},
            "The graph was built on a reduction other than a PCA, but doublet "
            "scoring projects its simulated doublets onto the graph's PCA; build "
            "the graph on reduction.pca",
        ),
    ]
    kinds = ("doublet_score", "mapping_reference")
    before = [store.artifacts.list(kind=kind, from_assay="RNA") for kind in kinds]

    for graph, options, message in cases:
        with pytest.raises(ValueError) as caught:
            store.qc.doublets(opened.clusters, graph, **options)
        assert str(caught.value) == message

    # Nothing was planned or written, and a graph of scaled library-size
    # values still scores.
    assert [store.artifacts.list(kind=kind, from_assay="RNA") for kind in kinds] == (
        before
    )
    assert store.qc.doublets(opened.clusters, opened.graph).kind == "doublet_score"


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
    store.cells.insert("lonely_sample", by_cell_id(store.cells, lonely))
    with pytest.raises(
        ValueError, match=r"sample \(lonely_sample='C'\) has one selected cell"
    ):
        store.qc.doublets(opened.clusters, opened.graph, sample_column="lonely_sample")
    assert store.artifacts.list(kind="doublet_score", from_assay="RNA") == []

    # Sample A is one cluster: its doublets pair any two of its cells.
    one_type = np.where(samples == "A", "a", two_captures.types.astype(str))
    store.cells.insert("one_type_in_a", by_cell_id(store.cells, one_type))
    clusters = store.snapshot_cluster_labels(
        "one_type_in_a", cell_selection=store.snapshot_cell_selection("I")
    )
    with pytest.warns(DiagnosticWarning) as caught:
        ref = store.qc.doublets(
            clusters, opened.graph, sample_column="sample", heterotypic_fraction=0.8
        )
    summaries = _payload(store, ref).attrs["samples"]
    assert [summary["heterotypicFraction"] for summary in summaries] == [0.0, 0.8]
    (warning,) = [
        str(warning.message)
        for warning in caught
        if "same cluster label" in str(warning.message)
    ]
    assert warning.startswith("In 1 sample every cell has the same cluster label")
    assert warning.endswith(": (sample='A')")
    parents = _payload(store, ref)["simulated_parents"][:]
    in_b = samples[parents[:, 0]] == "B"
    labels = one_type[parents]
    # Most of B's doublets are forced to pair two clusters.
    assert np.mean(labels[in_b, 0] != labels[in_b, 1]) >= 0.8


def test_doublet_samples_group_cells_by_every_column_in_label_order():
    from functools import partial

    from cytearc.datastore._operations.quality_control import _doublet_sample_columns
    from cytearc.metadata.selection import sample_codes

    _doublet_sample_codes = partial(
        sample_codes,
        reason="Doublets are simulated within each sample, so every selected "
        "cell needs one",
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
    # Saved doublet scores record this fingerprint, so it must never change.
    assert fingerprint == (
        "213a045a1348dff0b4a5994a0d8b15f7a633fdda68918754a013f81bc174ecfc"
    )
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
    with pytest.raises(
        ValueError,
        match="'lane' has no label for 2 selected cells. Doublets are simulated",
    ):
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
    datastore_zarr_root, monkeypatch
):
    store = DataStore(datastore_zarr_root, default_assay="RNA", zarr_mode="r")
    # The other keywords of a public call, at their defaults.
    parameters = inspect.signature(PipelineAccessor.rna).parameters
    defaults = {
        name: parameter.default
        for name, parameter in parameters.items()
        if name not in {"self", "run_until", "callback", "cancel"}
    }
    # The columns that each resolved recipe freezes in its input snapshot.
    snapshots: list[dict[str, Any]] = []
    snapshot_stage = rna_recipe.input_snapshot_stage

    def recorded(cells, **options: Any):
        snapshots.append(options)
        return snapshot_stage(cells, **options)

    monkeypatch.setattr(rna_recipe, "input_snapshot_stage", recorded)

    def resolve(**overrides: Any):
        return rna_recipe.resolve_rna(store, **(defaults | overrides))

    def samples(**overrides: Any) -> list[str]:
        return resolve(**overrides).config["doubletSampleColumns"]

    pooled = resolve()
    assert pooled.config["doubletSampleColumns"] == []
    named = resolve(doublets={"sample_column": "ids"})
    assert named.config["doubletSampleColumns"] == ["ids"]
    # The recipe resolves the samples; the run records the setting as given.
    assert named.config["stages"]["doublets"] == {"sample_column": "ids"}
    # Doublet samples are frozen apart, so the cell snapshot, and the
    # identities of the stages that read it, do not change.
    pooled_snapshot, named_snapshot = snapshots
    assert (
        named_snapshot["cell_snapshot_columns"]
        == pooled_snapshot["cell_snapshot_columns"]
    )
    assert pooled_snapshot["extra_snapshots"] == {}
    assert named_snapshot["extra_snapshots"] == {"doublet_sample_snapshot": ("ids",)}
    assert samples(doublets={"sample_column": None}) == []
    # With Harmony, doublets are scored within each batch by default. The
    # batches are frozen for Harmony and for doublets apart from the cell
    # snapshot, which Harmony leaves as it is.
    assert samples(harmony={"batch_columns": ["ids", "names"]}) == ["ids", "names"]
    harmony_snapshot = snapshots[-1]
    assert (
        harmony_snapshot["cell_snapshot_columns"]
        == pooled_snapshot["cell_snapshot_columns"]
    )
    assert harmony_snapshot["extra_snapshots"] == {
        "harmony_snapshot": ("ids", "names"),
        "doublet_sample_snapshot": ("ids", "names"),
    }
    chosen = samples(
        harmony={"batch_columns": ["ids"]},
        doublets={"sample_column": "names"},
    )
    assert chosen == ["names"]
    assert samples(harmony={"batch_columns": ["ids"]}, doublets=False) == []
    with pytest.raises(ValueError, match="cannot be None"):
        resolve(
            harmony={"batch_columns": ["ids"]},
            doublets={"sample_column": None},
        )
    with pytest.raises(TypeError, match="non-empty string or None"):
        resolve(doublets={"sample_column": ""})
    with pytest.raises(KeyError, match="Doublet sample column 'missing'"):
        resolve(doublets={"sample_column": "missing"})
    with pytest.raises(ValueError, match="collide with reserved run fields"):
        resolve(snapshot_columns=("predicted_doublet",))


# Samples whose cells share one cluster label

_ONE_CLUSTER = "In 1 sample every cell has the same cluster label"


def _one_cluster_in_a(opened: SimpleNamespace, synthetic: _SyntheticDoublets) -> Any:
    """Return cluster labels that give every cell of sample A one label, and B its types."""
    labels = np.where(synthetic.samples == "A", "a", synthetic.types.astype(str))
    opened.store.cells.insert("one_type_in_a", by_cell_id(opened.store.cells, labels))
    return opened.store.snapshot_cluster_labels(
        "one_type_in_a", cell_selection=opened.store.snapshot_cell_selection("I")
    )


def test_a_sample_of_one_cluster_warns_when_computed_and_when_reused(
    small_captures, tmp_path
):
    opened = _open_synthetic(small_captures, tmp_path)
    clusters = _one_cluster_in_a(opened, small_captures)

    def score() -> ArtifactRef:
        return opened.store.qc.doublets(
            clusters, opened.graph, sample_column="sample", heterotypic_fraction=0.8
        )

    computed, first, first_texts = _warned(score, _ONE_CLUSTER)
    reused, second, second_texts = _warned(score, _ONE_CLUSTER)

    assert reused == computed
    labels = opened.store.cells.fetch_all("one_type_in_a")
    samples = small_captures.samples
    assert [len(set(labels[samples == sample])) for sample in ("A", "B")] == [1, 2]
    # Sample A forms none of the heterotypic doublets, and sample B its share.
    summaries = _payload(opened.store, computed).attrs["samples"]
    assert [summary["heterotypicFraction"] for summary in summaries] == [0.0, 0.8]
    assert opened.store.artifacts.inspect(computed).diagnostics == {
        "oneClusterSamples": 1,
        "oneClusterSampleExamples": ["(sample='A')"],
    }
    expected = (
        f"{_ONE_CLUSTER}, so its doublets pair any two of its cells instead of "
        "forcing heterotypic_fraction=0.8 of them to pair two clusters: "
        "(sample='A')"
    )
    for warned, texts in ((first, first_texts), (second, second_texts)):
        assert texts == [expected]
        (warning,) = warned
        assert (warning.operation, warning.diagnostic, warning.value) == (
            "run_doublet_detection",
            "oneClusterSamples",
            1,
        )

    # Each sample of one label of its own: two clusters in all, one in each.
    opened.store.cells.insert(
        "one_type_each",
        by_cell_id(opened.store.cells, np.where(samples == "A", "a", "b")),
        overwrite=True,
    )
    each = opened.store.snapshot_cluster_labels(
        "one_type_each", cell_selection=opened.store.snapshot_cell_selection("I")
    )
    both, warned, texts = _warned(
        lambda: opened.store.qc.doublets(
            each, opened.graph, sample_column="sample", heterotypic_fraction=0.8
        ),
        "In each of 2 samples",
    )
    assert texts == [
        "In each of 2 samples every cell has the same cluster label, so their "
        "doublets pair any two of their cells instead of forcing "
        "heterotypic_fraction=0.8 of them to pair two clusters: (sample='A'), "
        "(sample='B')"
    ]
    assert [warning.value for warning in warned] == [2]
    summaries = _payload(opened.store, both).attrs["samples"]
    assert [summary["heterotypicFraction"] for summary in summaries] == [0.0, 0.0]


def test_doublets_that_pair_as_asked_give_no_one_cluster_warning(
    small_captures, tmp_path
):
    opened = _open_synthetic(small_captures, tmp_path)
    one_cluster = _one_cluster_in_a(opened, small_captures)

    # Each sample of the cell types holds two clusters, and without a
    # heterotypic fraction no sample is asked for doublets of two clusters.
    for clusters, fraction in ((opened.clusters, 0.8), (one_cluster, 0.0)):

        def score(clusters: Any = clusters, fraction: float = fraction) -> Any:
            return opened.store.qc.doublets(
                clusters,
                opened.graph,
                sample_column="sample",
                heterotypic_fraction=fraction,
            )

        for _ in range(2):
            ref, warned, texts = _warned(score, "")
            assert [w for w in warned if w.operation == "run_doublet_detection"] == []
            assert not [text for text in texts if "cluster label" in text]
        assert opened.store.artifacts.inspect(ref).diagnostics == {
            "oneClusterSamples": 0,
            "oneClusterSampleExamples": [],
        }


def test_doublet_scores_saved_without_diagnostics_are_computed_anew(
    small_captures, tmp_path
):
    opened = _open_synthetic(small_captures, tmp_path)
    store = opened.store

    def score() -> ArtifactRef:
        return store.qc.doublets(opened.clusters, opened.graph, sample_column="sample")

    saved = score()
    # As scores saved before doublet detection saved diagnostics.
    del artifact_group(store.zw, saved).attrs["diagnostics"]
    with pytest.raises(
        ValueError,
        match="holds no diagnostics, which run_doublet_detection saves with every "
        "artifact; call ds.qc.doublets again",
    ):
        store.artifacts.inspect(saved).diagnostics

    anew = score()

    assert anew != saved
    for name in (
        "values",
        "raw_scores",
        "predicted_doublets",
        "simulated_scores",
        "simulated_parents",
    ):
        np.testing.assert_array_equal(
            _payload(store, anew)[name][:], _payload(store, saved)[name][:]
        )
    # The earlier scores still select their singlets.
    singlets = store.select_cells(saved, value="predicted_doublets", include=[False])
    np.testing.assert_array_equal(
        read_stored_selection_mask(
            store.zw,
            singlets,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
        ~_payload(store, anew)["predicted_doublets"][:],
    )


# ATAC doublets, projected onto an LSI by TF-IDF ------------------------------

_TF_IDF = {"identity": "cytearc.assay.norm_tf_idf:selected-cell-df:total-count-tf"}
_LSI_REDUCTION = (
    "The graph was built on a reduction other than an LSI, but doublet scoring "
    "of an ATAC assay projects its simulated doublets onto the graph's LSI; "
    "build the graph on reduction.lsi"
)
_DOUBLET_ARRAYS = (
    "values",
    "raw_scores",
    "predicted_doublets",
    "simulated_scores",
    "simulated_parents",
)


@pytest.fixture(scope="module")
def one_capture_peaks(tmp_path_factory) -> _SyntheticDoublets:
    """One capture of peak counts, with a graph on an LSI of TF-IDF values."""
    path = tmp_path_factory.mktemp("one_capture_peaks") / "data.zarr"
    return _build_synthetic(path, _ONE_CAPTURE, assay="ATAC")


@pytest.fixture(scope="module")
def two_captures_peaks(tmp_path_factory) -> _SyntheticDoublets:
    """Two captures of peak counts, with a batch effect between them."""
    path = tmp_path_factory.mktemp("two_captures_peaks") / "data.zarr"
    return _build_synthetic(path, _TWO_CAPTURES, assay="ATAC")


def _lsi_projection(
    loadings: np.ndarray, idf: np.ndarray, *, renormalize_subset: bool = False
) -> doublets.LsiDoubletProjection:
    return doublets.LsiDoubletProjection(
        loadings,
        idf,
        np.zeros((1, loadings.shape[1]), dtype=np.float32),
        renormalize_subset=renormalize_subset,
    )


def test_lsi_projection_weights_summed_counts_by_tf_idf_by_hand():
    # Three selected peaks, two LSI dimensions, and each peak's IDF.
    loadings = np.array([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]])
    idf = np.array([1.0, 2.0, 0.5])
    # A doublet whose parents hold 3, 2 and 1 counts of the peaks together,
    # and one whose parents hold none.
    summed = csr_matrix(np.array([[3, 2, 1], [0, 0, 0]], dtype=np.uint32))
    before = summed.copy()

    # Its TF-IDF values are [3 * 1, 2 * 2, 1 * 0.5] / T, so its coordinates
    # are [3 + 4 * 0.5, -4 + 0.5 * 2] / T = [5, -3] / T, where T is the sum
    # of its parents' totals over every peak, here 16.
    by_totals = _lsi_projection(loadings, idf).project(
        summed, np.array([16.0, 0.0]), source="test"
    )
    np.testing.assert_allclose(by_totals, [[5 / 16, -3 / 16], [0, 0]], rtol=1e-12)
    # Renormalized over the selected peaks, T is the doublet's own 6 counts.
    renormalized = _lsi_projection(loadings, idf, renormalize_subset=True).project(
        summed, None, source="test"
    )
    np.testing.assert_allclose(renormalized, [[5 / 6, -3 / 6], [0, 0]], rtol=1e-12)
    # The counts are left as they were.
    assert summed.dtype == np.uint32
    assert (summed != before).nnz == 0
    # A stored zero of a doublet without counts divides by 1, as a cell's
    # zero total does, and stays at the origin.
    stored_zero = csr_matrix(
        (np.array([0], dtype=np.uint32), np.array([1]), np.array([0, 1])),
        shape=(1, 3),
    )
    for projection, denominators in (
        (_lsi_projection(loadings, idf), np.zeros(1)),
        (_lsi_projection(loadings, idf, renormalize_subset=True), None),
    ):
        np.testing.assert_array_equal(
            projection.project(stored_zero, denominators, source="test"), [[0, 0]]
        )
    with pytest.raises(ValueError, match="test needs one denominator per doublet"):
        _lsi_projection(loadings, idf).project(summed, np.ones(3), source="test")


@pytest.mark.parametrize(
    ("loadings", "idf", "shape", "message"),
    [
        (np.ones(3), np.ones(3), (1, 1), "matrix of features by dimensions"),
        (np.ones((3, 0)), np.ones(3), (1, 0), "matrix of features by dimensions"),
        (np.ones((3, 2)), np.ones(2), (1, 2), "one value per LSI feature"),
        (np.full((3, 2), np.nan), np.ones(3), (1, 2), "must be finite"),
        (np.ones((3, 2)), np.array([1.0, np.inf, 1.0]), (1, 2), "must be finite"),
        (np.ones((3, 2)), np.ones(3), (1, 3), "one column per LSI dimension"),
        (np.ones((3, 2)), np.ones(3), (2,), "one column per LSI dimension"),
    ],
)
def test_lsi_projection_checks_its_loadings_weights_and_coordinates(
    loadings, idf, shape, message
):
    with pytest.raises(ValueError, match=message):
        doublets.LsiDoubletProjection(
            loadings, idf, np.zeros(shape), renormalize_subset=False
        )


def test_lsi_projection_holds_its_loadings_and_charges_each_doublet():
    loadings = np.arange(12, dtype=np.float32).reshape(4, 3)
    projection = doublets.LsiDoubletProjection(
        loadings, np.ones(4), np.zeros((2, 3)), renormalize_subset=True
    )

    assert (projection.n_dims, projection.renormalize_subset) == (3, True)
    # The loadings are held in float64, beside the float64 weights.
    assert projection.loadings.dtype == np.float64
    assert projection.resident_bytes() == 4 * 3 * 8 + 4 * 8
    # Rows of at most 2 values give doublets of at most 4: their float64
    # values and int64 indices and a weighted copy, and the coordinates.
    counts = csr_matrix(np.array([[1, 0, 2, 0], [0, 3, 0, 0]]))
    assert projection.row_bytes(counts) == 4 * (2 * 8 + 8) + 3 * 16 + 64
    with pytest.raises(TypeError, match="renormalize_subset must be a boolean"):
        doublets.LsiDoubletProjection(
            loadings, np.ones(4), np.zeros((2, 3)), renormalize_subset=1
        )


def test_pca_projection_charges_its_model_and_each_doublet_by_hand():
    model = ScaledPCAProjectionModel(
        feature_means=np.zeros(5),
        feature_scales=np.ones(5),
        center=np.zeros(5),
        loadings=np.zeros((5, 3)),
    )
    reference = SimpleNamespace(
        model=model, normalization_parameters={"renormalize_subset": False}
    )

    projection = doublets.PcaDoubletProjection(reference)

    assert (projection.n_dims, projection.renormalize_subset) == (3, False)
    # The float64 loadings, means, scales and center.
    assert projection.resident_bytes() == (15 + 5 + 5 + 5) * 8
    # Per doublet: a float64 copy of its counts of the 5 features, its
    # normalized values with their scaled and centered copies, and its
    # coordinates.
    counts = csr_matrix(np.ones((2, 5), dtype=np.int16))
    assert projection.row_bytes(counts) == 5 * (8 + 3 * 8) + 3 * 32 + 64


class _ChargedProjection:
    """A projection that holds and charges set bytes and counts what it projects."""

    def __init__(self, n_cells: int, *, resident: int = 0, row: int = 0) -> None:
        self.n_dims = 2
        self.renormalize_subset = True
        self.coordinates = np.zeros((n_cells, 2), dtype=np.float32)
        self.resident, self.row, self.projected = resident, row, 0

    def resident_bytes(self) -> int:
        return self.resident

    def row_bytes(self, counts: csr_matrix) -> int:
        return self.row

    def project(
        self, summed: csr_matrix, denominators: Any, *, source: str
    ) -> np.ndarray:
        self.projected += summed.shape[0]
        return np.zeros((summed.shape[0], 2))


def test_doublet_scoring_charges_what_its_projection_holds_and_projects():
    budget = ResourceBudget(1_000_000, 1)
    # A projection that holds more than the budget is refused before any
    # count is read; without an assay, a read would fail otherwise.
    with pytest.raises(MemoryError):
        doublets.score_synthetic_doublets(
            None,
            _ChargedProjection(4, resident=2_000_000),
            np.arange(4),
            np.zeros(4),
            np.zeros(4, dtype=np.int64),
            np.arange(3),
            None,
            index=_INDEX,
            cluster_sample_fraction=1.0,
            max_cells_per_cluster=None,
            simulation_ratio=2.0,
            heterotypic_fraction=0.0,
            n_neighbors=2,
            expected_doublet_rate=0.05,
            threshold=None,
            random_seed=0,
            resources=budget,
        )
    # A doublet that the projection needs more than the budget for is
    # refused before any is projected.
    counts = np.arange(1, 13, dtype=np.int16).reshape(4, 3)
    assay = SimpleNamespace(rawData=ChunkedArray.from_numpy(counts), name="ATAC")
    (plan,) = doublets.plan_doublet_samples([4], simulation_ratio=2.0, n_neighbors=2)

    def simulate(projection: _ChargedProjection) -> Any:
        return doublets._simulate_sample(
            assay,
            projection,
            np.random.default_rng(0),
            np.arange(4),
            plan,
            active_indices=np.arange(4),
            labels=np.zeros(4, dtype=np.int64),
            feature_indices=np.arange(3),
            cell_totals=None,
            cluster_sample_fraction=1.0,
            max_cells_per_cluster=None,
            heterotypic_fraction=0.0,
            resources=budget,
            resident_bytes=0,
        )

    refused = _ChargedProjection(4, row=2_000_000)
    with pytest.raises(MemoryError):
        simulate(refused)
    assert refused.projected == 0
    # With room for its doublets, the projection maps every one of them.
    admitted = _ChargedProjection(4, row=1_000)
    coordinates, _parents, _fraction = simulate(admitted)
    assert admitted.projected == plan.simulated == len(coordinates) == 8


@pytest.mark.parametrize(
    ("top_n", "renormalize"),
    [(None, False), (150, False), (150, True)],
    ids=["every-peak", "prevalent-peaks", "renormalized"],
)
def test_lsi_projection_of_the_cells_reproduces_their_stored_lsi(
    one_capture_peaks, tmp_path, top_n, renormalize
):
    from cytearc.graph.feature_projection import resolve_graph_assay_inputs
    from cytearc.storage.feature_selection import read_feature_selection_indices
    from cytearc.storage.selections import read_stored_selection_indices

    opened = _open_synthetic(one_capture_peaks, tmp_path)
    store = opened.store
    graph = opened.graph
    truth = one_capture_peaks.truth
    # With renormalize_subset, IDF over the cells that are not homotypic
    # doublets, so that it is not that of every cell either.
    keep = truth != 2 if renormalize else np.ones(len(truth), dtype=bool)
    if top_n is not None:
        # The template's LSI holds every peak, over which a cell's total and
        # its total of the selected peaks agree; over 150 prevalent peaks
        # they differ.
        store.cells.insert("kept", by_cell_id(store.cells, keep))
        cells = store.snapshot_cell_selection("kept")
        peaks = store.features.prevalent_peaks(cells, top_n=top_n)
        normalized = store.features.normalize(
            cells, peaks, renormalize_subset=renormalize
        )
        lsi = store.reduction.lsi(normalized, dims=8, local_cache=False)
        graph = store.graph.connectivity(
            store.graph.neighbors(store.graph.ann_index(lsi), coordinates=lsi, k=10)
        )
    lineage = resolve_graph_assay_inputs(store.zw, graph, "ATAC")
    rows = read_stored_selection_indices(
        store.zw,
        lineage.cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64)
    features = read_feature_selection_indices(
        store.zw, "ATAC", lineage.feature_selection
    )
    np.testing.assert_array_equal(rows, np.flatnonzero(keep))
    assert len(features) == (400 if top_n is None else top_n)

    projection, totals = store._lsi_doublet_projection(
        store.ATAC, lineage, rows, features, resident_bytes=0
    )

    assert projection.renormalize_subset is renormalize
    assert (totals is None) is renormalize
    # The counts that the store was written with, apart from its reads.
    counts, *_ = _synthetic_counts(3, _ONE_CAPTURE, peaks=True)
    if totals is not None:
        # Each cell's total over every peak, not over the LSI's peaks.
        np.testing.assert_array_equal(totals, counts[rows].sum(axis=1))
        if top_n is not None:
            selected = counts[rows][:, features].sum(axis=1)
            assert np.mean(totals > selected) > 0.9
    coordinates = projection.project(
        csr_matrix(counts[rows][:, features]), totals, source="test"
    )
    # The LSI holds float32 coordinates of float32 TF-IDF values.
    stored = np.asarray(artifact_group(store.zw, lineage.reduction)["data"][:])
    np.testing.assert_allclose(
        coordinates, stored, rtol=0, atol=1e-6 * np.abs(stored).max()
    )


def test_atac_doublet_scores_count_simulated_doublets_among_exact_lsi_neighbors(
    one_capture_peaks, tmp_path
):
    opened = _open_synthetic(one_capture_peaks, tmp_path)
    store = opened.store
    ref = store.qc.doublets(opened.clusters, opened.graph)
    payload = _payload(store, ref)
    (sample,) = payload.attrs["samples"]
    k = sample["adjustedNeighbors"]
    lsi = artifact_group(store.zw, _graph_pca(store, opened.graph))

    # The doublets of the parents that the scores record, rebuilt from the
    # written counts as TF-IDF defines them: summed counts over the parents'
    # summed totals, times log2(1 + n / (df + 1)) of each peak among the n
    # cells, times the LSI loadings.
    counts, *_ = _synthetic_counts(3, _ONE_CAPTURE, peaks=True)
    counts = counts.astype(np.float64)
    idf = np.log2(1 + len(counts) / ((counts > 0).sum(axis=0) + 1))
    parents = payload["simulated_parents"][:]
    summed = counts[parents[:, 0]] + counts[parents[:, 1]]
    simulated = (summed / summed.sum(axis=1, keepdims=True) * idf) @ lsi["loadings"][:]
    points = np.vstack([lsi["data"][:], simulated]).astype(np.float32)
    # Each cell's simulated doublets among its k nearest other points, by
    # Euclidean distance.
    values = points.astype(np.float64)
    squared = (values**2).sum(axis=1)
    distances = squared[:, None] + squared[None, :] - 2 * values @ values.T
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances[: len(counts)], axis=1, kind="stable")[:, :k]
    exact = np.count_nonzero(nearest >= len(counts), axis=1)

    np.testing.assert_array_equal(np.rint(payload["raw_scores"][:] * k), exact)


def test_atac_doublets_recover_injected_heterotypic_doublets(
    one_capture_peaks, two_captures_peaks, tmp_path
):
    for synthetic, options in (
        (one_capture_peaks, {}),
        (two_captures_peaks, {"sample_column": "sample"}),
    ):
        opened = _open_synthetic(synthetic, tmp_path / synthetic.path.parent.name)
        ref = opened.store.qc.doublets(opened.clusters, opened.graph, **options)

        assert (ref.kind, ref.assay) == ("doublet_score", "ATAC")
        _assert_recovered(
            _payload(opened.store, ref), synthetic.truth, synthetic.samples
        )


def test_atac_doublet_scores_are_the_same_for_any_nthreads(
    two_captures_peaks, tmp_path
):
    scored = []
    for nthreads in (1, 4):
        target = tmp_path / f"threads_{nthreads}" / "data.zarr"
        shutil.copytree(two_captures_peaks.path, target)
        store = DataStore(
            str(target),
            default_assay="ATAC",
            min_features_per_cell=0,
            nthreads=nthreads,
        )
        (graph,) = store.artifacts.list(kind="connectivity_map", from_assay="ATAC")
        (clusters,) = store.artifacts.list(kind="cluster_labels", scope="datastore")
        # Two samples, which four threads score at once.
        ref = store.qc.doublets(clusters, graph, sample_column="sample")
        payload = _payload(store, ref)
        scored.append({name: payload[name][:] for name in _DOUBLET_ARRAYS})

    single, parallel = scored
    for name in _DOUBLET_ARRAYS:
        np.testing.assert_array_equal(single[name], parallel[name])


def test_the_lsi_doublet_graph_check_names_what_it_needs(monkeypatch):
    from cytearc.datastore._operations import quality_control

    statuses = {
        "normalized": SimpleNamespace(
            parameters={"normalization_method": {"external_hook": True, **_TF_IDF}}
        ),
        "reduction": SimpleNamespace(operation="run_lsi"),
    }
    monkeypatch.setattr(
        quality_control, "inspect_artifact", lambda _root, ref: statuses[ref]
    )
    lineage = SimpleNamespace(normalized="normalized", reduction="reduction")

    def refusal() -> str:
        with pytest.raises(ValueError) as caught:
            quality_control._require_lsi_doublet_graph(None, lineage)
        return str(caught.value)

    # TF-IDF passes, recorded with or without its external_hook key.
    quality_control._require_lsi_doublet_graph(None, lineage)
    statuses["normalized"].parameters = {"normalization_method": _TF_IDF}
    quality_control._require_lsi_doublet_graph(None, lineage)
    # A reduction other than an LSI is refused.
    statuses["reduction"].operation = "run_pca"
    assert refusal() == _LSI_REDUCTION
    # The normalizer is checked first and named as provenance records it.
    statuses["normalized"].parameters = {
        "normalization_method": {
            "external_hook": True,
            "identity": "cytearc.assay.norm_clr:feature-axis",
        }
    }
    assert refusal() == (
        "The graph was built on values normalized with "
        "cytearc.assay.norm_clr:feature-axis, but doublet scoring of an ATAC "
        "assay weights its simulated doublets by TF-IDF and needs a graph of "
        "norm_tf_idf values; normalize with cytearc.assay.norm_tf_idf and build "
        "the LSI and graph again"
    )
    statuses["normalized"].parameters = None
    assert refusal().startswith("The graph was built on values normalized with None")


@pytest.mark.slow
def test_atac_doublets_refuse_a_graph_whose_lsi_they_cannot_project_onto(
    two_captures_peaks, tmp_path
):
    opened = _open_synthetic(two_captures_peaks, tmp_path)
    store = opened.store
    cells = store.snapshot_cell_selection("I")
    peaks = store.features.universe()

    def graph_on(coordinates: ArtifactRef, **index: Any) -> ArtifactRef:
        ann_index = store.graph.ann_index(coordinates, **index)
        return store.graph.connectivity(
            store.graph.neighbors(ann_index, coordinates=coordinates, k=10)
        )

    def lsi(normalized: ArtifactRef) -> ArtifactRef:
        return store.reduction.lsi(normalized, dims=8, local_cache=False)

    tf_idf = store.features.normalize(cells, peaks)
    store.ATAC.normMethod = norm_clr
    clr = store.features.normalize(cells, peaks)
    store.ATAC.normMethod = norm_tf_idf
    loadings = np.random.default_rng(0).normal(size=(int(store.ATAC.feats.N), 8))
    cases = [
        (
            graph_on(lsi(clr)),
            {},
            "The graph was built on values normalized with "
            "cytearc.assay.norm_clr:feature-axis, but doublet scoring of an ATAC "
            "assay weights its simulated doublets by TF-IDF and needs a graph of "
            "norm_tf_idf values; normalize with cytearc.assay.norm_tf_idf and "
            "build the LSI and graph again",
        ),
        (
            graph_on(store.reduction.pca(tf_idf, dims=8, local_cache=False)),
            {},
            _LSI_REDUCTION,
        ),
        (graph_on(store.reduction.custom(loadings, tf_idf)), {}, _LSI_REDUCTION),
        (graph_on(tf_idf), {}, "Doublet detection requires an uncorrected LSI graph"),
        (
            graph_on(store.reduction.harmony(lsi(tf_idf), ["sample"])),
            {},
            "Doublet detection on a batch-corrected graph needs sample_column, the "
            "column of each cell's capture: doublets are then simulated and scored "
            "within each sample, on the LSI that the correction started from. To "
            "score every cell together, pass an uncorrected LSI graph",
        ),
        # A Harmony graph is scored on the LSI that it corrected.
        (
            graph_on(store.reduction.harmony(lsi(clr), ["sample"])),
            {"sample_column": "sample"},
            "The graph was built on values normalized with "
            "cytearc.assay.norm_clr:feature-axis",
        ),
    ]
    before = store.artifacts.list(kind="doublet_score", from_assay="ATAC")

    for graph, options, message in cases:
        with pytest.raises(ValueError) as caught:
            store.qc.doublets(opened.clusters, graph, **options)
        assert str(caught.value).startswith(message)

    # Nothing was planned or written, and the LSI graph of TF-IDF values
    # still scores.
    assert store.artifacts.list(kind="doublet_score", from_assay="ATAC") == before
    assert store.qc.doublets(opened.clusters, opened.graph).kind == "doublet_score"


def test_a_session_that_normalizes_otherwise_scores_a_tf_idf_graph_alike(
    one_capture_peaks, tmp_path, monkeypatch
):
    from cytearc.assay import normalization

    scored = []
    for normalizer in (norm_tf_idf, norm_clr):
        opened = _open_synthetic(one_capture_peaks, tmp_path / normalizer.__name__)
        store = opened.store
        # A graph of the cells that are not homotypic doublets and of 150
        # prevalent peaks, whose summary prevalent peaks saves, so that the
        # document frequencies are neither those of every cell nor of every
        # peak.
        keep = one_capture_peaks.truth != 2
        store.cells.insert("kept", by_cell_id(store.cells, keep))
        cells = store.snapshot_cell_selection("kept")
        normalized = store.features.normalize(
            cells, store.features.prevalent_peaks(cells, top_n=150)
        )
        lsi = store.reduction.lsi(normalized, dims=8, local_cache=False)
        graph = store.graph.connectivity(
            store.graph.neighbors(store.graph.ann_index(lsi), coordinates=lsi, k=10)
        )
        clusters = store.snapshot_cluster_labels("cell_type", cell_selection=cells)
        summaries = store.artifacts.list(kind="feature_summary", from_assay="ATAC")
        store.ATAC.normMethod = normalizer
        with monkeypatch.context() as patch:
            if normalizer is norm_tf_idf:
                # A TF-IDF session reads the document frequencies from the
                # cells' summary and counts nothing itself.
                def counted(*_args: Any, **_kwargs: Any) -> None:
                    raise AssertionError("the document frequencies were counted")

                patch.setattr(normalization, "stream_document_frequency", counted)

            payload = _payload(store, store.qc.doublets(clusters, graph))

        scored.append({name: payload[name][:] for name in _DOUBLET_ARRAYS})
        # Another session's summary would count other values, so that session
        # counts the cells of each peak itself and saves no summary.
        assert (
            store.artifacts.list(kind="feature_summary", from_assay="ATAC") == summaries
        )
    for name in _DOUBLET_ARRAYS:
        np.testing.assert_array_equal(scored[0][name], scored[1][name])


_FEW_DETECTABLE = (
    "the automatic threshold leaves fewer than 1% of the simulated doublets "
    "above it, so it calls almost no cell"
)


def test_a_threshold_that_calls_almost_no_doublet_warns_when_computed_and_reused(
    small_captures, tmp_path, monkeypatch
):
    opened = _open_synthetic(small_captures, tmp_path)
    store = opened.store

    def score(**options: Any) -> ArtifactRef:
        return store.qc.doublets(opened.clusters, opened.graph, **options)

    # The thresholds of these samples leave many simulated doublets above
    # them, and a threshold that the call sets is the call's choice.
    for options in ({"sample_column": "sample"}, {"threshold": 0.9999}):
        _ref, warned, _texts = _warned(lambda: score(**options), "")
        assert warned == []
    # A threshold above every simulated doublet, as the minimum method can
    # choose when one or two outlying doublets form its second mode.
    monkeypatch.setattr(
        doublets,
        "doublet_threshold",
        lambda values: (float(np.max(values)), "minimum"),
    )
    cases = (
        ({"sample_column": "sample", "random_seed": 1}, "In each of 2 samples"),
        ({"random_seed": 1}, "In 1 sample"),
    )
    computed = [_warned(lambda: score(**options), where) for options, where in cases]
    monkeypatch.undo()
    reused = [_warned(lambda: score(**options), where) for options, where in cases]

    suffix = (
        ". The scores still rank the cells; pass threshold to call those that "
        "score above it"
    )
    expected = [
        f"In each of 2 samples {_FEW_DETECTABLE}: (sample='A'), (sample='B'){suffix}",
        f"In 1 sample {_FEW_DETECTABLE}{suffix}",
    ]
    for (first, first_warned, first_texts), (again, warned, texts), text in zip(
        computed, reused, expected, strict=True
    ):
        assert again == first
        # Each sample records the threshold's method and that none of its
        # simulated doublets is above it.
        samples = _payload(store, first).attrs["samples"]
        assert {sample["thresholdMethod"] for sample in samples} == {"minimum"}
        assert [sample["detectableDoubletFraction"] for sample in samples] == [
            0.0
        ] * len(samples)
        for diagnostic, messages in ((first_warned, first_texts), (warned, texts)):
            assert messages == [text]
            (warning,) = diagnostic
            assert (warning.operation, warning.diagnostic, warning.value) == (
                "run_doublet_detection",
                "detectableDoubletFraction",
                [0.0] * len(samples),
            )


@pytest.mark.parametrize("assay", ["RNA", "ATAC"])
def test_a_cosine_graph_scores_rna_by_its_metric_and_atac_by_euclidean_distances(
    small_captures, one_capture_peaks, tmp_path, assay
):
    synthetic = one_capture_peaks if assay == "ATAC" else small_captures
    opened = _open_synthetic(synthetic, tmp_path)
    store = opened.store
    coordinates = _graph_pca(store, opened.graph)
    # The index of the template's graph with another metric alone.
    cosine = store.graph.connectivity(
        store.graph.neighbors(
            store.graph.ann_index(coordinates, ann_metric="cosine"),
            coordinates=coordinates,
            k=10,
        )
    )

    ref = store.qc.doublets(opened.clusters, cosine)
    euclidean = store.qc.doublets(opened.clusters, opened.graph)

    assert ref != euclidean
    same = [
        np.array_equal(
            _payload(store, ref)[name][:], _payload(store, euclidean)[name][:]
        )
        for name in ("values", "raw_scores")
    ]
    # An RNA sample's index takes the graph's metric, and an ATAC sample's
    # Euclidean distances in the LSI whatever the graph's metric.
    assert same == ([True, True] if assay == "ATAC" else [False, False])


def test_a_harmony_graph_scores_each_sample_on_the_lsi_it_corrected(
    two_captures_peaks, tmp_path
):
    opened = _open_synthetic(two_captures_peaks, tmp_path)
    store = opened.store
    correction = store.reduction.harmony(_graph_pca(store, opened.graph), ["sample"])
    corrected = store.graph.connectivity(
        store.graph.neighbors(
            store.graph.ann_index(correction), coordinates=correction, k=10
        )
    )

    ref = store.qc.doublets(opened.clusters, corrected, sample_column="sample")
    uncorrected = store.qc.doublets(
        opened.clusters, opened.graph, sample_column="sample"
    )

    assert ref != uncorrected
    for name in _DOUBLET_ARRAYS:
        np.testing.assert_array_equal(
            _payload(store, ref)[name][:], _payload(store, uncorrected)[name][:]
        )


def test_doublets_refuse_an_assay_that_is_neither_rna_nor_atac(tmp_path):
    counts, _truth, _samples, types = _synthetic_counts(3, _SMALL_CAPTURES)
    path = tmp_path / "data.zarr"
    write_count_store(str(path), {"ADT": counts[:, :20]}, "uint32")
    store = DataStore(str(path), default_assay="ADT", min_features_per_cell=0)
    store.cells.insert("cell_type", by_cell_id(store.cells, types.astype(str)))
    cells = store.snapshot_cell_selection("I")
    normalized = store.features.normalize(cells, store.features.universe())
    pca = store.reduction.pca(normalized, dims=5, local_cache=False)
    graph = store.graph.connectivity(
        store.graph.neighbors(store.graph.ann_index(pca), coordinates=pca, k=10)
    )
    clusters = store.snapshot_cluster_labels("cell_type", cell_selection=cells)

    with pytest.raises(TypeError) as caught:
        store.qc.doublets(clusters, graph)

    assert str(caught.value) == (
        "Doublet detection is only supported for RNA and ATAC assays; received ADTassay"
    )
    assert store.artifacts.list(kind="doublet_score", from_assay="ADT") == []


@pytest.mark.slow
def test_atac_doublets_of_an_integrated_graph_score_its_atac_lsi(tmp_path):
    rna, _truth, _samples, types = _synthetic_counts(3, _SMALL_CAPTURES)
    peaks, *_ = _synthetic_counts(3, _SMALL_CAPTURES, peaks=True)
    path = tmp_path / "data.zarr"
    write_count_store(str(path), {"RNA": rna, "ATAC": peaks}, "uint32")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=2
    )
    store.cells.insert("cell_type", by_cell_id(store.cells, types.astype(str)))
    cells = store.snapshot_cell_selection("I")

    def graph(assay: str, reduction: str) -> tuple[ArtifactRef, ArtifactRef]:
        normalized = store.features.normalize(
            cells, store.features.universe(from_assay=assay)
        )
        reduce = getattr(store.reduction, reduction)
        coordinates = reduce(normalized, dims=8, local_cache=False)
        neighbors = store.graph.neighbors(
            store.graph.ann_index(coordinates), coordinates=coordinates, k=10
        )
        return neighbors, store.graph.connectivity(neighbors)

    rna_neighbors, rna_graph = graph("RNA", "pca")
    atac_neighbors, atac_graph = graph("ATAC", "lsi")
    _neighbors, atac_pca_graph = graph("ATAC", "pca")
    clusters = store.snapshot_cluster_labels("cell_type", cell_selection=cells)
    wnn = store.integration.modalities([rna_neighbors, atac_neighbors])

    # Without smoothing, the scores read only the ATAC branch of the graph.
    integrated = store.qc.doublets(clusters, wnn, from_assay="ATAC")
    native = store.qc.doublets(clusters, atac_graph)

    assert integrated != native and integrated.assay == "ATAC"
    for name in _DOUBLET_ARRAYS:
        np.testing.assert_array_equal(
            _payload(store, integrated)[name][:], _payload(store, native)[name][:]
        )
    # The RNA branch of the same graph is scored on its PCA, as before.
    assert store.qc.doublets(clusters, wnn, from_assay="RNA").assay == "RNA"
    # An integrated graph whose ATAC branch has no LSI is refused.
    snn = store.integration.modalities([rna_graph, atac_pca_graph], method="snn")
    with pytest.raises(ValueError) as caught:
        store.qc.doublets(clusters, snn, from_assay="ATAC")
    assert str(caught.value) == _LSI_REDUCTION


@pytest.mark.slow  # Two ATAC runs of the 500-cell PBMC fixture and a planted copy.
def test_atac_doublets_rank_planted_doublets_of_distinct_types_above_singlets(
    tmp_path,
):
    # The fixture's cells, their clusters and LSI, from a default run.
    source = tmp_path / "source"
    with tarfile.open(full_path("500_pbmc_atac.zarr.tar.gz"), "r:gz") as archive:
        archive.extractall(source, filter="data")
    store = DataStore(str(source))
    run = store.pipeline.atac(label="cells", umap=False, markers=False)
    rows = np.flatnonzero(run.cells.fetch_all("I"))
    names, code = np.unique(
        np.asarray(run.cells.fetch("clusters")), return_inverse=True
    )
    lsi = np.asarray(store.artifacts.load(run["lsi"])["data"][:], dtype=np.float64)
    counts = store.ATAC.rawData[rows, :].compute(nthreads=1, msg="Reading counts")
    # Clusters whose LSI centroids lie farther apart than the median pair are
    # distinct types; a doublet of two such cells is one that simulated
    # doublets can stand for.
    centroids = np.array(
        [lsi[code == index].mean(axis=0) for index in range(len(names))]
    )
    apart = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)
    distinct = apart > np.median(apart[np.triu_indices(len(names), 1)])
    rng = np.random.default_rng(0)
    first = rng.choice(np.flatnonzero(distinct[code].any(axis=1)), 60, replace=False)
    second = [rng.choice(np.flatnonzero(distinct[code[cell]][code])) for cell in first]
    # 30 doublets hold both cells' fragments and 30 half of them, as many as
    # a singlet holds.
    summed = counts[first].astype(np.int64) + counts[second]
    planted = np.vstack([summed[:30], rng.binomial(summed[30:], 0.5)])
    truth = np.repeat([0, 1, 2], [len(rows), 30, 30])
    location = tmp_path / "planted.zarr"
    write_count_store(str(location), {"ATAC": np.vstack([counts, planted])}, "uint32")
    planted_store = DataStore(str(location), min_features_per_cell=0)

    planted_run = planted_store.pipeline.atac(
        label="planted", filtering=False, umap=False, markers=False, doublets=True
    )

    scores = np.asarray(planted_run.cells.fetch("doublet_score"))
    features = np.asarray(planted_store.cells.fetch_all("ATAC_nFeatures"))

    def auroc(values: np.ndarray, kinds: tuple[int, ...]) -> float:
        keep = np.isin(truth, (0, *kinds))
        return float(roc_auc_score(truth[keep] > 0, values[keep]))

    # The score ranks the planted doublets above the singlets, and above
    # what their detected peaks do, also those at a singlet's depth, whose
    # peaks barely set them apart.
    assert auroc(scores, (1, 2)) > 0.9
    assert auroc(scores, (2,)) > 0.9
    assert auroc(scores, (1, 2)) > auroc(features, (1, 2)) + 0.05
    assert auroc(scores, (2,)) > auroc(features, (2,)) + 0.1
