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
from cytearc.assay import (
    PearsonResidualNormalizer,
    norm_clr,
    norm_dummy,
    norm_lib_size,
    norm_lib_size_log,
    norm_pearson_residuals,
    norm_tf_idf,
)
from cytearc.datastore._pipeline.recipes import rna as rna_recipe
from cytearc.datastore.pipeline_accessor import PipelineAccessor
from cytearc.mapping.models import ScaledPCAProjectionModel
from cytearc.matrix import ChunkedArray
from cytearc.quality_control import doublets
from cytearc.storage.artifacts import ArtifactRef, artifact_group, callable_identity
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.pipeline_runs import load_pipeline_stage_records
from cytearc.storage.selections import read_stored_selection_mask
from cytearc.utils.logging import logger
from tests import full_path
from tests.storage_helpers import (
    by_cell_id,
    insert_nullable_cell_column,
    record_normalized_values,
    write_count_store,
)
from tests.test_doublet_streaming import _exact_simulated_neighbors
from tests.test_reduction_graph_diagnostics import _warned

_INDEX = doublets.DoubletIndexSettings(metric="l2", m=16, ef_construction=100, ef=50)


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
    # The same values satisfy the minimum and Otsu's split differently; cells
    # of the low mode alone call none.
    cells = np.zeros(10)
    assert doublets.doublet_threshold(values, cell_scores=cells)[1] == "minimum"


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
    # Without a minimum, Otsu's threshold stands for cells that score below
    # the peak. A symmetric peak splits at its middle.
    threshold, method = doublets.doublet_threshold(unimodal, cell_scores=np.zeros(100))
    assert (threshold, method) == (bins / 2 - 0.5, "otsu")
    assert threshold == doublets.otsu_threshold(unimodal)


def test_doublet_threshold_has_none_without_two_modes_when_cells_score_alike():
    # The peak of one maximum above; cells that score like the simulated
    # doublets have half of them above Otsu's threshold, as half of the
    # doublets are: a doublet rate of 100%, which Otsu's threshold stood for
    # before, calling half of the cells.
    bins = doublets.THRESHOLD_BINS
    unimodal = _bin_values([min(i + 1, bins - i) for i in range(bins)], bins)
    otsu = doublets.otsu_threshold(unimodal)
    assert np.mean(unimodal > otsu) == 0.5

    assert doublets.doublet_threshold(unimodal, cell_scores=unimodal) == (
        None,
        "none",
    )
    # Cells of which a quarter score above Otsu's threshold estimate 50%,
    # which still stands; one cell more above it does not.
    lower, upper = unimodal[unimodal <= otsu], unimodal[unimodal > otsu]
    called = upper[: len(upper) // 4]
    cells = np.concatenate([lower[: 3 * len(called)], called])
    assert np.mean(cells > otsu) == 0.25
    assert doublets.doublet_threshold(unimodal, cell_scores=cells) == (otsu, "otsu")
    assert doublets.doublet_threshold(
        unimodal, cell_scores=np.append(cells, upper[0])
    ) == (None, "none")


def test_doublet_threshold_has_none_beside_a_low_outlier_when_cells_score_alike():
    # One population of simulated doublets and one that scores far below the
    # rest, which keeps a maximum of its own: the minimum above it leaves
    # every doublet but the outlier above it, and calls every cell of the
    # population, as the kept minimum did before.
    rng = np.random.default_rng(23)
    values = np.concatenate([[0.0], rng.normal(0.5, 0.05, 2_000)])
    cells = rng.normal(0.5, 0.05, 1_000)
    minimum = doublets.histogram_minimum_threshold(values)
    assert minimum is not None and np.count_nonzero(values <= minimum) == 1
    assert np.count_nonzero(cells > minimum) == len(cells)
    # Otsu's threshold splits the population, cells and doublets alike.
    otsu = doublets.otsu_threshold(values)
    assert np.mean(cells > otsu) / np.mean(values > otsu) > 0.8

    assert doublets.doublet_threshold(values, cell_scores=cells) == (None, "none")
    # Cells of a lower population, as singlets beside doublets of two types,
    # take Otsu's threshold.
    singlets = rng.normal(0.2, 0.02, 1_000)
    assert doublets.doublet_threshold(values, cell_scores=singlets) == (otsu, "otsu")


def test_doublet_index_settings_are_validated():
    with pytest.raises(ValueError, match="'l2' or 'cosine'"):
        doublets.DoubletIndexSettings("ip", 16, 50, 50)
    with pytest.raises(ValueError, match="m must be at least 2"):
        doublets.DoubletIndexSettings("l2", 1, 50, 50)
    with pytest.raises(TypeError, match="ef_construction must be an integer"):
        doublets.DoubletIndexSettings("l2", 16, 5.0, 50)
    with pytest.raises(ValueError, match="ef must be at least 1"):
        doublets.DoubletIndexSettings("l2", 16, 50, 0)


def test_doublet_threshold_takes_otsu_when_an_outlier_forms_the_second_mode():
    # A body of two modes whose dip smooths away before the top bin, which
    # holds one outlying doublet, loses its maximum: the minimum then falls
    # in the empty bins below the outlier, above every other doublet.
    rng = np.random.default_rng(4)
    body = np.concatenate([rng.normal(0.05, 0.02, 700), rng.normal(0.15, 0.05, 300)])
    values = np.concatenate([np.clip(body, 0.0, None), [0.9]])
    minimum = doublets.histogram_minimum_threshold(values)
    assert minimum is not None and np.count_nonzero(values > minimum) == 1
    # Singlets score like the doublets of one type, and 2% of the cells are
    # doublets of two types.
    cells = np.concatenate(
        [np.clip(rng.normal(0.05, 0.02, 980), 0.0, None), rng.normal(0.2, 0.03, 20)]
    )

    threshold, method = doublets.doublet_threshold(values, cell_scores=cells)

    assert method == "otsu"
    assert threshold == doublets.otsu_threshold(values)
    # Otsu's threshold splits the body, with its high mode above it, and
    # estimates a doublet rate below 50%.
    detectable = np.mean(values > threshold)
    assert 0.2 < detectable < 0.4
    assert np.mean(cells > threshold) / detectable < 0.2


@pytest.mark.parametrize(
    ("low", "high", "method"),
    [(95, 5, "minimum"), (96, 4, "otsu"), (5, 95, "minimum"), (4, 96, "otsu")],
)
def test_doublet_threshold_needs_five_percent_of_doublets_on_each_side_of_the_minimum(
    low, high, method
):
    # Two blocks at the ends of the range: the minimum is the center of the
    # second bin, with the high block above it and the low block below.
    values = np.repeat([0.0, 1.0], [low, high])
    minimum = doublets.histogram_minimum_threshold(values)
    assert minimum is not None and np.count_nonzero(values > minimum) == high

    # No cell scores above either threshold, so Otsu's threshold estimates a
    # doublet rate of 0.
    threshold, chosen = doublets.doublet_threshold(values, cell_scores=np.zeros(50))

    assert chosen == method
    assert threshold == (
        minimum if method == "minimum" else doublets.otsu_threshold(values)
    )


@pytest.mark.parametrize(
    ("low", "high", "called", "method"),
    [
        # 4% of the doublets lie above Otsu's threshold, so 1 cell of 50
        # estimates a doublet rate of 50%, and 2 cells of 100%.
        (96, 4, 1, "otsu"),
        (96, 4, 2, "none"),
        # 96% lie above it: 24 cells of 50 estimate 50%, and 25 cells 52%.
        (4, 96, 24, "otsu"),
        (4, 96, 25, "none"),
    ],
)
def test_doublet_threshold_takes_otsu_only_where_its_doublet_rate_is_at_most_half(
    low, high, called, method
):
    # Two blocks at the ends of the range, whose minimum leaves fewer than 5%
    # of the doublets on one side, and cells at the same two scores.
    values = np.repeat([0.0, 1.0], [low, high])
    cells = np.repeat([0.0, 1.0], [50 - called, called])
    otsu = doublets.otsu_threshold(values)
    assert np.mean(cells > otsu) == called / 50
    assert np.mean(values > otsu) == high / 100

    threshold, chosen = doublets.doublet_threshold(values, cell_scores=cells)

    assert (threshold, chosen) == (
        (otsu, "otsu") if method == "otsu" else (None, "none")
    )


@pytest.mark.parametrize(("called", "method"), [(25, "minimum"), (26, "none")])
def test_doublet_threshold_keeps_a_minimum_only_where_its_doublet_rate_is_at_most_half(
    called, method
):
    # Two blocks of 50 doublets at the ends of the range: the minimum leaves
    # half of them on each side. 25 of 100 cells above it estimate a doublet
    # rate of 50%, and 26 of 52%, as in a sample of mostly doublets. Otsu's
    # threshold, in the first bin, splits the doublets and cells alike.
    values = np.repeat([0.0, 1.0], [50, 50])
    cells = np.repeat([0.0, 1.0], [100 - called, called])
    minimum = doublets.histogram_minimum_threshold(values)
    assert minimum is not None and np.mean(values > minimum) == 0.5
    assert np.mean(cells > minimum) == called / 100
    otsu = doublets.otsu_threshold(values)
    assert otsu < minimum and np.mean(values > otsu) == 0.5
    assert np.mean(cells > otsu) == called / 100

    threshold, chosen = doublets.doublet_threshold(values, cell_scores=cells)

    assert (threshold, chosen) == (
        (minimum, "minimum") if method == "minimum" else (None, "none")
    )


def test_doublet_threshold_takes_otsu_where_its_minimum_would_call_too_many_cells():
    # A low mode of 7% of the simulated doublets in the first ten bins, ten
    # empty bins, and a main mode that rises to bin 40 and thins out to the
    # last bins, whose heights fall by one so that none is a maximum.
    bins = doublets.THRESHOLD_BINS
    low = [8, 20, 36, 48, 56, 48, 36, 20, 8, 4]
    fall = [round(30 - 27 * (i - 40) / 212) for i in range(40, 253)] + [2, 1, 0]
    values = _bin_values(low + [0] * 10 + list(range(1, 21)) + fall, bins)
    # The minimum, the first empty bin, leaves the low mode below it, and
    # Otsu's threshold splits the main mode's tail.
    minimum = doublets.histogram_minimum_threshold(values)
    assert minimum == 10.5
    assert 0.05 < np.mean(values <= minimum) < 0.1
    otsu = doublets.otsu_threshold(values)
    assert otsu > 100
    # Most cells score in the main mode, and two in its tail: the minimum
    # would call 72 of 102 cells, an estimate of 76%, and Otsu's threshold
    # the two, an estimate of 5%.
    cells = np.repeat([0.5, 30.5, 200.5], [30, 70, 2])
    assert np.mean(cells > minimum) / np.mean(values > minimum) > 0.7
    assert np.flatnonzero(cells > otsu).tolist() == [100, 101]
    assert np.mean(cells > otsu) / np.mean(values > otsu) < 0.1

    assert doublets.doublet_threshold(values, cell_scores=cells) == (otsu, "otsu")


def test_doublet_threshold_has_none_when_doublets_look_like_the_cells():
    # Simulated doublets and cells of one population score alike, and one
    # doublet scores far above the rest: the minimum before it leaves too
    # few doublets above it, and Otsu's threshold splits the one mode with
    # as many cells as doublets above it, a doublet rate of about 100%.
    rng = np.random.default_rng(17)
    values = np.concatenate([rng.normal(0.05, 0.01, 2_000), [0.3]])
    cells = rng.normal(0.05, 0.01, 1_000)
    minimum = doublets.histogram_minimum_threshold(values)
    assert minimum is not None and np.count_nonzero(values > minimum) == 1
    otsu = doublets.otsu_threshold(values)
    assert np.mean(cells > otsu) / np.mean(values > otsu) > 0.8

    # The sample has no threshold, where the minimum was kept before.
    assert doublets.doublet_threshold(values, cell_scores=cells) == (None, "none")


# Single clusters of real fixtures scored alone by qc.doublets with
# random_seed=4444, two simulated doublets per cell: the number of simulated
# doublets among the k neighbors of each simulated doublet and of each cell,
# as {count: points}, from which the scores at any expected doublet rate
# follow. Each sample's simulated doublets look like its cells, and the
# threshold of master's rule (1.5.0 development) called the cells given.
_ONE_CLUSTER_SAMPLES = {
    # The largest cluster of the 1k PBMC fixture, 197 cells, with save_k=10
    # (k = 30) and expected_doublet_rate=0.5: Otsu's threshold of a histogram
    # of one maximum called 100 cells.
    "one maximum": (
        30,
        0.5,
        {14: 1, 15: 1, 16: 4, 17: 5, 18: 8, 19: 16, 20: 30, 21: 33, 22: 53, 23: 65}
        | {24: 66, 25: 48, 26: 40, 27: 17, 28: 6, 29: 1},
        {12: 1, 13: 2, 15: 1, 16: 3, 17: 8, 18: 5, 19: 11, 20: 17, 21: 20, 22: 29}
        | {23: 23, 24: 32, 25: 23, 26: 12, 27: 9, 28: 1},
        100,
    ),
    # Its second largest cluster, 183 cells, at the defaults (k = 90): the
    # minimum above one low outlier called 182 cells.
    "low outlier": (
        90,
        0.05,
        {53: 1, 57: 2, 58: 1, 60: 7, 61: 9, 62: 15, 63: 26, 64: 37, 65: 35, 66: 41}
        | {67: 43, 68: 40, 69: 40, 70: 31, 71: 19, 72: 12, 73: 5, 74: 2},
        {54: 1, 56: 3, 57: 2, 58: 2, 59: 4, 60: 6, 61: 7, 62: 9, 63: 14, 64: 17}
        | {65: 12, 66: 14, 67: 20, 68: 18, 69: 17, 70: 12, 71: 12, 72: 5, 73: 5}
        | {74: 2, 76: 1},
        182,
    ),
    # The largest cluster of the PBMC ATAC fixture, 190 cells, at the
    # defaults: a minimum with 54% of the simulated doublets above it called
    # 105 cells.
    "minimum": (
        90,
        0.05,
        {52: 1, 56: 5, 57: 9, 58: 19, 59: 19, 60: 30, 61: 36, 62: 30, 63: 26, 64: 35}
        | {65: 38, 66: 30, 67: 31, 68: 29, 69: 19, 70: 18, 71: 4, 72: 1},
        {55: 3, 56: 2, 57: 6, 58: 3, 59: 10, 60: 9, 61: 13, 62: 22, 63: 17, 64: 19}
        | {65: 14, 66: 20, 67: 17, 68: 15, 69: 9, 70: 6, 71: 4, 73: 1},
        105,
    ),
}


@pytest.mark.parametrize("case", list(_ONE_CLUSTER_SAMPLES))
def test_doublet_threshold_has_none_for_single_clusters_of_real_fixtures(case):
    k, rate, simulated_counts, cell_counts, called = _ONE_CLUSTER_SAMPLES[case]

    def scores(table: dict[int, int]) -> np.ndarray:
        counts = np.repeat(list(table), list(table.values()))
        return doublets.doublet_likelihood(
            counts, k, expected_doublet_rate=rate, simulation_ratio=2.0
        )

    simulated, cells = scores(simulated_counts), scores(cell_counts)
    assert (len(simulated), len(cells)) == (2 * len(cells), len(cells))
    minimum = doublets.histogram_minimum_threshold(simulated)
    otsu = doublets.otsu_threshold(simulated)
    otsu_rate = np.mean(cells > otsu) / np.mean(simulated > otsu)
    # The case, and the threshold that master's rule took in it.
    if case == "one maximum":
        assert minimum is None
        taken = otsu
    else:
        assert minimum is not None
        above = np.mean(simulated > minimum)
        if case == "low outlier":
            assert np.count_nonzero(simulated <= minimum) == 1
        else:
            assert 0.05 <= min(above, 1 - above)
        taken = minimum
    assert np.count_nonzero(cells > taken) == called
    # Otsu's threshold, and the threshold taken, call cells at a rate above
    # 50% of the simulated doublets above them.
    assert otsu_rate > 0.5
    assert np.mean(cells > taken) / np.mean(simulated > taken) > 0.5

    assert doublets.doublet_threshold(simulated, cell_scores=cells) == (None, "none")


@pytest.mark.parametrize(("cell", "expected"), [(0.4, "otsu"), (0.41, "none")])
def test_doublet_threshold_of_equal_simulated_scores_detects_no_doublet(cell, expected):
    # Equal scores have no minimum, and Otsu's threshold is their value, with
    # no simulated doublet above it. Cells at the value call none and leave
    # it standing; a cell above it estimates an unbounded doublet rate.
    values = np.full(30, 0.4)
    cells = np.array([0.1, 0.4, cell])

    threshold, method = doublets.doublet_threshold(values, cell_scores=cells)

    assert method == expected
    assert threshold == (0.4 if expected == "otsu" else None)


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        (np.array([]), "non-empty one-dimensional"),
        (np.array([0.1, np.inf]), "must be finite"),
    ],
)
def test_doublet_threshold_validates_the_cell_scores(cells, message):
    with pytest.raises(ValueError, match=message):
        doublets.doublet_threshold(np.array([0.1, 0.2, 0.3]), cell_scores=cells)


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


@pytest.mark.parametrize("split_points", [1_000, 100], ids=["one-part", "parts"])
def test_count_simulated_neighbors_counts_the_exact_nearest_doublets(
    monkeypatch, split_points
):
    # 100 points form one index, or 4 parts once samples of 100 points split.
    monkeypatch.setattr(doublets, "_INDEX_SPLIT_POINTS", split_points)
    rng = np.random.default_rng(5)
    observed = rng.normal(size=(40, 4))
    # Doublets lie between two observed cells, as added profiles do.
    simulated = (
        observed[rng.integers(0, 40, 60)] + observed[rng.integers(0, 40, 60)]
    ) / 2
    simulated += rng.normal(scale=0.05, size=simulated.shape)
    points = np.vstack([observed, simulated]).astype(np.float32)
    settings = doublets.DoubletIndexSettings("l2", 16, 200, 200)

    counts = [
        doublets.count_simulated_neighbors(
            points,
            40,
            12,
            settings,
            random_seed=3,
            workers=workers,
            resources=ResourceBudget(256 * 1024**2, workers),
        )
        for workers in (1, 4, 8)
    ]

    np.testing.assert_array_equal(counts[0], _exact_counts(points, 40, 12))
    # Every index is built on one thread, so the counts are the same for any
    # number of workers.
    for other in counts[1:]:
        np.testing.assert_array_equal(other, counts[0])


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


def test_count_simulated_neighbors_refuses_more_points_than_keys_hold(monkeypatch):
    # The merge keys each candidate's position in 32 bits; with 3, at most 7
    # points have positions that fit.
    monkeypatch.setattr(doublets, "_POSITION_BITS", np.uint64(7))

    def count(n_points: int) -> np.ndarray:
        return doublets.count_simulated_neighbors(
            np.random.default_rng(0).normal(size=(n_points, 2)).astype(np.float32),
            3,
            2,
            _INDEX,
            random_seed=0,
            workers=1,
            resources=ResourceBudget(256 * 1024**2, 1),
        )

    assert count(7).shape == (7,)
    with pytest.raises(ValueError, match=r"must number fewer than 2\*\*32"):
        count(8)


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
    path: Path,
    design: dict[str, dict[str, Any]],
    *,
    assay: str = "RNA",
    normalizer: Any = None,
    normalization: dict[str, Any] | None = None,
    feat_scaling: bool = True,
) -> _SyntheticDoublets:
    """Write the design's counts with a graph and the cell types as clusters.

    An RNA graph is built on a PCA of library-size values, or of the values
    of ``normalizer`` with the flags of ``normalization`` and the PCA's
    ``feat_scaling``, and an ATAC graph on an LSI of TF-IDF values.
    """
    counts, truth, samples, types = _synthetic_counts(3, design, peaks=assay == "ATAC")
    write_count_store(str(path), {assay: counts}, "uint32")
    store = DataStore(
        str(path), default_assay=assay, min_features_per_cell=0, nthreads=2
    )
    store.cells.insert("sample", by_cell_id(store.cells, samples))
    store.cells.insert("cell_type", by_cell_id(store.cells, types.astype(str)))
    if normalizer is not None:
        store.RNA.normMethod = normalizer
    cells = store.snapshot_cell_selection("I")
    normalized = store.features.normalize(
        cells, store.features.universe(), **(normalization or {})
    )
    reduction = (
        store.reduction.lsi(normalized, dims=8, local_cache=False)
        if assay == "ATAC"
        else store.reduction.pca(
            normalized, dims=10, feat_scaling=feat_scaling, local_cache=False
        )
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


def _doublet_revisions_up_to(monkeypatch: pytest.MonkeyPatch, revision: int) -> None:
    """Cut the registry back to ``run_doublet_detection`` revision ``revision``.

    The registry is then the one that the release which saved such scores knew.
    """
    revisions = operation_revisions.OPERATION_REVISIONS
    monkeypatch.setattr(
        operation_revisions,
        "OPERATION_REVISIONS",
        MappingProxyType(
            {
                **revisions,
                "run_doublet_detection": revisions["run_doublet_detection"][
                    : revision - 1
                ],
            }
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize("earlier", [1, 2, 3])
def test_doublet_artifact_of_an_earlier_revision_stays_readable_and_is_recomputed(
    one_capture, tmp_path, monkeypatch, earlier
):
    opened = _open_synthetic(one_capture, tmp_path)
    store = opened.store
    _doublet_revisions_up_to(monkeypatch, earlier)
    legacy = store.qc.doublets(opened.clusters, opened.graph)
    assert store.artifacts.inspect(legacy).revision == earlier
    saved = {name: _payload(store, legacy)[name][:] for name in _DOUBLET_ARRAYS}
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
    assert store.artifacts.inspect(current).revision == 4
    # The log and the lineage name each change since the saved revision.
    changes = "; ".join(
        entry.change
        for entry in operation_revisions.OPERATION_REVISIONS["run_doublet_detection"]
        if entry.revision > earlier
    )
    assert changes.endswith("affects scores with an automatic threshold")
    (line,) = [message for message in messages if "Recomputing" in message]
    assert line == (
        f"Recomputing run_doublet_detection: artifact {legacy.artifact_id[:12]} "
        f"is revision {earlier}, current 4: {changes}"
    )
    assert store.qc.doublets(opened.clusters, opened.graph) == current
    # The earlier scores stay as they were saved: loadable, and traced as
    # stale by the revisions after them.
    legacy_status = store.artifacts.inspect(legacy)
    assert (legacy_status.revision, legacy_status.is_current) == (earlier, False)
    assert [entry.revision for entry in legacy_status.superseded_by] == list(
        range(earlier + 1, 5)
    )
    payload = store.artifacts.load(legacy)
    for name, values in saved.items():
        np.testing.assert_array_equal(payload[name][:], values)
    markdown = store.artifacts.lineage(legacy).to_markdown()
    assert "- Status: `stale`" in markdown
    assert f"- Revision: `{earlier}, current 4: {changes}`" in markdown


def test_doublet_scores_of_a_given_threshold_stay_current_at_revision_4(
    small_captures, tmp_path, monkeypatch
):
    # Revision 4 changes only how an automatic threshold is chosen, so scores
    # of revision 3 with a given threshold call the same cells and are reused.
    opened = _open_synthetic(small_captures, tmp_path)
    store = opened.store
    _doublet_revisions_up_to(monkeypatch, 3)
    given = store.qc.doublets(opened.clusters, opened.graph, threshold=0.3)
    automatic = store.qc.doublets(opened.clusters, opened.graph)
    monkeypatch.undo()

    assert store.qc.doublets(opened.clusters, opened.graph, threshold=0.3) == given
    status = store.artifacts.inspect(given)
    assert (status.revision, status.current_revision, status.is_current) == (
        3,
        3,
        True,
    )
    assert status.superseded_by == ()
    # Scores of an automatic threshold are computed anew.
    recomputed = store.qc.doublets(opened.clusters, opened.graph)
    assert recomputed != automatic
    assert store.artifacts.inspect(recomputed).revision == 4
    assert [
        entry.revision for entry in store.artifacts.inspect(automatic).superseded_by
    ] == [4]


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


@pytest.mark.parametrize(
    ("recorded", "named"),
    [
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
            {"identity": "cytearc.assay.norm_tf_idf:selected-cell-df:total-count-tf"},
            "cytearc.assay.norm_tf_idf:selected-cell-df:total-count-tf",
        ),
        # Options that no PearsonResidualNormalizer takes name no built-in.
        (
            {"identity": "cytearc.assay.pearson_residuals:theta=-1.0:clip=sqrt-n"},
            "cytearc.assay.pearson_residuals:theta=-1.0:clip=sqrt-n",
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
        "scoring normalizes its simulated doublets as the graph's cells were"
    )
    # Each built-in RNA normalizer passes, recorded with or without its
    # external_hook key, on a PCA with or without feature scaling.
    for normalizer in (
        norm_lib_size,
        norm_lib_size_log,
        norm_pearson_residuals,
        PearsonResidualNormalizer(theta=50.0, clip=4.0),
        norm_clr,
        norm_dummy,
    ):
        identity = callable_identity(normalizer)
        for method in (identity, {"external_hook": True, **identity}):
            for scaling in (True, False):
                statuses["normalized"].parameters = {"normalization_method": method}
                statuses["reduction"].parameters = {"feat_scaling": scaling}
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


def _halve_counts(_assay: Any, counts: Any) -> Any:
    """A custom normalizer, defined at module level so provenance can name it."""
    return counts / 2


@pytest.mark.slow
def test_doublets_score_a_pca_graph_of_any_builtin_and_refuse_the_others(
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
    store.RNA.normMethod = _halve_counts
    halved = store.features.normalize(cells, features)
    store.RNA.normMethod = norm_lib_size
    unscaled = pca(library_size, feat_scaling=False)
    loadings = np.random.default_rng(0).normal(size=(int(store.RNA.feats.N), 10))
    refused = [
        (
            graph_on(pca(halved)),
            "The graph was built on values normalized with "
            "tests.test_doublet_detection._halve_counts, but doublet scoring "
            "normalizes its simulated doublets as the graph's cells were, which it "
            "can do for the built-in RNA normalizers alone: norm_lib_size, "
            "norm_lib_size_log, norm_pearson_residuals, norm_clr and norm_dummy; "
            "normalize with one of them and build the PCA and graph again",
        ),
        (
            graph_on(store.reduction.custom(loadings, library_size)),
            "The graph was built on a reduction other than a PCA, but doublet "
            "scoring projects its simulated doublets onto the graph's PCA; build "
            "the graph on reduction.pca",
        ),
    ]
    kinds = ("doublet_score", "mapping_reference")
    before = [store.artifacts.list(kind=kind, from_assay="RNA") for kind in kinds]

    for graph, message in refused:
        with pytest.raises(ValueError) as caught:
            store.qc.doublets(opened.clusters, graph)
        assert str(caught.value) == message
    # Nothing was planned or written.
    assert [store.artifacts.list(kind=kind, from_assay="RNA") for kind in kinds] == (
        before
    )
    # Residuals with or without feature scaling, library-size values without
    # it, and a Harmony graph of those, which is scored on the PCA it
    # corrected, each score.
    for graph, options in (
        (graph_on(pca(residuals, feat_scaling=False)), {}),
        (graph_on(pca(residuals)), {}),
        (graph_on(unscaled), {}),
        (
            graph_on(store.reduction.harmony(unscaled, ["sample"])),
            {"sample_column": "sample"},
        ),
    ):
        assert store.qc.doublets(opened.clusters, graph, **options).kind == (
            "doublet_score"
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


# RNA doublets of the other built-in normalizers -------------------------------

# Each normalizer with the flags and PCA scaling that a run of it takes by
# default: Pearson residuals are already standardized, so their PCA does not
# scale them.
_RNA_NORMALIZERS = {
    "lib_size_log": (norm_lib_size_log, {}, True),
    "pearson_residuals": (norm_pearson_residuals, {}, False),
    "clr": (norm_clr, {}, True),
    "passthrough": (norm_dummy, {"log_transform": True}, True),
}


@pytest.fixture(scope="module", params=sorted(_RNA_NORMALIZERS))
def one_capture_normalized(request, tmp_path_factory) -> tuple[str, _SyntheticDoublets]:
    """One capture with a graph on the values of each built-in RNA normalizer."""
    normalizer, flags, scaling = _RNA_NORMALIZERS[request.param]
    path = tmp_path_factory.mktemp(f"one_capture_{request.param}") / "data.zarr"
    return request.param, _build_synthetic(
        path,
        _ONE_CAPTURE,
        normalizer=normalizer,
        normalization=flags,
        feat_scaling=scaling,
    )


def _normalized_by_hand(
    name: str,
    cells: np.ndarray,
    rows: np.ndarray,
    totals: np.ndarray,
    *,
    size_factor: float,
) -> np.ndarray:
    """Normalize ``rows`` of counts as ``cells``, the counts of every cell, were.

    ``totals`` holds each row's total over every gene. Each normalizer is
    written out from its definition.
    """
    if name == "lib_size_log":
        return np.log1p(size_factor * rows / totals[:, None])
    if name == "passthrough":
        return np.log1p(rows)
    if name == "clr":
        return np.log1p(rows / np.exp(np.log1p(cells).mean(axis=0)))
    # Analytic Pearson residuals with theta 100, clipped at the square root
    # of the number of cells, and 0 where no count is expected.
    mu = totals[:, None] * (cells.sum(axis=0) / cells.sum())
    residuals = np.zeros_like(mu)
    positive = mu > 0
    residuals[positive] = (rows[positive] - mu[positive]) / np.sqrt(
        mu[positive] + mu[positive] ** 2 / 100
    )
    return np.clip(residuals, -np.sqrt(len(cells)), np.sqrt(len(cells)))


# The automatic threshold of the passthrough scores calls almost no cell
# (#546), which a warning reports; this test checks the scores.
@pytest.mark.filterwarnings("ignore:In 1 sample the automatic threshold")
def test_rna_doublets_of_each_normalizer_count_simulated_doublets_among_exact_neighbors(
    one_capture_normalized, tmp_path
):
    name, synthetic = one_capture_normalized
    opened = _open_synthetic(synthetic, tmp_path)
    store = opened.store
    # Doublets are normalized as the graph's cells were, whatever normalizer
    # the session sets.
    store.RNA.normMethod = norm_lib_size
    payload = _payload(store, store.qc.doublets(opened.clusters, opened.graph))
    (sample,) = payload.attrs["samples"]
    k = sample["adjustedNeighbors"]
    pca = store.artifacts.inspect(_graph_pca(store, opened.graph))
    stored = artifact_group(store.zw, pca.ref)
    size_factor = store.artifacts.inspect(pca.input_ref("normalized")).parameters[
        "size_factor"
    ]

    # The doublets of the parents that the scores record, rebuilt from the
    # written counts of every gene, the graph's features, and projected as
    # the PCA projected its cells: scaled by the PCA's feature scaling when
    # it scaled them, then centered and multiplied by the loadings.
    cells = _synthetic_counts(3, _ONE_CAPTURE)[0].astype(np.float64)
    parents = payload["simulated_parents"][:]
    summed = cells[parents[:, 0]] + cells[parents[:, 1]]
    values = _normalized_by_hand(
        name, cells, summed, summed.sum(axis=1), size_factor=size_factor
    )
    if pca.parameters["feat_scaling"]:
        scaling = artifact_group(store.zw, pca.input_ref("feature_scaling"))
        values = (values - scaling["mean"][:]) / scaling["scale"][:]
    simulated = (values - stored["center"][:]) @ stored["loadings"][:]
    points = np.vstack([stored["data"][:], simulated]).astype(np.float32)
    low, high = _exact_simulated_neighbors(points, len(cells), k)
    low, high = low[: len(cells)], high[: len(cells)]
    scored = np.rint(payload["raw_scores"][:] * k)

    assert pca.parameters["feat_scaling"] is _RNA_NORMALIZERS[name][2]
    assert np.all((low <= scored) & (scored <= high))
    # Almost every cell has no tie at its k-th neighbor, and its count is the
    # exact one.
    assert np.count_nonzero(low == high) >= 0.99 * len(cells)
    # Heterotypic doublets rank above singlets under every normalizer.
    truth, scores = synthetic.truth, payload["values"][:]
    assert roc_auc_score((truth == 1)[truth != 2], scores[truth != 2]) >= 0.98


def test_passthrough_doublets_refuse_an_assay_of_values_normalized_before_import(
    tmp_path,
):
    # A passthrough graph of such values would sum two cells' normalized
    # values into a doublet, so doublet scoring refuses the assay.
    path = tmp_path / "data.zarr"
    synthetic = _build_synthetic(
        path,
        _SMALL_CAPTURES,
        normalizer=norm_dummy,
        normalization={"log_transform": True},
    )
    record_normalized_values(path, "RNA")
    opened = _open_synthetic(synthetic, tmp_path / "copy")

    with pytest.raises(TypeError) as caught:
        opened.store.qc.doublets(opened.clusters, opened.graph)

    assert str(caught.value).startswith(
        "Doublet scoring needs counts: it simulates each doublet by adding two "
        "cells' counts. Assay 'RNA' holds normalized values"
    )
    assert opened.store.artifacts.list(kind="doublet_score", from_assay="RNA") == []


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
    with pytest.raises(ValueError, match="one value per row of counts"):
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


@pytest.mark.parametrize(
    ("normalizer", "arrays", "values_per_feature"),
    [
        # Library-size values with their scaled and centered copies.
        (norm_lib_size, {}, 3),
        # Pearson residuals hold their expected counts, residuals and three
        # temporaries of their square roots beside those copies.
        (norm_pearson_residuals, {"pearson_feature_fractions": np.full(5, 0.2)}, 7),
    ],
)
def test_pca_projection_charges_its_model_and_each_doublet_by_hand(
    normalizer, arrays, values_per_feature
):
    model = ScaledPCAProjectionModel(
        feature_means=np.zeros(5),
        feature_scales=np.ones(5),
        center=np.zeros(5),
        loadings=np.zeros((5, 3)),
    )
    reference = SimpleNamespace(
        model=model,
        normalization_arrays=arrays,
        normalization_parameters={
            "normalization_method": callable_identity(normalizer),
            "size_factor": 1000.0,
            "log_transform": False,
            "renormalize_subset": False,
        },
        selected_cell_count=9,
    )

    projection = doublets.PcaDoubletProjection(reference)

    assert (projection.n_dims, projection.renormalize_subset) == (3, False)
    # The float64 loadings, means, scales and center, and fitted fractions.
    held = 5 * len(arrays)
    assert projection.resident_bytes() == (15 + 5 + 5 + 5 + held) * 8
    # Per doublet: a float64 copy of its counts of the 5 features, what
    # normalizing them holds, and its coordinates.
    counts = csr_matrix(np.ones((2, 5), dtype=np.int16))
    assert projection.row_bytes(counts) == (
        5 * (8 + values_per_feature * 8) + 3 * 32 + 64
    )


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
    # exact Euclidean distance. A point whose distance ties the k-th to within
    # the float32 rounding of the index may count either way, as it does on
    # one cell of this store, so the exact distances bound each count.
    low, high = _exact_simulated_neighbors(points, len(counts), k)
    low, high = low[: len(counts)], high[: len(counts)]
    scored = np.rint(payload["raw_scores"][:] * k)

    assert np.all((low <= scored) & (scored <= high))
    # Almost every cell has no such tie, and its count is the exact one.
    assert np.count_nonzero(low == high) >= 0.99 * len(counts)


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
_SET_THRESHOLD = (
    "The scores still rank the cells; pass threshold, which applies to every "
    "sample, to call those that score above it"
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
        lambda values, *, cell_scores: (float(np.max(values)), "minimum"),
    )
    cases = (
        ({"sample_column": "sample", "random_seed": 1}, "In each of 2 samples"),
        ({"random_seed": 1}, "In 1 sample"),
    )
    computed = [_warned(lambda: score(**options), where) for options, where in cases]
    monkeypatch.undo()
    reused = [_warned(lambda: score(**options), where) for options, where in cases]

    suffix = f". {_SET_THRESHOLD}"
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


def test_the_warning_still_fires_when_only_otsus_threshold_splits_the_doublets(
    small_captures, tmp_path, monkeypatch
):
    opened = _open_synthetic(small_captures, tmp_path)
    store = opened.store

    def one_outlier(points, n_observed, adjusted_neighbors, *args, **kwargs):
        # No point has a simulated doublet among its neighbors but the last
        # simulated doublet, all of whose neighbors are.
        counts = np.zeros(len(points), dtype=np.int64)
        counts[-1] = adjusted_neighbors
        return counts

    # The minimum leaves one simulated doublet above it, too few, and Otsu's
    # threshold leaves the same one.
    monkeypatch.setattr(doublets, "count_simulated_neighbors", one_outlier)
    computed = _warned(
        lambda: store.qc.doublets(opened.clusters, opened.graph, random_seed=2),
        "In 1 sample",
    )
    monkeypatch.undo()
    reused = _warned(
        lambda: store.qc.doublets(opened.clusters, opened.graph, random_seed=2),
        "In 1 sample",
    )

    (ref, _diagnostic, texts), (again, _again, again_texts) = computed, reused
    assert again == ref
    (sample,) = _payload(store, ref).attrs["samples"]
    assert sample["thresholdMethod"] == "otsu"
    assert sample["detectableDoubletFraction"] == 1 / sample["simulatedDoublets"]
    assert sample["predictedDoublets"] == 0
    expected = f"In 1 sample {_FEW_DETECTABLE}. {_SET_THRESHOLD}"
    assert texts == again_texts == [expected]


_NO_THRESHOLD = (
    "every automatic threshold would estimate a doublet rate above 50%, as in a "
    "sample of one cell type, whose doublets look like its cells, so the sample "
    "has no threshold and calls no cell"
)


def _counts_alike(points, n_observed, adjusted_neighbors, *args, **kwargs):
    """Count simulated neighbors of cells and simulated doublets alike.

    Points take every count from 0 to ``adjusted_neighbors`` in turn, so the
    cells' counts and the simulated doublets' counts spread the same way, as
    in a sample whose simulated doublets look like its cells.
    """
    return np.arange(len(points), dtype=np.int64) % (adjusted_neighbors + 1)


def test_a_sample_without_a_threshold_calls_no_cell_and_warns_when_computed_and_reused(
    small_captures, tmp_path, monkeypatch
):
    opened = _open_synthetic(small_captures, tmp_path)
    store = opened.store

    def score() -> ArtifactRef:
        return store.qc.doublets(
            opened.clusters, opened.graph, sample_column="sample", random_seed=3
        )

    monkeypatch.setattr(doublets, "count_simulated_neighbors", _counts_alike)
    computed = _warned(score, "In each of 2 samples")
    monkeypatch.undo()
    reused = _warned(score, "In each of 2 samples")

    (ref, first, first_texts), (again, warned, texts) = computed, reused
    assert again == ref
    payload = _payload(store, ref)
    simulated, scores = payload["simulated_scores"][:], payload["values"][:]
    samples = payload.attrs["samples"]
    # The simulated doublets of A come first, then those of B.
    bounds = np.cumsum([0, *(sample["simulatedDoublets"] for sample in samples)])
    for sample, label, start, stop in zip(
        samples, ("A", "B"), bounds[:-1], bounds[1:], strict=True
    ):
        cells = scores[small_captures.samples == label]
        sample_simulated = simulated[start:stop]
        # Otsu's threshold splits cells and simulated doublets alike.
        otsu = doublets.otsu_threshold(sample_simulated)
        assert np.mean(cells > otsu) / np.mean(sample_simulated > otsu) > 0.5
        # The sample records no threshold, no call, no detectable doublet,
        # and so no estimate.
        assert sample["labels"] == {"sample": label}
        assert (sample["thresholdMethod"], sample["threshold"]) == ("none", None)
        assert sample["predictedDoublets"] == 0
        assert sample["detectableDoubletFraction"] == 0.0
        assert sample["estimatedDoubletRate"] is None
    calls = payload["predicted_doublets"][:]
    assert not calls.any()
    # Every scored cell is a singlet that select_cells keeps.
    singlets = store.select_cells(ref, value="predicted_doublets", include=[False])
    assert np.count_nonzero(
        read_stored_selection_mask(
            store.zw,
            singlets,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
    ) == len(calls)
    expected = (
        f"In each of 2 samples {_NO_THRESHOLD}: (sample='A'), (sample='B'). "
        f"{_SET_THRESHOLD}"
    )
    for diagnostic, messages in ((first, first_texts), (warned, texts)):
        assert messages == [expected]
        (warning,) = diagnostic
        assert (warning.operation, warning.diagnostic, warning.value) == (
            "run_doublet_detection",
            "noThresholdSamples",
            2,
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


@pytest.mark.parametrize(("rate", "planted"), [(0.05, 1), (0.3, 8)])
def test_otsus_threshold_calls_planted_doublets_where_the_minimum_calls_most_cells(
    small_captures, tmp_path, rate, planted
):
    # The two captures pooled into one sample and scored on a cosine graph:
    # the histogram minimum leaves 5% or more of the simulated doublets on
    # each side but lies below about half of the cells, mostly singlets.
    opened = _open_synthetic(small_captures, tmp_path)
    store = opened.store
    coordinates = _graph_pca(store, opened.graph)
    cosine = store.graph.connectivity(
        store.graph.neighbors(
            store.graph.ann_index(coordinates, ann_metric="cosine"),
            coordinates=coordinates,
            k=10,
        )
    )

    ref = store.qc.doublets(opened.clusters, cosine, expected_doublet_rate=rate)

    payload = _payload(store, ref)
    (sample,) = payload.attrs["samples"]
    simulated, scores = payload["simulated_scores"][:], payload["values"][:]
    minimum = doublets.histogram_minimum_threshold(simulated)
    assert minimum is not None
    above = np.mean(simulated > minimum)
    assert min(above, 1 - above) >= 0.05
    assert np.mean(scores > minimum) / above > 0.5
    truth = small_captures.truth
    assert np.count_nonzero((scores > minimum) & (truth == 0)) > 50
    # Otsu's threshold estimates a plausible rate and calls planted doublets
    # of two types alone; a sample without a threshold would call none.
    calls = payload["predicted_doublets"][:]
    assert sample["thresholdMethod"] == "otsu"
    assert sample["estimatedDoubletRate"] <= 0.5
    assert np.all(truth[calls] == 1)
    assert np.count_nonzero(calls) >= planted


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
    # One simulated doublet far above the rest kept a mode of its own here,
    # and the minimum before it called no cell; Otsu's threshold, which
    # replaces a minimum that leaves too few doublets on a side, calls most
    # planted doublets and few singlets.
    (sample,) = _payload(planted_store, planted_run["doublets"]).attrs["samples"]
    assert sample["thresholdMethod"] == "otsu"
    calls = np.asarray(planted_run.cells.fetch("predicted_doublet"), dtype=bool)
    assert np.count_nonzero(calls[truth > 0]) >= 30
    assert np.mean(calls[truth == 0]) <= 0.02


@pytest.fixture(scope="module")
def pbmc_analysis_cells(tmp_path_factory) -> SimpleNamespace:
    """The counts and clusters of the 1k PBMC fixture's cells after a default run."""
    source = tmp_path_factory.mktemp("pbmc") / "source.zarr"
    with tarfile.open(full_path("1K_pbmc_citeseq.zarr.tar.gz")) as archive:
        archive.extractall(source, filter="data")
    store = DataStore(str(source), default_assay="RNA")
    run = store.pipeline.rna(label="cells", doublets=False, umap=False, markers=False)
    rows = np.flatnonzero(run.cells.fetch_all("I"))
    return SimpleNamespace(
        counts=store.RNA.rawData[rows, :].compute(nthreads=1, msg="Reading counts"),
        labels=np.asarray(run.cells.fetch("clusters")),
        feature_ids=list(store.RNA.feats.fetch_all("ids")),
        feature_names=list(store.RNA.feats.fetch_all("names")),
    )


def _pbmc_store(
    cells: SimpleNamespace, counts: np.ndarray, location: Path
) -> DataStore:
    write_count_store(
        str(location),
        {"RNA": counts},
        "uint32",
        feature_ids={"RNA": cells.feature_ids},
        feature_names={"RNA": cells.feature_names},
    )
    return DataStore(str(location), min_features_per_cell=0)


@pytest.mark.slow  # A run of the 1k PBMC fixture and of a planted copy.
def test_rna_doublets_call_planted_doublets_where_the_minimum_called_none(
    pbmc_analysis_cells, tmp_path
):
    counts, labels = pbmc_analysis_cells.counts, pbmc_analysis_cells.labels
    # 60 doublets of two cells of different clusters: 30 hold both cells'
    # counts and 30 half of them, as many as a singlet holds.
    rng = np.random.default_rng(11)
    first = rng.choice(len(labels), 60, replace=False)
    second = [rng.choice(np.flatnonzero(labels != labels[cell])) for cell in first]
    summed = counts[first].astype(np.int64) + counts[second]
    planted = np.vstack([summed[:30], rng.binomial(summed[30:], 0.5)])
    truth = np.repeat([0, 1, 2], [len(labels), 30, 30])
    planted_store = _pbmc_store(
        pbmc_analysis_cells, np.vstack([counts, planted]), tmp_path / "planted.zarr"
    )

    planted_run = planted_store.pipeline.rna(
        label="planted", filtering=False, umap=False, markers=False
    )

    # One or two simulated doublets far above the rest form the second mode
    # of these scores, and the minimum before them called no cell. Otsu's
    # threshold calls planted doublets, and singlets at about the rate of
    # the fixture's own doublets and the doublets that look like them.
    (sample,) = _payload(planted_store, planted_run["doublets"]).attrs["samples"]
    assert sample["thresholdMethod"] == "otsu"
    assert sample["detectableDoubletFraction"] > 0.2
    assert sample["estimatedDoubletRate"] < 0.5
    calls = np.asarray(planted_run.cells.fetch("predicted_doublet"), dtype=bool)
    assert np.count_nonzero(calls[truth > 0]) >= 15
    assert np.mean(calls[truth == 0]) <= 0.05


@pytest.mark.slow  # A run of the 1k PBMC fixture and of one of its clusters.
def test_rna_doublets_of_one_cluster_call_no_cell_and_warn(
    pbmc_analysis_cells, tmp_path
):
    # The cells of the fixture's largest cluster, alone: their simulated
    # doublets look like the cells, and their scores have one mode.
    values, sizes = np.unique(pbmc_analysis_cells.labels, return_counts=True)
    cluster = pbmc_analysis_cells.labels == values[np.argmax(sizes)]
    store = _pbmc_store(
        pbmc_analysis_cells,
        pbmc_analysis_cells.counts[cluster],
        tmp_path / "cluster.zarr",
    )

    run = store.pipeline.rna(
        label="cluster", filtering=False, umap=False, markers=False
    )

    payload = _payload(store, run["doublets"])
    (sample,) = payload.attrs["samples"]
    simulated, scores = payload["simulated_scores"][:], payload["values"][:]
    # One or two simulated doublets above the rest form the second mode, and
    # the minimum before them leaves fewer than 5% of the doublets above it.
    # Otsu's threshold splits the one mode, cells and doublets alike: a
    # doublet rate above 50%, which no droplet experiment holds.
    minimum = doublets.histogram_minimum_threshold(simulated)
    assert minimum is not None and np.mean(simulated > minimum) < 0.05
    otsu = doublets.otsu_threshold(simulated)
    assert np.mean(scores > otsu) / np.mean(simulated > otsu) > 0.5
    assert np.mean(scores > otsu) > 0.2
    # The sample has no threshold and calls no cell, and the stage warns of
    # it and its summary shows it.
    assert (sample["thresholdMethod"], sample["threshold"]) == ("none", None)
    assert sample["detectableDoubletFraction"] == 0.0
    assert sample["predictedDoublets"] == 0
    assert not np.asarray(run.cells.fetch("predicted_doublet"), dtype=bool).any()
    (stage,) = [
        record
        for record in load_pipeline_stage_records(store.zw, run.run_id)
        if record.stage == "doublets"
    ]
    assert stage.warnings == (f"In 1 sample {_NO_THRESHOLD}. {_SET_THRESHOLD}",)
    assert stage.summary is not None
    assert stage.summary["samples"] == [
        {
            "labels": {},
            "cells": sample["cells"],
            "threshold": None,
            "thresholdMethod": "none",
            "predictedDoublets": 0,
        }
    ]


@pytest.mark.slow  # A run of the 1k PBMC fixture and of one of its clusters.
def test_rna_doublets_of_one_cluster_beside_a_low_outlier_call_no_cell(
    pbmc_analysis_cells, tmp_path
):
    # The cells of the fixture's second largest cluster, alone: their
    # simulated doublets look like the cells, and one scores far below the
    # rest.
    values, sizes = np.unique(pbmc_analysis_cells.labels, return_counts=True)
    second = values[np.argsort(-sizes, kind="stable")[1]]
    store = _pbmc_store(
        pbmc_analysis_cells,
        pbmc_analysis_cells.counts[pbmc_analysis_cells.labels == second],
        tmp_path / "cluster.zarr",
    )

    run = store.pipeline.rna(
        label="cluster", filtering=False, umap=False, markers=False
    )

    payload = _payload(store, run["doublets"])
    (sample,) = payload.attrs["samples"]
    simulated, scores = payload["simulated_scores"][:], payload["values"][:]
    # The outlier keeps a maximum of its own, and the minimum above it leaves
    # fewer than 5% of the simulated doublets at or below it and nearly every
    # cell above it. Otsu's threshold splits the one mode, cells and doublets
    # alike, so master's rule kept the minimum and called those cells.
    minimum = doublets.histogram_minimum_threshold(simulated)
    assert minimum is not None and np.mean(simulated <= minimum) < 0.05
    assert np.mean(scores > minimum) > 0.95
    otsu = doublets.otsu_threshold(simulated)
    assert np.mean(scores > otsu) / np.mean(simulated > otsu) > 0.5
    # The sample has no threshold and calls no cell, and the stage warns.
    assert (sample["thresholdMethod"], sample["threshold"]) == ("none", None)
    assert sample["predictedDoublets"] == 0
    assert not np.asarray(run.cells.fetch("predicted_doublet"), dtype=bool).any()
    (stage,) = [
        record
        for record in load_pipeline_stage_records(store.zw, run.run_id)
        if record.stage == "doublets"
    ]
    assert stage.warnings == (f"In 1 sample {_NO_THRESHOLD}. {_SET_THRESHOLD}",)
