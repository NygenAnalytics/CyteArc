"""HVG flavors, batch-aware selection, detection bounds, and Pearson residuals.

The seurat_v3 and pearson_residuals flavors are checked against direct NumPy
implementations of their published formulas and against outputs of scanpy
1.12.4 with scikit-misc 0.5.3 stored in ``scanpy_hvg_1_12_4_golden.json``
(``scripts/export_scanpy_hvg_golden.py`` regenerates it). Batch-aware
selection follows the ordering of Seurat's ``SelectIntegrationFeatures``.
"""

import hashlib
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from cytearc.assay import ATACassay, PearsonResidualNormalizer, norm_pearson_residuals
from cytearc.assay.base import Assay
from cytearc.assay.hvg_statistics import (
    grouped_clipped_squares,
    grouped_moments,
    grouped_pearson_residual_moments,
)
from cytearc.assay.normalization import applicable_normalization_flags
from cytearc.datastore.datastore import DataStore
from cytearc.features.variability import (
    DEFAULT_HVG_BLACKLIST,
    _fit_loess_points,
    _loess_vertices,
    combine_batch_rankings,
    default_hvg_min_cells,
    default_hvg_ubiquity_limit,
    fit_loess,
    fit_lowess,
    hvg_flavor_options,
    hvg_options,
    pearson_residual_clip,
    pearson_residuals,
    resolve_hvg_detection_bounds,
    select_highly_variable_features,
    seurat_v3_clip_values,
    seurat_v3_expected_variance,
    seurat_v3_standardized_variance,
)
from cytearc.storage.artifacts import artifact_path, inspect_artifact
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.operation_revisions import OPERATION_REVISIONS
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.utils.logging import logger
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts, insert_nullable_cell_column

_GOLDEN = Path(__file__).parent / "scanpy_hvg_1_12_4_golden.json"


def _warnings(run: Any) -> list[str]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        run()
    finally:
        logger.remove(sink)
    return messages


# Local regression


def _reference_loess(x: np.ndarray, y: np.ndarray, span: float, degree: int):
    """Weighted least squares at every point, written from Cleveland (1979)."""
    q = min(len(x), max(math.floor(span * len(x)), degree + 1))
    fitted = []
    for x0 in x:
        distance = np.abs(x - x0)
        h = np.sort(distance)[q - 1]
        weight = np.where(distance < h, (1 - (distance / h) ** 3) ** 3, 0.0)
        design = np.vander(x - x0, degree + 1, increasing=True)
        coefficients = np.linalg.lstsq(
            design * np.sqrt(weight)[:, None], y * np.sqrt(weight), rcond=None
        )[0]
        fitted.append(coefficients[0])
    return np.asarray(fitted)


@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("span", [0.3, 0.75, 1.0])
def test_direct_loess_matches_weighted_least_squares(span, degree) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=150)
    y = np.sin(2 * x) + rng.normal(0, 0.2, size=150)

    np.testing.assert_allclose(
        fit_loess(x, y, span=span, degree=degree, surface="direct"),
        _reference_loess(x, y, span, degree),
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.parametrize("surface", ["direct", "interpolate"])
def test_loess_reproduces_a_quadratic_trend(surface) -> None:
    x = np.random.default_rng(5).uniform(-2, 2, 300)
    y = 3 * x**2 - x + 1

    np.testing.assert_allclose(fit_loess(x, y, surface=surface), y, atol=1e-9)


def test_interpolated_loess_blends_vertex_fits_with_cubic_hermite_polynomials() -> None:
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0, 10, 200))
    y = np.cos(x) + rng.normal(0, 0.1, 200)
    vertices = _loess_vertices(x, math.floor(200 * 0.3 * 0.2))
    values = np.empty(len(vertices))
    slopes = np.empty(len(vertices))
    _fit_loess_points(x, y, 60, 2, vertices, values, slopes)
    # Each vertex carries the value and slope of the weighted least-squares
    # quadratic at its position.
    for vertex, value, slope in zip(vertices, values, slopes, strict=True):
        distance = np.abs(x - vertex)
        h = np.sort(distance)[59]
        weight = np.sqrt(np.where(distance < h, (1 - (distance / h) ** 3) ** 3, 0.0))
        design = np.vander(x - vertex, 3, increasing=True)
        coefficients = np.linalg.lstsq(
            design * weight[:, None], y * weight, rcond=None
        )[0]
        assert (value, slope) == pytest.approx(tuple(coefficients[:2]), rel=1e-9)
    cell = np.searchsorted(vertices, x[37], side="right") - 1
    width = vertices[cell + 1] - vertices[cell]
    t = (x[37] - vertices[cell]) / width
    expected = (
        (1 - t) ** 2 * (1 + 2 * t) * values[cell]
        + t**2 * (3 - 2 * t) * values[cell + 1]
        + (t * (1 - t) ** 2 * slopes[cell] - t**2 * (1 - t) * slopes[cell + 1]) * width
    )

    assert fit_loess(x, y)[37] == pytest.approx(expected, rel=1e-12)
    # The root cell spans the data widened by 0.5% of their range, and every
    # leaf holds at most 12 points; a point at a split vertex is in the lower
    # cell.
    assert vertices[0] == pytest.approx(x[0] - 0.005 * (x[-1] - x[0]), rel=1e-12)
    assert vertices[-1] == pytest.approx(x[-1] + 0.005 * (x[-1] - x[0]), rel=1e-12)
    cells = np.searchsorted(vertices, x, side="left") - 1
    assert np.bincount(cells).max() <= 12


def test_loess_vertices_never_split_tied_values() -> None:
    x = np.array([1.0, 1, 1, 1, 1, 1, 2, 3])

    # The median sits inside six ties, so the split moves to the last tie;
    # the cell of ties cannot be split again.
    np.testing.assert_array_equal(_loess_vertices(x, 2), [0.99, 1.0, 3.01])


def test_loess_handles_ties_tiny_inputs_and_empty_windows() -> None:
    np.testing.assert_array_equal(fit_loess(np.empty(0), np.empty(0)), np.empty(0))
    assert fit_loess(np.array([2.0]), np.array([5.0]))[0] == pytest.approx(
        5.0, rel=1e-12
    )
    # More ties than the window holds: the fit is the mean of the ties.
    x = np.r_[np.zeros(8), np.arange(1.0, 3.0)]
    y = np.r_[np.arange(8.0), 10.0, 20.0]
    fitted = fit_loess(x, y, span=0.3, surface="direct")
    np.testing.assert_allclose(fitted[:8], 3.5)
    # Ties after other points: the window slides to the first tie, and the
    # fit is the mean of every tied point.
    x = np.r_[-2.0, -1.0, np.zeros(6), 1.0]
    y = np.r_[5.0, 6.0, np.arange(6.0), 9.0]
    fitted = fit_loess(x, y, span=0.3, surface="direct")
    np.testing.assert_allclose(fitted[2:8], 2.5)
    # A point midway between the only two points of its window gives both
    # zero weight, so the fit is their mean.
    values = np.empty(1)
    slopes = np.empty(1)
    _fit_loess_points(
        np.array([0.0, 2.0]),
        np.array([1.0, 3.0]),
        2,
        2,
        np.array([1.0]),
        values,
        slopes,
    )
    assert (values[0], slopes[0]) == (2.0, 0.0)
    # Two distinct values cannot carry a quadratic, so the fit is linear.
    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([1.0, 1.0, 3.0, 3.0])
    np.testing.assert_allclose(fit_loess(x, y, span=1.0, surface="direct"), y)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"x": np.ones((2, 2)), "y": np.ones((2, 2))}, ValueError, "one-dimensional"),
        ({"x": np.ones(3), "y": np.ones(2)}, ValueError, "one-dimensional"),
        ({"x": np.array([1.0, np.nan]), "y": np.ones(2)}, ValueError, "finite"),
        ({"span": 0}, ValueError, "span must be greater than 0"),
        ({"span": 1.5}, ValueError, "span must be greater than 0"),
        ({"span": "0.3"}, TypeError, "span must be a real number"),
        ({"degree": 3}, ValueError, "degree must be at most 2"),
        ({"surface": "kd"}, ValueError, "surface must be"),
    ],
)
def test_loess_rejects_invalid_arguments(arguments, error, message) -> None:
    options = {"x": np.arange(5.0), "y": np.arange(5.0), **arguments}
    with pytest.raises(error, match=message):
        fit_loess(options.pop("x"), options.pop("y"), **options)


# seurat_v3 and Pearson residual formulas


def test_seurat_v3_standardized_variance_matches_clipped_z_scores() -> None:
    rng = np.random.default_rng(17)
    success = np.linspace(0.05, 0.9, 40)
    counts = rng.negative_binomial(2, success, size=(60, 40)).astype(np.float64)
    counts[:, 0] = 0
    counts[:, 1] = 2
    counts[7, 20] = 300
    n = len(counts)
    mean = counts.mean(axis=0)
    variance = counts.var(axis=0, ddof=1)

    expected = seurat_v3_expected_variance(mean, variance, span=0.3)
    clips = seurat_v3_clip_values(mean, expected, n)
    squares = ((np.minimum(counts, clips) - mean) ** 2).sum(axis=0)
    standardized = seurat_v3_standardized_variance(squares, expected, n)

    varying = variance > 0
    np.testing.assert_allclose(
        expected[varying],
        10 ** fit_loess(np.log10(mean[varying]), np.log10(variance[varying])),
    )
    assert expected[0] == expected[1] == 0
    # Seurat standardizes counts, clips the z-scores at sqrt(n), and divides
    # their sum of squares around 0 by n - 1.
    sigma = np.sqrt(expected[varying])
    z = np.minimum((counts[:, varying] - mean[varying]) / sigma, np.sqrt(n))
    np.testing.assert_allclose(
        standardized[varying], (z**2).sum(axis=0) / (n - 1), rtol=1e-12
    )
    assert standardized[0] == standardized[1] == 0
    # Only the outlier of gene 20 lies above its clip.
    np.testing.assert_array_equal(np.argwhere(counts > clips), [[7, 20]])
    with pytest.raises(ValueError, match="at least two cells"):
        seurat_v3_standardized_variance(squares, expected, 1)
    with pytest.raises(ValueError, match="positive mean"):
        seurat_v3_expected_variance(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    # Without a varying feature there is no trend to fit.
    np.testing.assert_array_equal(
        seurat_v3_expected_variance(np.array([0.0, 2.0]), np.zeros(2)), np.zeros(2)
    )


def test_pearson_residuals_match_a_hand_computed_example() -> None:
    counts = np.array([[0, 2], [3, 1], [0, 0]])
    totals = counts.sum(axis=1)
    fractions = counts.sum(axis=0) / totals.sum()
    # Expected counts are [[1, 1], [2, 2], [0, 0]] for theta=100.
    first, second = 1 / np.sqrt(1.01), 1 / np.sqrt(2.04)

    np.testing.assert_allclose(
        pearson_residuals(counts, totals, fractions, theta=100.0, clip=10.0),
        [[-first, first], [second, -second], [0.0, 0.0]],
    )
    np.testing.assert_allclose(
        pearson_residuals(counts, totals, fractions, theta=100.0, clip=0.8),
        [[-0.8, 0.8], [second, -second], [0.0, 0.0]],
    )
    assert pearson_residual_clip(None, 49) == 7.0
    assert pearson_residual_clip(2, 49) == 2.0


# Detection bounds


@pytest.mark.parametrize(
    ("n_selected", "min_cells", "limit"),
    [
        (1, 1, 0),
        (2, 1, 1),
        (4, 1, 3),
        (10, 2, 9),
        (15, 3, 14),
        (41, 3, 40),
        (150, 3, 149),
        (300, 3, 298),
        (301, 4, 299),
        (892, 9, 887),
        (1000, 10, 995),
        (1901, 20, 1891),
        (3696, 20, 3677),
        (4000, 20, 3980),
        (100_000, 20, 99_500),
    ],
)
def test_default_detection_bounds_scale_with_the_selection(
    n_selected, min_cells, limit
) -> None:
    assert default_hvg_min_cells(n_selected) == min_cells
    assert default_hvg_ubiquity_limit(n_selected) == limit
    # A selection of two or more cells keeps a band of detection counts.
    assert n_selected < 2 or min_cells <= limit
    assert resolve_hvg_detection_bounds(
        None, None, n_selected=n_selected, keep_bounds=False
    ) == (min_cells, limit + 1)
    assert resolve_hvg_detection_bounds(
        None, None, n_selected=n_selected, keep_bounds=True
    ) == (min_cells, limit)


def test_detection_bounds_of_other_flavors_and_explicit_values() -> None:
    for flavor in ("seurat_v3", "pearson_residuals"):
        assert resolve_hvg_detection_bounds(
            None, None, n_selected=500, keep_bounds=False, flavor=flavor
        ) == (0, np.inf)
    assert resolve_hvg_detection_bounds(
        7, np.inf, n_selected=500, keep_bounds=False
    ) == (7, np.inf)
    assert resolve_hvg_detection_bounds(
        0, 250.9, n_selected=500, keep_bounds=True, flavor="seurat_v3"
    ) == (0, 250)


# Candidate filters and ranking


def _select(**overrides: Any) -> np.ndarray:
    options: dict[str, Any] = dict(
        corrected_variance=np.array([10.0, 8.0, 6.0, 4.0, 2.0]),
        normalized_cell_counts=np.full(5, 5),
        mean_nonzero=np.ones(5),
        active_features=np.ones(5, dtype=bool),
        feature_names=np.array(["a", "b", "c", "d", "e"]),
        min_cells=0,
        max_cells=np.inf,
        top_n=2,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        blacklist="",
        keep_bounds=False,
    )
    options.update(overrides)
    return np.asarray(select_highly_variable_features(**options))


def test_variance_bounds_filter_candidates_before_top_n() -> None:
    # min_var is the log2 of the bound: 2**2 = 4, and top_n still applies.
    np.testing.assert_array_equal(_select(min_var=2), [True, True, False, False, False])
    # max_var removes genes before ranking, so top_n genes remain.
    np.testing.assert_array_equal(
        _select(max_var=math.log2(7)), [False, False, True, True, False]
    )
    np.testing.assert_array_equal(
        _select(max_var=math.log2(6), keep_bounds=True),
        [False, False, True, True, False],
    )
    # top_n=10 with max_var=2 keeps every gene below 2**2 = 4.
    many = np.arange(20, 0, -1, dtype=np.float64) / 4
    selected = _select(
        corrected_variance=many,
        normalized_cell_counts=np.full(20, 5),
        mean_nonzero=np.ones(20),
        active_features=np.ones(20, dtype=bool),
        feature_names=np.array([f"g{index}" for index in range(20)]),
        top_n=10,
        max_var=2,
    )
    np.testing.assert_array_equal(np.flatnonzero(selected), np.arange(5, 15))


def test_selection_with_no_candidate_names_every_filter() -> None:
    with pytest.raises(ValueError) as caught:
        _select(min_var=10, max_mean=1)
    message = str(caught.value)
    assert message.startswith("HVG selection contains no features. No features passed")
    assert "min_var=10" in message and "max_mean=1" in message


def test_selection_warns_when_fewer_candidates_than_top_n_pass() -> None:
    selections: list[np.ndarray] = []
    messages = _warnings(lambda: selections.append(_select(top_n=10, min_var=2)))
    assert messages == [
        "Only 3 features passed the HVG candidate filters, fewer than top_n=10; "
        "all of them are selected"
    ]
    # The variances 10, 8, and 6 lie above 2**2 = 4.
    np.testing.assert_array_equal(selections, [[True, True, True, False, False]])


def test_nan_scores_never_pass_the_candidate_filters() -> None:
    selected = _select(corrected_variance=np.array([np.nan, 8.0, 6.0, 4.0, 2.0]))
    np.testing.assert_array_equal(selected, [False, True, True, False, False])


def test_batches_combine_by_count_then_median_rank() -> None:
    scores = np.array(
        [
            [5.0, 4, 3, 2, 6, 1],
            [1.0, 6, 5, 4, 3, 2],
            [5.0, 4, 1, 6, 2, 3],
        ]
    )
    ranking = combine_batch_rankings(scores, np.ones(scores.shape, dtype=bool), top_n=3)

    np.testing.assert_array_equal(ranking.batch_count, [2, 3, 1, 2, 1, 0])
    np.testing.assert_array_equal(ranking.median_rank, [1, 2, 1, 1, 0, np.nan])
    # Seurat's order: gene 1 is highly variable in three batches, then genes
    # 0 and 3 in two, tied at median rank 1. Sorting by median rank first, as
    # scanpy's seurat_v3 does, would pick gene 4, ranked first in one batch.
    np.testing.assert_array_equal(
        ranking.selected, [True, True, False, True, False, False]
    )


def test_one_batch_selects_the_top_scores() -> None:
    scores = np.array([[3.0, 7.0, 7.0, 1.0, 5.0]])
    candidates = np.array([[True, True, True, True, False]])

    ranking = combine_batch_rankings(scores, candidates, top_n=2)

    np.testing.assert_array_equal(ranking.selected, [False, True, True, False, False])
    np.testing.assert_array_equal(ranking.median_rank, [np.nan, 0, 1, np.nan, np.nan])


def test_batch_ranking_warns_and_fails_loudly() -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidates = np.array([[True, False, False], [True, True, True]])
    messages = _warnings(
        lambda: combine_batch_rankings(
            scores, candidates, top_n=2, batch_names=["left", "right"]
        )
    )
    assert messages == [
        "Batch 'left' has 1 HVG candidates, fewer than top_n=2; all of them count "
        "as highly variable in it"
    ]

    rankings = []
    messages = _warnings(
        lambda: rankings.append(
            combine_batch_rankings(
                scores, np.array([[True, False, False]] * 2), top_n=2
            )
        )
    )
    # Unnamed batches are named by their position.
    assert messages == [
        *(
            f"Batch '{batch}' has 1 HVG candidates, fewer than top_n=2; all of them "
            "count as highly variable in it"
            for batch in (0, 1)
        ),
        "Only 1 features are highly variable in some batch, fewer than top_n=2; "
        "all of them are selected",
    ]
    np.testing.assert_array_equal(rankings[0].selected, [True, False, False])
    with pytest.raises(ValueError, match="in any batch"):
        combine_batch_rankings(scores, np.zeros(scores.shape, dtype=bool), top_n=2)


def test_one_batch_ranking_warns_like_a_selection_without_batches() -> None:
    # A flavor without batch ranks one batch, which a message never names.
    messages = _warnings(
        lambda: combine_batch_rankings(
            np.array([[3.0, 2.0, 1.0]]),
            np.array([[True, False, False]]),
            top_n=2,
            batch_names=["all"],
        )
    )

    assert messages == [
        "Only 1 features passed the HVG candidate filters, fewer than top_n=2; "
        "all of them are selected"
    ]


@pytest.mark.parametrize(("n_batches", "where"), [(1, ""), (2, " in any batch")])
def test_batch_selection_without_candidates_names_every_filter(
    n_batches, where
) -> None:
    from cytearc.features.variability import select_batch_highly_variable_features

    with pytest.raises(ValueError) as caught:
        select_batch_highly_variable_features(
            np.ones((n_batches, 3)),
            np.ones((n_batches, 3)),
            np.array([5, 5, 5]),
            np.array(["a", "b", "c"]),
            min_cells=6,
            max_cells=np.inf,
            top_n=2,
            min_var=-np.inf,
            max_var=np.inf,
            min_mean=-np.inf,
            max_mean=np.inf,
            blacklist="",
            keep_bounds=False,
        )

    assert str(caught.value) == (
        f"HVG selection contains no features. No features passed HVG candidate "
        f"filters{where} (min_cells=6, max_cells=inf, log2 bounds min_mean=-inf, "
        "max_mean=inf, min_var=-inf, max_var=inf). Widen the bounds or change the "
        "blacklist."
    )


# Options


def test_hvg_options_take_a_flavor_default_min_cells() -> None:
    options = hvg_options(
        min_cells=None,
        top_n=5,
        n_bins=20,
        lowess_frac=0.2,
        keep_bounds=False,
        bin_strategy="fixed",
    )
    # None is kept for the flavor to resolve, and the other options as given.
    assert options == (None, 5, 20, 0.2, False, "fixed")


_FLAVOR_DEFAULTS: dict[str, Any] = {
    "flavor": "cytearc",
    "max_cells": None,
    "min_mean": -np.inf,
    "max_mean": np.inf,
    "min_var": -np.inf,
    "max_var": np.inf,
    "n_bins": 200,
    "lowess_frac": 0.1,
    "bin_strategy": "adaptive",
    "span": 0.3,
    "theta": 100.0,
    "clip": None,
}


def test_flavor_options_are_checked_and_canonical() -> None:
    options = hvg_flavor_options(
        **{
            **_FLAVOR_DEFAULTS,
            "flavor": "pearson_residuals",
            "max_cells": np.int64(40),
            "theta": 10,
            "clip": np.float32(4),
        }
    )
    assert options == ("pearson_residuals", 40.0, 0.3, 10.0, 4.0)
    assert [type(value) for value in options[1:]] == [float] * 4
    assert (
        hvg_flavor_options(
            **{**_FLAVOR_DEFAULTS, "flavor": "seurat_v3", "span": 1}
        ).span
        == 1.0
    )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"flavor": "seurat"}, ValueError, "flavor must be one of"),
        ({"flavor": None}, ValueError, "flavor must be one of"),
        ({"max_cells": "10"}, TypeError, "max_cells must be a number"),
        ({"max_cells": True}, TypeError, "max_cells must be a number"),
        ({"max_cells": -1}, ValueError, "max_cells must be at least 0"),
        ({"max_cells": np.nan}, ValueError, "max_cells must be at least 0"),
        ({"min_mean": "low"}, TypeError, "min_mean must be a real number"),
        ({"max_var": np.nan}, ValueError, "max_var must not be NaN"),
        ({"span": 0}, ValueError, "span must be greater than 0"),
        ({"theta": 0}, ValueError, "theta must be greater than 0"),
        ({"theta": np.inf}, ValueError, "theta must be finite"),
        ({"clip": -1}, ValueError, "clip must be at least 0"),
        ({"flavor": "seurat_v3", "min_var": 1}, ValueError, "min_var apply only"),
        (
            {"flavor": "pearson_residuals", "n_bins": 20, "bin_strategy": "fixed"},
            ValueError,
            "n_bins, bin_strategy apply only to flavor='cytearc'",
        ),
        ({"span": 0.5}, ValueError, "span applies only"),
        ({"flavor": "seurat_v3", "theta": 5}, ValueError, "theta and clip apply only"),
        ({"clip": 3}, ValueError, "theta and clip apply only"),
    ],
)
def test_flavor_options_reject_invalid_and_foreign_options(
    overrides, error, message
) -> None:
    with pytest.raises(error, match=message):
        hvg_flavor_options(**{**_FLAVOR_DEFAULTS, **overrides})


# Gene families


def test_default_blacklist_spares_ccn_proteins_and_matches_current_histones() -> None:
    from cytearc.utils.arrays import regex_match_mask

    names = np.array(
        [
            "CCN1",
            "CCN2",
            "Ccn6",
            "CCND1",
            "Ccnb1",
            "CCNT1",
            "H2AC6",
            "H2BC12",
            "H3C2",
            "H4C3",
            "H1-2",
            "H1f4",
            "H2AW",
            "H3-4",
            "H4-16",
            "HIST1H1C",
            "H3-3A",
            "H2AZ1",
            "H1-0",
            "H2BC12L",
            "H2AX",
        ]
    )
    excluded = regex_match_mask(names, DEFAULT_HVG_BLACKLIST)

    assert set(names[excluded]) == {
        "CCND1",
        "Ccnb1",
        "CCNT1",
        "H2AC6",
        "H2BC12",
        "H3C2",
        "H4C3",
        "H1-2",
        "H1f4",
        "H2AW",
        "H3-4",
        "H4-16",
        "HIST1H1C",
    }


# Streamed per-batch statistics over a store with several cell bands

_N_CELLS, _N_GENES = 90, 48
_BATCHES = np.array(["x", "y", "z"], dtype=object)[np.arange(_N_CELLS) % 3]
# The batch code of each cell, in the sorted order of the batch names.
_CODES = np.arange(_N_CELLS) % 3


def _banded_counts() -> np.ndarray:
    rng = np.random.default_rng(23)
    success = np.linspace(0.05, 0.9, _N_GENES)
    counts = rng.negative_binomial(2, success, size=(_N_CELLS, _N_GENES))
    # Never detected, and constant in every cell.
    counts[:, 0] = 0
    counts[:, 1] = 3
    # An outlier above its seurat_v3 clip.
    counts[13, 5] = 600
    # A bimodal gene detected in 88 of 90 cells: the earlier default cap of
    # 90 - 20 = 70 cells excluded it, the 99.5% cap keeps it.
    counts[:, 7] = np.where(np.arange(_N_CELLS) % 2 == 0, 1, 40)
    counts[[10, 50], 7] = 0
    # Detected only in batch x, so the other batches expect no counts.
    counts[:, 9] = np.where(_CODES == 0, counts[:, 9] + 1, 0)
    return counts


_COUNTS = _banded_counts()
_NAMES = np.array([f"G{index}" for index in range(_N_GENES)], dtype=object)
_NAMES[3] = "MT-CO1"


@pytest.fixture(scope="module")
def banded_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("hvg_flavors") / "store.zarr"
    root = zarr.open_group(str(path), mode="w")
    ids = np.array([f"c{index}" for index in range(_N_CELLS)])
    create_cell_data(root, None, ids=ids, names=ids)
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        _N_CELLS,
        feat_ids=np.array([f"id{index}" for index in range(_N_GENES)]),
        feat_names=_NAMES,
        dtype="uint16",
        policy=CountMatrixPolicy(unitBytes=2000, chunkBytes=400),
    )
    counts[:] = _COUNTS
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "RNA", None)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=2
    )
    store.cells.insert("batch", _BATCHES, overwrite=True)
    return path


@pytest.fixture
def banded(banded_template, tmp_path) -> DataStore:
    shutil.copytree(banded_template, tmp_path / "store.zarr")
    return DataStore(
        str(tmp_path / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=2,
    )


def _library_size(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    return 1000 * counts / np.where(totals == 0, 1, totals)


def _batch_fractions(counts: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            counts[_CODES == code].sum(axis=0) / counts[_CODES == code].sum()
            for code in range(3)
        ]
    )


def test_grouped_statistics_match_dense_numpy(banded) -> None:
    rna = banded.RNA
    # Twenty cells per countsT band, so every batch spans several bands.
    assert rna.rawDataT.chunks[1] == 20
    cells = np.arange(_N_CELLS)
    features = np.arange(_N_GENES)
    counts = _COUNTS.astype(np.float64)
    for library_size in (False, True):
        values = _library_size(counts) if library_size else counts
        moments = grouped_moments(
            rna, cells, _CODES, 3, features, library_size=library_size
        )
        np.testing.assert_array_equal(moments.sizes, [30, 30, 30])
        for code in range(3):
            block = values[_CODES == code]
            np.testing.assert_allclose(
                moments.total[code], block.sum(axis=0), rtol=1e-13
            )
            np.testing.assert_array_equal(
                moments.detected[code], (block > 0).sum(axis=0)
            )
            np.testing.assert_allclose(
                moments.m2[code],
                ((block - block.mean(axis=0)) ** 2).sum(axis=0),
                rtol=1e-11,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                moments.maximum[code], block.max(axis=0), rtol=1e-15
            )

    subset = np.array([2, 5, 7, 9, 31])
    means = np.stack(
        [counts[_CODES == code][:, subset].mean(axis=0) for code in range(3)]
    )
    clips = means + np.array([[1.0], [2.0], [3.0]])
    squares = grouped_clipped_squares(
        rna, cells, _CODES, 3, subset, means=means, clips=clips
    )
    fractions = _batch_fractions(counts)
    totals = counts.sum(axis=1)
    residual_clips = np.array([2.0, 3.0, 4.0])
    total, m2 = grouped_pearson_residual_moments(
        rna,
        cells,
        _CODES,
        3,
        subset,
        totals=totals,
        fractions=fractions[:, subset],
        theta=50.0,
        clips=residual_clips,
    )
    for code in range(3):
        block = counts[_CODES == code][:, subset]
        np.testing.assert_allclose(
            squares[code],
            ((np.minimum(block, clips[code]) - means[code]) ** 2).sum(axis=0),
            rtol=1e-12,
        )
        residuals = pearson_residuals(
            block,
            totals[_CODES == code],
            fractions[code, subset],
            theta=50.0,
            clip=residual_clips[code],
        )
        np.testing.assert_allclose(total[code], residuals.sum(axis=0), atol=1e-12)
        np.testing.assert_allclose(
            m2[code],
            ((residuals - residuals.mean(axis=0)) ** 2).sum(axis=0),
            rtol=1e-12,
        )
    # Batches y and z expect no counts of gene 9, so its residuals are 0.
    assert total[1, 3] == total[2, 3] == m2[1, 3] == m2[2, 3] == 0


def test_grouped_statistics_check_their_batches(banded) -> None:
    rna = banded.RNA
    empty = grouped_moments(
        rna, np.arange(4), np.zeros(4), 1, np.empty(0), library_size=False
    )
    assert empty.total.shape == (1, 0)
    # Statistics of no features read no bands.
    no_features = np.empty(0, dtype=np.int64)
    squares = grouped_clipped_squares(
        rna,
        np.arange(4),
        np.zeros(4),
        1,
        no_features,
        means=np.empty((1, 0)),
        clips=np.empty((1, 0)),
    )
    total, m2 = grouped_pearson_residual_moments(
        rna,
        np.arange(4),
        np.zeros(4),
        1,
        no_features,
        totals=np.ones(4),
        fractions=np.empty((1, 0)),
        theta=100.0,
        clips=np.ones(1),
    )
    assert squares.shape == total.shape == m2.shape == (1, 0)
    with pytest.raises(ValueError, match="one batch code per selected cell"):
        grouped_moments(
            rna, np.arange(4), np.zeros(3), 1, np.arange(2), library_size=False
        )
    with pytest.raises(ValueError, match="Batch codes must lie from 0 to 1"):
        grouped_moments(
            rna, np.arange(4), np.full(4, 2), 2, np.arange(2), library_size=False
        )
    with pytest.raises(ValueError, match="at least two selected cells"):
        rna._batch_hvg_scores(
            np.arange(3),
            np.array([0, 0, 1]),
            2,
            flavor="seurat_v3",
            n_bins=200,
            lowess_frac=0.1,
            bin_strategy="adaptive",
            span=0.3,
            theta=100.0,
            clip=None,
        )


# The flavors as features.hvgs computes them


def _reference_scores(flavor: str, counts: np.ndarray, **options: Any) -> np.ndarray:
    """Score genes over the rows of ``counts`` from the published formulas."""
    counts = counts.astype(np.float64)
    n = len(counts)
    if flavor == "cytearc":
        values = _library_size(counts)
        average = values.mean(axis=0)
        positive = average > 0
        scores = np.zeros(counts.shape[1])
        scores[positive] = fit_lowess(
            average[positive], values.var(axis=0)[positive], 200, 0.1
        )
        return scores
    if flavor == "seurat_v3":
        mean = counts.mean(axis=0)
        variance = counts.var(axis=0, ddof=1)
        varying = variance > 0
        expected = 10 ** fit_loess(np.log10(mean[varying]), np.log10(variance[varying]))
        z = np.minimum(
            (counts[:, varying] - mean[varying]) / np.sqrt(expected), np.sqrt(n)
        )
        scores = np.zeros(counts.shape[1])
        scores[varying] = (z**2).sum(axis=0) / (n - 1)
        return scores
    totals = counts.sum(axis=1)
    residuals = pearson_residuals(
        counts,
        totals,
        counts.sum(axis=0) / totals.sum(),
        theta=options.get("theta", 100.0),
        clip=options.get("clip") or np.sqrt(n),
    )
    return residuals.var(axis=0)


def _top(scores: np.ndarray, candidates: np.ndarray, top_n: int) -> np.ndarray:
    index = np.flatnonzero(candidates)
    selected = np.zeros(len(scores), dtype=bool)
    selected[index[np.lexsort((index, -scores[index]))][:top_n]] = True
    return selected


@pytest.mark.parametrize(
    ("flavor", "options", "flavor_parameters"),
    [
        ("seurat_v3", {}, {"flavor": "seurat_v3", "span": 0.3}),
        (
            "pearson_residuals",
            {},
            {"flavor": "pearson_residuals", "theta": 100.0, "clip": None},
        ),
        (
            "pearson_residuals",
            {"theta": 20, "clip": 2},
            {"flavor": "pearson_residuals", "theta": 20.0, "clip": 2.0},
        ),
    ],
)
def test_count_flavors_match_their_formulas(
    banded, flavor, options, flavor_parameters
) -> None:
    cells = banded.snapshot_cell_selection()
    ref = banded.features.hvgs(
        cells, flavor=flavor, top_n=10, blacklist="", show_plot=False, **options
    )

    status = inspect_artifact(banded.zw, ref)
    group = banded.artifacts.load(ref)
    scores = _reference_scores(flavor, _COUNTS, **options)
    np.testing.assert_allclose(group["corrected_variance"][:], scores, rtol=1e-10)
    np.testing.assert_array_equal(
        group["values"][:], _top(scores, np.ones(_N_GENES, dtype=bool), 10)
    )
    detected = (_COUNTS > 0).sum(axis=0)
    np.testing.assert_array_equal(group["detected"][:], detected)
    np.testing.assert_allclose(
        group["mean_nonzero"][:],
        np.divide(
            _COUNTS.sum(axis=0), detected, out=np.zeros(_N_GENES), where=detected > 0
        ),
    )
    assert status.parameters == {
        **flavor_parameters,
        "min_cells": 0,
        "max_cells": {"special_float": "inf"},
        "top_n": 10,
        "blacklist": "",
        "keep_bounds": False,
    }
    assert set(status.inputs or {}) == {
        "cell_selection",
        "dataset_fingerprint",
        "feature_snapshot",
    }
    assert banded.features.resolve("RNA", ref) == ref


@pytest.mark.parametrize("flavor", ["cytearc", "seurat_v3", "pearson_residuals"])
def test_batches_select_by_count_then_median_rank(banded, flavor) -> None:
    cells = banded.snapshot_cell_selection()
    ref = banded.features.hvgs(
        cells, flavor=flavor, batch="batch", top_n=8, show_plot=False
    )

    scores = np.stack(
        [_reference_scores(flavor, _COUNTS[_CODES == code]) for code in range(3)]
    )
    detected = (_COUNTS > 0).sum(axis=0)
    # 90 cells: min_cells 3 and the 99.5% cap 89 for cytearc, none otherwise;
    # the default blacklist removes MT-CO1.
    lower, upper = (3, 89) if flavor == "cytearc" else (0, _N_CELLS)
    candidates = (detected >= lower) & (detected <= upper) & (_NAMES != "MT-CO1")
    expected = combine_batch_rankings(
        scores, np.broadcast_to(candidates, scores.shape), top_n=8
    )
    group = banded.artifacts.load(ref)
    np.testing.assert_array_equal(group["values"][:], expected.selected)
    np.testing.assert_array_equal(group["batch_count"][:], expected.batch_count)
    np.testing.assert_array_equal(group["median_batch_rank"][:], expected.median_rank)
    # The adaptive trend of the cytearc flavor is a numerical optimization, so
    # rounding differences of its inputs move its result by about 1e-8.
    np.testing.assert_allclose(
        group["corrected_variance"][:],
        scores.mean(axis=0),
        rtol=1e-7 if flavor == "cytearc" else 1e-10,
        atol=1e-12,
    )
    status = inspect_artifact(banded.zw, ref)
    assert status.parameters["batch_column"] == "batch"
    assert status.input_ref("batch").kind == "metadata_snapshot"
    assert banded.features.resolve("RNA", ref) == ref
    # The same request reuses the selection.
    assert (
        banded.features.hvgs(
            cells, flavor=flavor, batch="batch", top_n=8, show_plot=False
        )
        == ref
    )


def test_batch_labels_may_come_from_a_label_artifact(banded) -> None:
    cells = banded.snapshot_cell_selection()
    labels = banded.snapshot_cluster_labels("batch", cell_selection=cells)

    from_artifact = banded.features.hvgs(
        cells, flavor="seurat_v3", batch=labels, top_n=8, show_plot=False
    )
    from_column = banded.features.hvgs(
        cells, flavor="seurat_v3", batch="batch", top_n=8, show_plot=False
    )

    status = inspect_artifact(banded.zw, from_artifact)
    assert status.input_ref("batch") == labels
    assert "batch_column" not in (status.parameters or {})
    for name in ("values", "batch_count", "median_batch_rank", "corrected_variance"):
        np.testing.assert_array_equal(
            banded.artifacts.load(from_artifact)[name][:],
            banded.artifacts.load(from_column)[name][:],
        )
    assert banded.features.resolve("RNA", from_artifact) == from_artifact


def test_cytearc_batches_with_a_custom_normalizer_summarize_each_batch(
    banded,
) -> None:
    from cytearc.assay.normalization import norm_dummy

    banded.RNA.normMethod = norm_dummy
    cells = banded.snapshot_cell_selection()
    ref = banded.features.hvgs(cells, batch="batch", top_n=8, show_plot=False)

    raw = _COUNTS.astype(np.float64)
    scores = []
    for code in range(3):
        block = raw[_CODES == code]
        average = block.mean(axis=0)
        positive = average > 0
        score = np.zeros(_N_GENES)
        score[positive] = fit_lowess(
            average[positive], block.var(axis=0)[positive], 200, 0.1
        )
        scores.append(score)
    np.testing.assert_allclose(
        banded.artifacts.load(ref)["corrected_variance"][:],
        np.mean(scores, axis=0),
        # Streamed moments and NumPy's two-pass variance differ in the last
        # bits, which the local fits of some releases amplify to a few parts
        # per billion.
        rtol=1e-8,
    )


def _artifact_refs(store: DataStore) -> set[Any]:
    return set(store.artifacts.list()) | set(store.artifacts.list(scope="datastore"))


@pytest.mark.parametrize(
    ("batch", "error", "message"),
    [
        ("absent", KeyError, "Batch column 'absent' is not in the cell metadata"),
        (3, TypeError, "batch must be a cell metadata column name or an ArtifactRef"),
    ],
)
def test_invalid_batches_are_refused_before_writing(
    banded, batch, error, message
) -> None:
    cells = banded.snapshot_cell_selection()
    before = _artifact_refs(banded)
    with pytest.raises(error, match=message):
        banded.features.hvgs(cells, batch=batch, show_plot=False)
    with pytest.raises(ValueError, match="categorical cell labels"):
        banded.features.hvgs(cells, batch=cells, show_plot=False)
    assert _artifact_refs(banded) == before


def test_batches_need_labels_and_two_cells_each(banded) -> None:
    cells = banded.snapshot_cell_selection()
    missing = np.zeros(_N_CELLS, dtype=bool)
    missing[[4, 8]] = True
    insert_nullable_cell_column(banded, "partial", _BATCHES.astype(str), missing)
    with pytest.raises(ValueError, match="miss 2 selected cells"):
        banded.features.hvgs(cells, batch="partial", show_plot=False)

    singles = _BATCHES.copy()
    singles[0] = "lone"
    banded.cells.insert("singles", singles, overwrite=True)
    with pytest.raises(ValueError, match=r"these have fewer: 'lone' \(1\)"):
        banded.features.hvgs(
            cells, flavor="pearson_residuals", batch="singles", show_plot=False
        )


def test_count_flavors_need_two_selected_cells(banded) -> None:
    keep = np.zeros(_N_CELLS, dtype=bool)
    keep[5] = True
    banded.cells.insert("one", keep, overwrite=True)
    with pytest.raises(ValueError, match="needs at least two selected cells"):
        banded.features.hvgs(
            banded.snapshot_cell_selection("one"), flavor="seurat_v3", show_plot=False
        )


def test_foreign_flavor_options_are_refused_before_writing(banded) -> None:
    cells = banded.snapshot_cell_selection()
    before = _artifact_refs(banded)
    with pytest.raises(ValueError, match="min_var apply only to flavor='cytearc'"):
        banded.features.hvgs(cells, flavor="seurat_v3", min_var=1, show_plot=False)
    with pytest.raises(ValueError, match="span applies only"):
        banded.features.hvgs(cells, span=0.5, show_plot=False)
    assert _artifact_refs(banded) == before


def test_count_flavor_plots_its_stored_statistics_when_computed_and_reused(
    banded, monkeypatch
) -> None:
    import cytearc.plotting as plotting

    plotted: list[dict[str, Any]] = []

    def record(
        *,
        mean_nonzero: Any,
        corrected_variance: Any,
        n_cells: Any,
        selected: Any,
        label_size: float = 12,
        show: bool = True,
    ) -> None:
        plotted.append({key: value for key, value in locals().items()})

    monkeypatch.setattr(plotting, "highly_variable_features", record)
    cells = banded.snapshot_cell_selection()
    first = banded.features.hvgs(cells, flavor="seurat_v3", top_n=5, label_size=9)

    def no_scores(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("A reused selection must not score the genes again")

    monkeypatch.setattr(type(banded.RNA), "_batch_hvg_scores", no_scores)
    again = banded.features.hvgs(cells, flavor="seurat_v3", top_n=5, label_size=9)
    assert again == first

    group = banded.artifacts.load(first)
    assert len(plotted) == 2
    for call in plotted:
        assert call["show"] is True and call["label_size"] == 9
        for keyword, name in (
            ("mean_nonzero", "mean_nonzero"),
            ("corrected_variance", "corrected_variance"),
            ("n_cells", "detected"),
            ("selected", "values"),
        ):
            np.testing.assert_array_equal(call[keyword], group[name][:])


def _edit_provenance(store: DataStore, ref: Any, **changes: Any) -> None:
    group = zarr.open_group(store.zw.store, mode="r+", path=artifact_path(ref))
    provenance = dict(group.attrs["provenance"])
    for key, value in changes.items():
        section = {**provenance[key], **value}
        provenance[key] = {
            name: item for name, item in section.items() if item is not None
        }
    group.attrs["provenance"] = provenance


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"parameters": {"flavor": "seurat"}}, "HVG flavor is incompatible"),
        ({"parameters": {"flavor": "cytearc"}}, "HVG flavor is incompatible"),
        ({"parameters": {"batch_column": None}}, "records no batch column"),
        ({"inputs": {"dataset_fingerprint": "0" * 64}}, "current prepared dataset"),
    ],
)
def test_count_flavor_provenance_is_validated(banded, changes, message) -> None:
    cells = banded.snapshot_cell_selection()
    ref = banded.features.hvgs(
        cells, flavor="seurat_v3", batch="batch", top_n=5, show_plot=False
    )
    _edit_provenance(banded, ref, **changes)
    with pytest.raises(ArtifactResolutionError, match=message):
        banded.features.resolve("RNA", ref)


def test_batch_label_artifacts_must_be_complete_when_resolved(banded) -> None:
    cells = banded.snapshot_cell_selection()
    labels = banded.snapshot_cluster_labels("batch", cell_selection=cells)
    ref = banded.features.hvgs(
        cells, flavor="seurat_v3", batch=labels, top_n=5, show_plot=False
    )
    group = zarr.open_group(banded.zw.store, mode="r+", path=artifact_path(labels))
    group.attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError, match="batch labels are missing"):
        banded.features.resolve("RNA", ref)


# Detection bounds of the cytearc flavor


def test_default_cap_keeps_genes_of_homogeneous_selections(banded) -> None:
    cells = banded.snapshot_cell_selection()
    ref = banded.features.hvgs(cells, top_n=5, show_plot=False)

    parameters = inspect_artifact(banded.zw, ref).parameters
    assert (parameters["min_cells"], parameters["max_cells"]) == (3, 90)
    # Gene 7, detected in 88 of 90 cells, is among the most variable genes;
    # the earlier defaults excluded it.
    assert banded.artifacts.load(ref)["values"][:][7]
    earlier = banded.features.hvgs(
        cells, top_n=5, min_cells=20, max_cells=70, show_plot=False
    )
    assert not banded.artifacts.load(earlier)["values"][:][7]


def test_small_selections_keep_a_band_of_detection_counts(datastore_ephemeral) -> None:
    store = datastore_ephemeral
    active = np.flatnonzero(np.asarray(store.cells.fetch_all("I"), dtype=bool))
    keep = np.zeros(store.cells.N, dtype=bool)
    keep[active[:41]] = True
    store.cells.insert("small", keep, overwrite=True)
    cells = store.snapshot_cell_selection("small")

    # The detection bounds do not depend on the trend, so a coarse trend of
    # few bins, which fits far faster than the default one, selects the genes.
    ref = store.features.hvgs(cells, top_n=1000, n_bins=20, show_plot=False)

    parameters = inspect_artifact(store.zw, ref).parameters
    assert (parameters["min_cells"], parameters["max_cells"]) == (3, 41)
    counts = np.asarray(store.RNA.rawData[active[:41], :].compute())
    detected = (counts > 0).sum(axis=0)
    selected = np.asarray(store.artifacts.load(ref)["values"][:], dtype=bool)
    # The earlier bounds, 20 and 41 - 20, kept only genes detected in exactly
    # 20 of the 41 cells.
    assert selected.sum() == 1000
    assert detected[selected].min() == 3
    assert detected[selected].max() <= 40
    assert np.unique(detected[selected]).size > 10


# Pearson residual normalization


@pytest.mark.parametrize("renormalize_subset", [False, True])
def test_pearson_residual_normalizer_matches_numpy(banded, renormalize_subset) -> None:
    banded.RNA.normMethod = norm_pearson_residuals
    cells = banded.snapshot_cell_selection()
    subset = np.array([2, 5, 7, 9, 31])
    features = banded.features.snapshot(feature_indexes=subset)

    normalized = banded.features.normalize(
        cells, features, renormalize_subset=renormalize_subset
    )

    counts = _COUNTS.astype(np.float64)
    totals = counts[:, subset].sum(axis=1) if renormalize_subset else counts.sum(axis=1)
    expected = pearson_residuals(
        counts[:, subset],
        totals,
        counts[:, subset].sum(axis=0) / totals.sum(),
        theta=100.0,
        clip=np.sqrt(_N_CELLS),
    )
    np.testing.assert_allclose(
        banded.artifacts.load(normalized)["data"][:], expected, rtol=1e-6, atol=1e-6
    )
    parameters = inspect_artifact(banded.zw, normalized).parameters
    assert parameters["normalization_method"] == {
        "external_hook": True,
        "identity": "cytearc.assay.pearson_residuals:theta=100.0:clip=sqrt-n",
    }
    assert (parameters["log_transform"], parameters["renormalize_subset"]) == (
        False,
        renormalize_subset,
    )
    reduction = banded.reduction.pca(
        normalized, dims=2, feat_scaling=False, show_elbow_plot=False
    )
    assert banded.artifacts.load(reduction)["data"].shape == (_N_CELLS, 2)
    with pytest.raises(ValueError, match="does not support log_transform=True"):
        banded.features.normalize(cells, features, log_transform=True)


def test_pearson_residual_normalizer_options(banded) -> None:
    normalizer = PearsonResidualNormalizer(theta=5, clip=1.5)
    assert repr(normalizer) == "PearsonResidualNormalizer(theta=5.0, clip=1.5)"
    assert normalizer.artifact_identity == (
        "cytearc.assay.pearson_residuals:theta=5.0:clip=1.5"
    )
    banded.RNA.normMethod = normalizer
    values = np.asarray(banded.RNA.normed(feat_idx=np.array([5, 7])).compute())
    counts = _COUNTS.astype(np.float64)
    totals = counts.sum(axis=1)
    np.testing.assert_allclose(
        values,
        pearson_residuals(
            counts[:, [5, 7]],
            totals,
            counts[:, [5, 7]].sum(axis=0) / totals.sum(),
            theta=5.0,
            clip=1.5,
        ),
    )
    assert np.abs(values).max() == 1.5
    for arguments, error, message in (
        ({"theta": 0}, ValueError, "theta must be greater than 0"),
        ({"theta": "5"}, TypeError, "theta must be a real number"),
        ({"clip": -1}, ValueError, "clip must be at least 0"),
    ):
        with pytest.raises(error, match=message):
            PearsonResidualNormalizer(**arguments)


def test_pearson_residuals_need_totals_and_follow_the_flag_policy(banded) -> None:
    counts = np.array([[0, 2], [3, 1], [0, 0]])
    totals = counts.sum(axis=1)
    np.testing.assert_allclose(
        norm_pearson_residuals(SimpleNamespace(scalar=totals), counts),
        pearson_residuals(
            counts, totals, counts.sum(axis=0) / 6, theta=100.0, clip=np.sqrt(3)
        ),
    )
    np.testing.assert_array_equal(
        norm_pearson_residuals(SimpleNamespace(scalar=np.zeros(3)), np.zeros((3, 2))),
        np.zeros((3, 2)),
    )
    with pytest.raises(ValueError, match="only an RNA assay's normed provides"):
        norm_pearson_residuals(SimpleNamespace(), counts)

    banded.RNA.normMethod = norm_pearson_residuals
    assert applicable_normalization_flags(banded.RNA) == {"renormalize_subset"}
    stub = SimpleNamespace(normMethod=norm_pearson_residuals)
    assert Assay._normalization_flags(stub) == frozenset()  # type: ignore[arg-type]
    assert ATACassay._normalization_flags(stub) == frozenset()  # type: ignore[arg-type]
    # A custom ATAC normalizer still reads the term frequencies.
    custom = SimpleNamespace(normMethod=lambda assay, counts: counts)
    assert ATACassay._normalization_flags(custom) == {  # type: ignore[arg-type]
        "renormalize_subset"
    }


# Operation revision of select_hvgs

_BLACKLIST_BEFORE_1_3 = (
    "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|"
    "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$"
)


def test_hvg_revision_selects_variance_bounds_and_earlier_defaults() -> None:
    (revision,) = OPERATION_REVISIONS["select_hvgs"]
    applies = revision.applies
    assert (revision.revision, revision.release) == (2, "1.3.0")
    assert applies is not None
    unbounded = {"special_float": "-inf"}
    # The parameters that 1.2 recorded for its defaults over 892 cells.
    earlier = {
        "min_cells": 20,
        "max_cells": 872,
        "top_n": 2000,
        "min_var": unbounded,
        "max_var": {"special_float": "inf"},
        "min_mean": unbounded,
        "max_mean": {"special_float": "inf"},
        "n_bins": 200,
        "lowess_frac": 0.1,
        "blacklist": _BLACKLIST_BEFORE_1_3,
        "keep_bounds": False,
        "bin_strategy": "adaptive",
        "variance_estimator": "regularized_local_quantile",
        "variance_quantile": 0.25,
    }
    for parameters, expected in (
        (earlier, True),
        ({"min_var": 1.0, "max_var": {"special_float": "inf"}}, True),
        ({"min_var": unbounded, "max_var": 3}, True),
        ({"min_var": unbounded, "max_var": {"special_float": "inf"}}, False),
        ({"min_var": True, "max_var": "2"}, False),
        # The earlier default bounds with any blacklist, such as a species one.
        ({**earlier, "blacklist": ""}, True),
        ({**earlier, "blacklist": DEFAULT_HVG_BLACKLIST}, True),
        # The earlier default blacklist with other bounds.
        ({**earlier, "min_cells": 9}, True),
        ({**earlier, "min_cells": 9, "blacklist": DEFAULT_HVG_BLACKLIST}, False),
        ({"min_cells": True, "blacklist": ["^MT-"]}, False),
        ({}, False),
    ):
        assert applies("feature_selection", parameters, {}) is expected


def test_hvg_selection_records_its_revision_and_recomputes_earlier_ones(
    banded,
) -> None:
    cells = banded.snapshot_cell_selection()
    options = {"min_cells": 0, "max_cells": np.inf, "top_n": 5, "blacklist": ""}
    bounded = banded.features.hvgs(cells, min_var=-1, show_plot=False, **options)
    assert inspect_artifact(banded.zw, bounded).revision == 2

    # A selection of the earlier code records no revision.
    group = zarr.open_group(banded.zw.store, mode="r+", path=artifact_path(bounded))
    provenance = dict(group.attrs["provenance"])
    provenance.pop("revision")
    group.attrs["provenance"] = provenance
    recomputed = banded.features.hvgs(cells, min_var=-1, show_plot=False, **options)
    assert recomputed != bounded
    earlier = inspect_artifact(banded.zw, bounded)
    assert earlier.revision == 1 and not earlier.is_current
    assert [entry.revision for entry in earlier.superseded_by] == [2]

    # A selection that the revision leaves out is reused.
    unbounded = banded.features.hvgs(cells, show_plot=False, **options)
    assert inspect_artifact(banded.zw, unbounded).revision == 1
    assert banded.features.hvgs(cells, show_plot=False, **options) == unbounded

    # The earlier defaults, requested explicitly, record the revision too.
    defaults = banded.features.hvgs(
        cells,
        min_cells=20,
        max_cells=70,
        top_n=5,
        blacklist=_BLACKLIST_BEFORE_1_3,
        show_plot=False,
    )
    assert inspect_artifact(banded.zw, defaults).revision == 2
    # So do the earlier default bounds with another blacklist.
    bounds = banded.features.hvgs(
        cells, min_cells=20, max_cells=70, top_n=5, blacklist="", show_plot=False
    )
    assert inspect_artifact(banded.zw, bounds).revision == 2


# Pipeline settings


def test_pipeline_hvg_settings_cover_the_flavors(banded) -> None:
    import inspect

    from cytearc.datastore._pipeline.recipes.rna import resolve_rna
    from cytearc.datastore.pipeline_accessor import PipelineAccessor

    # The other keywords of a public call, at their defaults.
    parameters = inspect.signature(PipelineAccessor.rna).parameters
    defaults = {
        name: parameter.default
        for name, parameter in parameters.items()
        if name not in {"self", "callback", "cancel"}
    }
    for settings in (
        {"flavor": "pearson_residuals", "batch": "batch", "theta": 50, "clip": 3.0},
        {"flavor": "seurat_v3", "span": 0.5},
    ):
        plan = resolve_rna(banded, **(defaults | {"hvg": settings}))
        assert plan.config["stages"]["hvg"] == settings
    for hvg, error, message in (
        ({"flavor": "bogus"}, ValueError, "flavor must be one of"),
        ({"span": 0.5}, ValueError, "span applies only"),
        ({"batch": 3}, TypeError, r"hvg\['batch'\] must be a cell"),
        ({"batch": "absent"}, KeyError, "HVG batch column 'absent' was not found"),
    ):
        with pytest.raises(error, match=message):
            banded.pipeline.rna(hvg=hvg)
    # Each call was refused before it recorded a run.
    assert banded.pipeline.list_runs() == ()


# Agreement with scanpy 1.12.4 and scikit-misc 0.5.3


def _golden() -> dict[str, Any]:
    return json.loads(_GOLDEN.read_text())


@pytest.fixture(scope="module")
def golden_store(datastore_zarr_root, tmp_path_factory) -> DataStore:
    location = tmp_path_factory.mktemp("hvg_golden") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    store = DataStore(str(location), default_assay="RNA", nthreads=2)
    active = np.flatnonzero(np.asarray(store.cells.fetch_all("I"), dtype=bool))
    labels = np.full(store.cells.N, "none", dtype=object)
    labels[active] = np.where(np.arange(len(active)) < len(active) // 2, "a", "b")
    store.cells.insert("golden_batch", labels.astype(str), overwrite=True)
    return store


def _golden_run(store: DataStore, flavor: str, **options: Any) -> dict[str, np.ndarray]:
    ref = store.features.hvgs(
        store.snapshot_cell_selection(),
        flavor=flavor,
        top_n=2000,
        blacklist="",
        show_plot=False,
        **options,
    )
    group = store.artifacts.load(ref)
    return {name: np.asarray(group[name][:]) for name in group.array_keys()}


def test_golden_fixture_pins_its_inputs(golden_store) -> None:
    golden = _golden()
    provenance = golden["provenance"]
    assert (provenance["package"], provenance["packageVersion"]) == (
        "scanpy",
        "1.12.4",
    )
    assert provenance["scikitMiscVersion"] == "0.5.3"
    assert (provenance["nCells"], provenance["nGenes"], provenance["topN"]) == (
        892,
        36601,
        2000,
    )
    active = np.flatnonzero(np.asarray(golden_store.cells.fetch_all("I"), dtype=bool))
    matrix = csr_matrix(
        np.asarray(golden_store.RNA.rawData[active, :].compute()).astype(np.int64)
    )
    digest = hashlib.sha256()
    for values in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(values, dtype="<i8").tobytes())
    assert digest.hexdigest() == golden["inputs"]["countsSha256"]


def _selection_agreement(
    ours: dict[str, np.ndarray], reference: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    expected = np.zeros(len(ours["values"]), dtype=bool)
    expected[reference["selected"]] = True
    index = np.asarray(reference["indices"])
    rho = spearmanr(ours["corrected_variance"][index], reference["scores"]).statistic
    return expected, np.flatnonzero(expected ^ ours["values"]), float(rho)


def test_seurat_v3_matches_scanpy(golden_store) -> None:
    reference = _golden()["scanpy"]["seurat_v3"]
    ours = _golden_run(golden_store, "seurat_v3")

    expected, differing, rho = _selection_agreement(ours, reference)
    index = np.asarray(reference["indices"])
    # The trend matches scikit-misc's interpolated LOESS to within 1e-12 on the
    # log10 scale, so the scores agree to the 10 digits the fixture keeps.
    np.testing.assert_allclose(
        ours["corrected_variance"][index], reference["scores"], rtol=1e-9
    )
    assert rho > 0.99999
    # Genes tied at the last selected score are ordered by index here and in
    # sort order by scanpy, so only such ties may differ.
    boundary = np.sort(ours["corrected_variance"][expected])[0]
    np.testing.assert_allclose(
        ours["corrected_variance"][differing], boundary, rtol=1e-9
    )
    assert len(differing) <= 4


def test_pearson_residuals_match_scanpy(golden_store) -> None:
    reference = _golden()["scanpy"]["pearson_residuals"]
    ours = _golden_run(golden_store, "pearson_residuals")

    _, differing, rho = _selection_agreement(ours, reference)
    index = np.asarray(reference["indices"])
    np.testing.assert_allclose(
        ours["corrected_variance"][index], reference["scores"], rtol=1e-9
    )
    assert rho > 0.999999
    assert len(differing) == 0


def test_batch_pearson_residuals_match_scanpy(golden_store) -> None:
    # scanpy clips every batch at the square root of the first batch's cell
    # count, so both use an explicit clip.
    reference = _golden()["scanpy"]["pearson_residuals_batch"]
    ours = _golden_run(
        golden_store, "pearson_residuals", batch="golden_batch", clip=20.0
    )

    _, differing, _ = _selection_agreement(ours, reference)
    index = np.asarray(reference["indices"])
    np.testing.assert_allclose(
        ours["corrected_variance"][index], reference["scores"], rtol=1e-9
    )
    assert len(differing) == 0
    selected = np.asarray(reference["selected"])
    np.testing.assert_array_equal(
        ours["batch_count"][selected], reference["batchCount"]
    )
    # scanpy ranks tied scores in sort order, so a median rank may differ by 1.
    difference = ours["median_batch_rank"][selected] - np.asarray(
        reference["medianRank"]
    )
    assert np.abs(difference).max() <= 1


def test_batch_seurat_v3_matches_scanpy_seurat_v3_paper(golden_store) -> None:
    reference = _golden()["scanpy"]["seurat_v3_paper_batch"]
    ours = _golden_run(golden_store, "seurat_v3", batch="golden_batch")

    _, differing, rho = _selection_agreement(ours, reference)
    index = np.asarray(reference["indices"])
    # scanpy averages each batch's counts with Welford updates, whose last bits
    # break ties between genes with equal means. The interpolated trend places
    # its vertices between tied means, so scanpy's batch scores differ from
    # those of the exact means used here by up to about 1e-3.
    np.testing.assert_allclose(
        ours["corrected_variance"][index], reference["scores"], rtol=1e-3
    )
    assert rho > 0.99999
    assert len(differing) <= 12


def test_corrected_variance_of_selections_without_counts_is_zero() -> None:
    from cytearc.assay.rna import _corrected_variance

    for n_selected in (0, 4):
        np.testing.assert_array_equal(
            _corrected_variance(
                np.zeros(3),
                np.zeros(3),
                n_selected,
                n_bins=200,
                lowess_frac=0.1,
                bin_strategy="adaptive",
            ),
            np.zeros(3),
        )


def test_pearson_residuals_refuse_negative_cell_totals(tmp_path) -> None:
    from tests.storage_helpers import write_count_store

    counts = _COUNTS[:12, :6].astype(np.float64)
    counts[2] = 0
    counts[2, :2] = [-3.0, 1.0]
    write_count_store(str(tmp_path / "store.zarr"), {"RNA": counts}, "float64")
    store = DataStore(
        str(tmp_path / "store.zarr"), default_assay="RNA", min_features_per_cell=-1
    )
    with pytest.raises(ValueError, match="Pearson residuals require finite"):
        store.features.hvgs(
            store.snapshot_cell_selection(),
            flavor="pearson_residuals",
            show_plot=False,
        )
    store.RNA.normMethod = norm_pearson_residuals
    with pytest.raises(ValueError, match="negative or non-finite totals"):
        store.RNA.normed()


def test_batch_selection_filters_each_batch_before_ranking() -> None:
    from cytearc.features.variability import select_batch_highly_variable_features

    ranking = select_batch_highly_variable_features(
        np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]),
        np.array([[1.0, 1.0, 8.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        np.array([5, 5, 5, 1]),
        np.array(["a", "MT-b", "c", "d"]),
        min_cells=2,
        max_cells=np.inf,
        top_n=1,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=2,
        blacklist="^MT-",
        keep_bounds=False,
    )

    # MT-b is blacklisted and d detected in one cell. Batch 0 drops c, whose
    # mean over detected cells exceeds 2**2, so a leads it and c leads batch
    # 1; tied in count and median rank, the lower index wins.
    np.testing.assert_array_equal(ranking.batch_count, [1, 0, 1, 0])
    np.testing.assert_array_equal(ranking.median_rank, [0, np.nan, 0, np.nan])
    np.testing.assert_array_equal(ranking.selected, [True, False, False, False])
