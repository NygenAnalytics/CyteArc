"""Public contracts of CyteArc's Harmony2: facade, validation, records and fit behaviour."""

import dataclasses
import inspect
import math
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cytearc.embeddings
import cytearc.embeddings.harmony
from cytearc.embeddings.harmony import (
    HarmonyResult,
    fit_harmony,
    harmony_cluster_count,
    harmony_peak_bytes,
    validate_harmony_parameters,
)
from cytearc.embeddings.harmony.fit import objective_cycle_slope, rounds_converged
from cytearc.embeddings.harmony.parameters import HARMONY_DEFAULTS
from tests.signature_contracts import signature_digest

# Small fits with few rounds often stop before the objective settles.
pytestmark = pytest.mark.filterwarnings("ignore:Harmony stopped before convergence")

_HARMONY_SIGNATURE_DIGEST = (
    "c6f8d86857d205e831ff1a407eca4eb95f89dd16a734941b9f22d3b8e9e3a066"
)


def _values(n_cells: int = 40, dims: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(dims, n_cells)) + 0.5


def _labels(n_cells: int = 40) -> pd.DataFrame:
    return pd.DataFrame({"batch": np.where(np.arange(n_cells) % 2, "b", "a")})


# Facade


def test_harmony_facade_exports_its_public_objects() -> None:
    harmony = cytearc.embeddings.harmony
    assert harmony.__all__ == [
        "HarmonyResult",
        "fit_harmony",
        "harmony_cluster_count",
        "harmony_peak_bytes",
        "validate_harmony_parameters",
    ]
    for name in harmony.__all__:
        assert getattr(harmony, name).__module__ == "cytearc.embeddings.harmony"
    assert cytearc.embeddings.fit_harmony is harmony.fit_harmony
    assert cytearc.embeddings.HarmonyResult is harmony.HarmonyResult
    # Retired names do not come back, and there is no initialisation hook.
    for module in (cytearc.embeddings, harmony):
        for name in (
            "run_harmony",
            "Harmony",
            "moe_correct_ridge",
            "safe_entropy",
            "ClusterFn",
        ):
            assert not hasattr(module, name), name
            assert name not in module.__all__
    assert "cluster_fn" not in inspect.signature(fit_harmony).parameters


def test_harmony_result_holds_the_fitted_state() -> None:
    assert [field.name for field in dataclasses.fields(HarmonyResult)] == [
        "original",
        "corrected",
        "assignments",
        "centroids",
        "sigma",
        "ridge_penalty",
        "active_levels",
        "batch_columns",
        "batch_levels",
        "parameters",
    ]
    result = fit_harmony(_values(), _labels(), n_clusters=3, max_iter_harmony=2)
    assert isinstance(result, HarmonyResult)
    assert result.centroids.shape == (3, 3)
    assert result.sigma.shape == (3,)
    assert result.ridge_penalty.shape == result.active_levels.shape == (3, 2)
    assert result.ridge_penalty.dtype == np.float64
    assert result.active_levels.dtype == np.bool_
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.sigma = np.ones(3)  # type: ignore[misc]


def test_harmony_signatures_are_pinned() -> None:
    methods: dict[str, object] = {
        "fit_harmony": fit_harmony,
        "harmony_cluster_count": harmony_cluster_count,
        "harmony_peak_bytes": harmony_peak_bytes,
        "validate_harmony_parameters": validate_harmony_parameters,
    }
    # Update the digest deliberately when a public signature changes.
    assert signature_digest(methods) == _HARMONY_SIGNATURE_DIGEST


def test_option_defaults_are_those_of_fit_harmony() -> None:
    signature = inspect.signature(fit_harmony)
    options = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name != "nthreads"
    }
    assert options == dict(HARMONY_DEFAULTS)
    assert list(options) == list(HARMONY_DEFAULTS)
    # The memory estimate assumes the update blocks of the fit's default.
    assert (
        inspect.signature(harmony_peak_bytes).parameters["block_size"].default
        == signature.parameters["block_size"].default
    )


# Input validation


@pytest.mark.parametrize(
    ("values", "metadata", "error", "match"),
    [
        (np.zeros(4), _labels(4), ValueError, "two-dimensional"),
        (np.array([["a", "b"]]), _labels(2), TypeError, "real numbers"),
        (np.ones((2, 1)), _labels(1), ValueError, "at least two cells"),
        (np.ones((0, 4)), _labels(4), ValueError, "at least one dimension"),
        (_values(4), {"batch": ["a"] * 4}, TypeError, "pandas DataFrame"),
        (_values(4), _labels(3), ValueError, "metadata rows"),
        (
            _values(4),
            pd.DataFrame(index=range(4)),
            ValueError,
            "at least one batch metadata column",
        ),
        (
            _values(4),
            pd.DataFrame([["a", "b"]] * 4, columns=["x", "x"]),
            ValueError,
            "column names must be unique",
        ),
        (
            _values(4),
            pd.DataFrame({"batch": ["a", None, "b", "a"]}),
            ValueError,
            "cannot contain missing",
        ),
        (
            np.array([[1.0, np.nan, 2.0, 3.0]]),
            _labels(4),
            ValueError,
            "contains non-finite",
        ),
        (
            np.array([[1.0, 2.0, np.inf, 3.0]]),
            _labels(4),
            ValueError,
            "contains non-finite",
        ),
    ],
)
def test_fit_harmony_refuses_invalid_data(
    values: np.ndarray, metadata: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        fit_harmony(values, metadata)  # type: ignore[arg-type]


def test_cluster_count_must_lie_between_one_and_the_cell_count() -> None:
    values, metadata = _values(10), _labels(10)
    for n_clusters in (0, 11):
        with pytest.raises(ValueError, match="between one and the cell count"):
            fit_harmony(values, metadata, n_clusters=n_clusters)
        with pytest.raises(ValueError, match="between one and the cell count"):
            harmony_cluster_count(10, n_clusters)
    with pytest.raises(TypeError, match="n_clusters must be an integer"):
        fit_harmony(values, metadata, n_clusters=2.0)  # type: ignore[arg-type]


def test_sigma_is_scalar_or_one_value_per_cluster() -> None:
    values, metadata = _values(), _labels()
    with pytest.raises(ValueError, match="sigma must be scalar"):
        fit_harmony(values, metadata, n_clusters=2, sigma=[0.1, 0.2, 0.3])
    for sigma in (0.0, -1.0, math.inf):
        with pytest.raises(
            ValueError, match="sigma values must be finite and positive"
        ):
            fit_harmony(values, metadata, sigma=sigma)
    result = fit_harmony(values, metadata, n_clusters=2, sigma=[0.1, 0.2])
    assert result.sigma.tolist() == [0.1, 0.2]


def test_theta_has_one_value_per_level_in_design_order() -> None:
    values, metadata = _values(), _labels()
    with pytest.raises(
        ValueError,
        match=(
            r"Each Harmony batch level must have a theta.*one per batch level "
            r"\(batch=a, batch=b\); got shape \(3,\)"
        ),
    ):
        fit_harmony(values, metadata, theta=[1.0, 2.0, 3.0])
    for theta in (-1.0, math.nan):
        with pytest.raises(
            ValueError, match="theta values must be finite and non-negative"
        ):
            fit_harmony(values, metadata, theta=theta)


def test_ridge_penalty_is_fixed_and_positive_or_dynamic() -> None:
    values, metadata = _values(), _labels()
    with pytest.raises(
        ValueError, match="ridge_penalty values must be finite and positive"
    ):
        fit_harmony(values, metadata, ridge_penalty=0.0)
    with pytest.raises(ValueError, match="not both"):
        fit_harmony(values, metadata, ridge_penalty=1.0, ridge_alpha=0.2)
    with pytest.raises(ValueError, match="ridge_alpha must be greater than 0"):
        fit_harmony(values, metadata, ridge_alpha=0.0)
    with pytest.raises(
        ValueError,
        match=(
            r"Each Harmony batch level must have a ridge_penalty.*\(batch=a, "
            r"batch=b\); got shape \(3,\)"
        ),
    ):
        fit_harmony(values, metadata, ridge_penalty=[1.0, 2.0, 3.0])


def test_iteration_limits_must_be_at_least_one() -> None:
    values, metadata = _values(), _labels()
    for option in ("max_iter_cluster", "max_iter_harmony"):
        with pytest.raises(ValueError, match="must be at least 1"):
            fit_harmony(values, metadata, **{option: 0})


@pytest.mark.parametrize("keyword", ["nclust", "max_iter_kmeans", "lamb", "cluster_fn"])
def test_fit_harmony_has_no_retired_keywords(keyword: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fit_harmony(_values(), _labels(), **{keyword: 5})


def test_batch_columns_need_orderable_values_with_distinct_text() -> None:
    values = _values(4)
    with pytest.raises(ValueError, match="mixes values that cannot be ordered"):
        fit_harmony(
            values, pd.DataFrame({"batch": pd.Series([1, "a", 1, "a"], dtype=object)})
        )
    with pytest.raises(ValueError, match="distinct values with the same text '1'"):
        fit_harmony(
            values,
            pd.DataFrame(
                {"batch": pd.Categorical([1, "1", 1, "1"], categories=[1, "1"])}
            ),
        )


# validate_harmony_parameters


def test_validate_harmony_parameters_returns_a_new_equal_dict() -> None:
    assert validate_harmony_parameters(None) == {}
    theta = [1.0, 2.0]
    mapping = {"n_clusters": 3, "theta": theta, "ridge_alpha": 0.5}
    resolved = validate_harmony_parameters(mapping)
    assert resolved == mapping
    assert resolved is not mapping
    # The values are untouched: the artifact identity records the user's mapping.
    assert resolved["theta"] is theta
    # An explicit None asks for the default cluster count.
    assert validate_harmony_parameters({"n_clusters": None}) == {"n_clusters": None}


@pytest.mark.parametrize(
    ("parameters", "error", "match"),
    [
        (["theta"], TypeError, "mapping"),
        ({1: 2}, TypeError, "keys must be strings"),
        ({"data_mat": 1}, ValueError, "Unsupported.*data_mat"),
        ({"bogus": 1}, ValueError, "Unsupported Harmony parameters: bogus"),
        (
            {"nclust": 5},
            ValueError,
            "Unsupported Harmony parameter 'nclust': it was renamed to 'n_clusters'",
        ),
        (
            {"max_iter_kmeans": 5},
            ValueError,
            "Unsupported Harmony parameter 'max_iter_kmeans': it was renamed to "
            "'max_iter_cluster'",
        ),
        (
            {"lamb": 1.0},
            ValueError,
            "Unsupported Harmony parameter 'lamb': it was renamed to 'ridge_penalty'; "
            "ridge_penalty is a fixed penalty, so leave it out to use Harmony2's "
            "dynamic penalty",
        ),
        ({"cluster_fn": "kmeans"}, ValueError, "no initialisation hook"),
        ({"nthreads": 2}, ValueError, "nthreads"),
        ({"values": 2}, ValueError, "the DataStore supplies the coordinates"),
        ({"n_clusters": 0}, ValueError, "n_clusters must be at least 1"),
        ({"n_clusters": True}, TypeError, "n_clusters must be an integer"),
        ({"max_iter_cluster": True}, TypeError, "max_iter_cluster must be an integer"),
        ({"max_iter_harmony": 2.0}, TypeError, "max_iter_harmony must be an integer"),
        ({"max_iter_harmony": 0}, ValueError, "max_iter_harmony must be at least 1"),
        ({"random_state": None}, TypeError, "random_state must be an integer"),
        ({"random_state": -1}, ValueError, "random_state must be at least 0"),
        ({"theta": "2"}, TypeError, "theta must contain real numbers"),
        ({"theta": [1.0, True]}, TypeError, "theta must contain real numbers"),
        ({"sigma": {"a": 1.0}}, TypeError, "sigma must contain real numbers"),
        ({"theta": [[1.0, 2.0]]}, ValueError, "one-dimensional sequence"),
        (
            {"theta": [1.0, [2.0, 3.0]]},
            ValueError,
            "theta must be a scalar or a one-dimensional sequence",
        ),
        ({"sigma": [0.1, None]}, TypeError, "sigma must contain real numbers"),
        ({"theta": []}, ValueError, "theta must not be empty"),
        ({"tau": "x"}, TypeError, "tau must be a real number"),
        ({"tau": math.nan}, ValueError, "tau must be finite"),
        ({"tau": -1.0}, ValueError, "tau must be at least 0"),
        ({"block_size": math.inf}, ValueError, "block_size must be finite"),
        ({"block_size": 0}, ValueError, "block_size must be greater than 0"),
        ({"epsilon_cluster": -1}, ValueError, "epsilon_cluster must be at least 0"),
        ({"epsilon_harmony": -1e-3}, ValueError, "epsilon_harmony must be at least 0"),
        (
            {"prune_threshold": 1},
            ValueError,
            "prune_threshold must be at least 0 and less than 1",
        ),
        ({"ridge_penalty": [1.0, -1.0]}, ValueError, "finite and positive"),
        ({"ridge_penalty": 1.0, "ridge_alpha": 0.3}, ValueError, "not both"),
    ],
)
def test_validate_harmony_parameters_refuses_invalid_options(
    parameters: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        validate_harmony_parameters(parameters)  # type: ignore[arg-type]


# Fit behaviour


def test_original_is_an_independent_float64_copy() -> None:
    values = np.arange(80, dtype=np.int64).reshape(2, 40) % 7 + 1
    result = fit_harmony(values, _labels(), n_clusters=2)

    np.testing.assert_array_equal(result.original, values)
    assert not np.shares_memory(result.original, values)
    for array in (result.original, result.corrected, result.assignments):
        assert array.dtype == np.float64
    assert result.original.shape == result.corrected.shape == (2, 40)
    assert result.assignments.shape == (2, 40)
    assert result.batch_columns == ("batch",)
    assert result.batch_levels == (("a", "b"),)


class _Progress:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.updates: list[int] = []
        self.closed = 0

    def update(self, count: int = 1) -> None:
        self.updates.append(count)

    def close(self) -> None:
        self.closed += 1


def _record_progress(monkeypatch: pytest.MonkeyPatch) -> list[_Progress]:
    bars: list[_Progress] = []

    def tqdmbar(**options: Any) -> _Progress:
        bars.append(_Progress(**options))
        return bars[-1]

    monkeypatch.setattr("cytearc.embeddings.harmony.fit.tqdmbar", tqdmbar)
    return bars


def test_progress_covers_the_rounds_and_closes_once(monkeypatch) -> None:
    bars = _record_progress(monkeypatch)

    result = fit_harmony(
        _values(), _labels(), n_clusters=2, max_iter_harmony=5, epsilon_harmony=1e9
    )

    (bar,) = bars
    assert bar.options["total"] == 5
    # A huge tolerance stops after two rounds; the bar still reaches its total.
    assert result.parameters["rounds"] == 2
    assert result.parameters["converged"] is True
    assert sum(bar.updates) == 5
    assert bar.closed == 1


def test_progress_closes_once_when_a_round_fails(monkeypatch) -> None:
    bars = _record_progress(monkeypatch)

    def fail(*_arguments: Any) -> np.ndarray:
        raise RuntimeError("regression failed")

    monkeypatch.setattr(
        "cytearc.embeddings.harmony.correction.solve_cluster_regressions", fail
    )
    with pytest.raises(RuntimeError, match="regression failed"):
        fit_harmony(_values(), _labels(), n_clusters=2, max_iter_harmony=5)

    (bar,) = bars
    assert bar.closed == 1
    assert sum(bar.updates) == 0


def test_numeric_batch_columns_and_a_private_generator() -> None:
    values = _values(60, seed=3)
    metadata = pd.DataFrame({"batch": np.arange(60) % 3})
    state = np.random.get_state()

    first = fit_harmony(values, metadata, n_clusters=3, random_state=7)
    second = fit_harmony(values, metadata, n_clusters=3, random_state=7)

    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert first.batch_levels == (("0", "1", "2"),)
    np.testing.assert_array_equal(first.corrected, second.corrected)
    np.testing.assert_array_equal(first.assignments, second.assignments)
    assert first.parameters == second.parameters
    assert first.parameters["randomState"] == 7


def test_a_cancelling_cluster_keeps_a_zero_centroid() -> None:
    values = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    metadata = pd.DataFrame({"batch": ["a", "b", "a", "b"]})

    result = fit_harmony(values, metadata, n_clusters=1, ridge_penalty=1.0)

    # One cluster holds every cell, so the fixed-penalty regression removes
    # two thirds of each batch mean: lambda = 1 against a mass of 2.
    np.testing.assert_allclose(
        result.corrected,
        np.array([[2.0, -2.0, -1.0, 1.0], [-1.0, 1.0, 2.0, -2.0]]) / 3,
        rtol=0,
        atol=1e-12,
    )
    assert np.all(result.centroids == 0.0)


def test_an_exactly_zero_objective_converges_without_warnings() -> None:
    values = np.array([[2.0, 3.0, 5.0, 7.0]])
    metadata = pd.DataFrame({"batch": ["a", "b", "a", "b"]})

    # Under Harmony2's stabilised penalty a balanced cluster still has a
    # positive diversity term, so the objective is zero only with theta = 0.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(divide="raise", invalid="raise"):
            result = fit_harmony(
                values, metadata, n_clusters=1, theta=0.0, ridge_penalty=1.0
            )

    np.testing.assert_allclose(
        result.corrected, [[2.5, 2.5, 5.5, 6.5]], rtol=0, atol=1e-12
    )
    assert result.parameters["objective"] == [0.0, 0.0]
    assert result.parameters["converged"] is True


def test_the_dynamic_penalty_shrinks_a_balanced_batch_offset() -> None:
    values = np.array([[2.0, 3.0, 5.0, 7.0]])
    metadata = pd.DataFrame({"batch": ["a", "b", "a", "b"]})

    result = fit_harmony(values, metadata, n_clusters=1, theta=0.0)

    # lambda = 0.2 * 2 against a mass of 2 per batch keeps 1/6 of each offset.
    np.testing.assert_allclose(
        result.corrected, [[2.625, 2.375, 5.625, 6.375]], rtol=0, atol=1e-12
    )


def _nan_regressions(statistics: Any, *_arguments: Any) -> np.ndarray:
    return np.full_like(statistics.level_sums, np.nan)


def _nan_centroids(
    corrected: np.ndarray,
    _inverse_norms: np.ndarray,
    assignments: np.ndarray,
    _workspace: Any,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (corrected.shape[1], assignments.shape[1])
    return np.full(shape, np.nan), np.ones(assignments.shape[1])


def _nan_assignments(
    _similarity: np.ndarray,
    cells: np.ndarray,
    _inverse_norms: np.ndarray,
    _two_over_sigma: np.ndarray,
    _penalty: np.ndarray,
    _level_codes: np.ndarray,
    assignments: np.ndarray,
    entropies: np.ndarray,
) -> None:
    assignments[cells] = np.nan
    entropies[:] = np.nan


@pytest.mark.parametrize(
    ("target", "replacement", "match"),
    [
        (
            "cytearc.embeddings.harmony.correction.solve_cluster_regressions",
            _nan_regressions,
            "^Harmony produced non-finite corrected coordinates in round 0$",
        ),
        (
            "cytearc.embeddings.harmony.clustering.cosine_centroids",
            _nan_centroids,
            "^Harmony produced non-finite centroids in round 0$",
        ),
        (
            "cytearc.embeddings.harmony.clustering.assign_cells",
            _nan_assignments,
            "^Harmony produced non-finite cluster assignments in round 0$",
        ),
    ],
)
def test_non_finite_states_raise(
    monkeypatch, target: str, replacement: Callable[..., Any], match: str
) -> None:
    monkeypatch.setattr(target, replacement)
    with pytest.raises(ValueError, match=match):
        fit_harmony(_values(), _labels(), n_clusters=2)


def test_a_cluster_without_cells_is_refused(monkeypatch) -> None:
    def everything_in_the_first_cluster(
        _similarity: np.ndarray,
        cells: np.ndarray,
        _inverse_norms: np.ndarray,
        _two_over_sigma: np.ndarray,
        _penalty: np.ndarray,
        _level_codes: np.ndarray,
        assignments: np.ndarray,
        entropies: np.ndarray,
    ) -> None:
        assignments[cells] = 0.0
        assignments[cells, 0] = 1.0
        entropies[:] = 0.0

    monkeypatch.setattr(
        "cytearc.embeddings.harmony.clustering.assign_cells",
        everything_in_the_first_cluster,
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Harmony left 1 of 2 clusters without assigned cells; use a larger "
            "sigma or fewer clusters"
        ),
    ):
        fit_harmony(_values(), _labels(), n_clusters=2)


def test_unused_categories_are_not_batch_levels() -> None:
    values = _values()
    labels = np.where(np.arange(40) % 2, "b", "a")
    declared = pd.DataFrame(
        {"batch": pd.Categorical(labels, categories=["a", "b", "c"])}
    )

    with_unused = fit_harmony(values, declared, n_clusters=2)
    plain = fit_harmony(values, pd.DataFrame({"batch": labels}), n_clusters=2)

    np.testing.assert_array_equal(with_unused.corrected, plain.corrected)
    assert with_unused.batch_levels == (("a", "b"),)
    assert with_unused.parameters["phiColumns"] == ["batch_a", "batch_b"]
    assert len(with_unused.parameters["theta"]) == 2  # type: ignore[arg-type]
    assert with_unused.ridge_penalty.shape == (2, 2)


def test_design_order_follows_categories_sorted_values_and_columns() -> None:
    values = _values(4)
    categorical = fit_harmony(
        values,
        pd.DataFrame(
            {"batch": pd.Categorical(["b", "a", "b", "a"], categories=["c", "b", "a"])}
        ),
        n_clusters=1,
    )
    assert categorical.batch_levels == (("b", "a"),)
    assert categorical.parameters["phiColumns"] == ["batch_b", "batch_a"]

    numbers = fit_harmony(values, pd.DataFrame({"batch": [10, 9, 1, 10]}), n_clusters=1)
    assert numbers.batch_levels == (("1", "9", "10"),)

    values = _values(6)
    metadata = pd.DataFrame(
        {
            "lane": ["y", "x", "y", "x", "y", "x"],
            "donor": ["d", "e", "f", "d", "e", "f"],
        }
    )
    result = fit_harmony(values, metadata, n_clusters=1, theta=[1.0, 3.0])
    assert result.batch_columns == ("lane", "donor")
    assert result.batch_levels == (("x", "y"), ("d", "e", "f"))
    assert result.parameters["phiColumns"] == [
        "lane_x",
        "lane_y",
        "donor_d",
        "donor_e",
        "donor_f",
    ]
    # Per-column theta follows the design order of phiColumns.
    assert result.parameters["theta"] == [1.0, 1.0, 3.0, 3.0, 3.0]
    per_level = fit_harmony(
        values, metadata, n_clusters=1, theta=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert per_level.parameters["theta"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_theta_is_expanded_and_discounted_by_tau() -> None:
    metadata = pd.DataFrame({"x": ["p", "q", "p", "q"], "y": ["r", "r", "s", "s"]})

    result = fit_harmony(_values(4), metadata, n_clusters=2, tau=1.0, theta=[2.0, 3.0])

    # Each level holds 2 cells: theta_b (1 - exp(-(N_b / (K tau))^2)) with
    # N_b / (K tau) = 1.
    assert result.parameters["theta"] == pytest.approx(
        [value * (1.0 - math.exp(-1.0)) for value in (2.0, 2.0, 3.0, 3.0)]
    )
    assert result.parameters["tau"] == 1.0


def test_parameters_record_the_resolved_fit() -> None:
    result = fit_harmony(_values(), _labels(), n_clusters=2, max_iter_harmony=3)
    parameters = result.parameters

    assert set(parameters) == {
        "algorithm",
        "nClusters",
        "theta",
        "tau",
        "sigma",
        "ridge",
        "ridgeAlpha",
        "ridgePenalty",
        "pruneThreshold",
        "blockSize",
        "maxIterCluster",
        "maxIterHarmony",
        "epsilonCluster",
        "epsilonHarmony",
        "randomState",
        "initialization",
        "phiColumns",
        "rounds",
        "converged",
        "objective",
        "clusterPasses",
    }
    assert parameters["algorithm"] == "harmony2"
    assert parameters["initialization"] == "kmeans++"
    assert parameters["ridge"] == "dynamic" and parameters["ridgeAlpha"] == 0.2
    assert parameters["pruneThreshold"] == 1e-5
    assert parameters["blockSize"] == 0.05
    assert (parameters["maxIterCluster"], parameters["maxIterHarmony"]) == (200, 3)
    rounds = parameters["rounds"]
    assert isinstance(rounds, int) and 1 <= rounds <= 3
    objective = parameters["objective"]
    passes = parameters["clusterPasses"]
    assert isinstance(objective, list) and isinstance(passes, list)
    assert len(objective) == len(passes) == rounds
    assert all(type(value) is float for value in objective)
    assert all(type(count) is int and count >= 1 for count in passes)


def test_a_fit_that_does_not_settle_warns_once() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(5, 300))
    metadata = pd.DataFrame({"b": rng.integers(0, 3, size=300)})

    with pytest.warns(
        UserWarning, match="Harmony stopped before convergence after 2 rounds"
    ) as caught:
        result = fit_harmony(
            values, metadata, n_clusters=5, max_iter_harmony=2, epsilon_harmony=0.0
        )

    assert len(caught) == 1
    assert "above epsilon_harmony=0.0" in str(caught[0].message)
    assert result.parameters["converged"] is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # One round never assesses convergence, so it never warns.
        single = fit_harmony(values, metadata, n_clusters=5, max_iter_harmony=1)
    assert single.parameters["converged"] is False


# Stopping rule of the rounds


def _first_converged_round(
    objectives: list[float], epsilon: float = 1e-4
) -> int | None:
    """The 0-based round after which the rule stops, or None."""
    for stop in range(1, len(objectives)):
        if rounds_converged(objectives[: stop + 1], epsilon):
            return stop
    return None


def test_cycle_slope_removes_a_two_round_alternation_and_keeps_a_trend() -> None:
    rounds = np.arange(5.0)
    level, trend = 5.0e4, 3.0
    # A constant, a two-round alternation and one whose amplitude changes
    # linearly all vanish; a linear trend comes back exactly.
    for amplitude in (np.full(5, 40.0), 40.0 + 7.0 * rounds):
        alternating = amplitude * (-1.0) ** rounds
        objectives = list(level + trend * rounds + alternating)
        assert objective_cycle_slope(objectives) == pytest.approx(trend, rel=1e-9)
    assert objective_cycle_slope([level] * 5) == 0.0
    # Only the last five objectives count.
    assert objective_cycle_slope([1.0e9] + [level] * 5) == 0.0
    with pytest.raises(ValueError, match="at least five"):
        objective_cycle_slope([level] * 4)


def test_rounds_converge_on_a_repeated_objective() -> None:
    # The papers' test on the first comparison, then two small changes in a row.
    assert _first_converged_round([100.0, 100.005]) == 1
    assert _first_converged_round([100.0, 103.0, 103.001, 103.002]) == 3
    assert not rounds_converged([100.0], 1e-4)
    # An objective that stays exactly zero converges without dividing by it.
    with np.errstate(all="raise"):
        assert rounds_converged([0.0, 0.0], 1e-4)
        assert rounds_converged([7.0, 0.0, 0.0, 0.0], 1e-4)


def test_a_chance_crossing_of_an_alternating_objective_does_not_stop_the_fit() -> None:
    # The objective rises by 1e-3 of its value per round in steps: an
    # alternation of the same size cancels the rise in every second round, so
    # every second change is zero although the objective keeps moving.
    rounds = np.arange(8.0)
    objectives = list(5.0e4 * (1.0 + 1e-3 * rounds - 5e-4 * (-1.0) ** rounds))
    changes = np.abs(np.diff(objectives)) / np.abs(objectives[:-1])
    assert changes[1] < 1e-12 and changes[2] > 1e-3
    # A rule that stops on one small change would stop after round 2.
    assert _first_converged_round(objectives) is None


def test_a_settled_two_round_cycle_converges() -> None:
    # Consecutive objectives alternate by 1e-3 of their level forever, so no
    # change between consecutive rounds ever falls below epsilon, but their
    # two-round average settles: the fit has converged at the fifth round.
    rounds = np.arange(10.0)
    settling = 5.0e4 + 400.0 * 0.3**rounds
    objectives = list(settling + 25.0 * (-1.0) ** rounds)
    changes = np.abs(np.diff(objectives)) / np.abs(objectives[:-1])
    assert changes.min() > 1e-4
    stop = _first_converged_round(objectives)
    assert stop is not None and stop >= 4
    level = (objectives[stop] + objectives[stop - 1]) / 2
    assert abs(objective_cycle_slope(objectives[: stop + 1])) <= 1e-4 * level
    # A cycle whose average still moves by 6e-4 of it per round has not converged.
    drifting = list(np.asarray(objectives) + 30.0 * rounds)
    assert _first_converged_round(drifting) is None


def _oscillating_rounds(
    monkeypatch: pytest.MonkeyPatch, amplitude: float, drift: float
) -> None:
    """Make the rounds report objectives that alternate (and drift) around the real ones."""
    import cytearc.embeddings.harmony.fit as fit_module
    from cytearc.embeddings.harmony.clustering import RoundSummary

    cluster_round = fit_module.cluster_round

    def oscillating(*arguments: Any) -> RoundSummary:
        summary = cluster_round(*arguments)
        round_index = arguments[-1]
        last = summary.objectives[-1]
        reported = last * (
            1.0 + amplitude * (-1.0) ** round_index + drift * round_index
        )
        return RoundSummary(
            objectives=(*summary.objectives[:-1], reported),
            centroids=summary.centroids,
        )

    monkeypatch.setattr(fit_module, "cluster_round", oscillating)


def test_a_fit_whose_objective_alternates_stops_without_a_warning(monkeypatch) -> None:
    _oscillating_rounds(monkeypatch, amplitude=1e-3, drift=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fit_harmony(_values(), _labels(), n_clusters=2, max_iter_harmony=10)

    objectives = np.asarray(result.parameters["objective"])
    # No two consecutive objectives agree within epsilon_harmony, so a rule
    # that compares consecutive rounds alone would run all ten rounds and warn.
    assert np.all(np.abs(np.diff(objectives)) > 1e-4 * np.abs(objectives[:-1]))
    assert result.parameters["converged"] is True
    assert 5 <= result.parameters["rounds"] < 10


@pytest.mark.parametrize("rounds", [3, 6])
def test_a_fit_whose_alternating_objective_keeps_moving_warns(
    monkeypatch, rounds: int
) -> None:
    _oscillating_rounds(monkeypatch, amplitude=1e-3, drift=1e-3)

    with pytest.warns(
        UserWarning, match=f"Harmony stopped before convergence after {rounds} rounds"
    ) as caught:
        result = fit_harmony(
            _values(), _labels(), n_clusters=2, max_iter_harmony=rounds
        )

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "in the last two rounds" in message
    assert "epsilon_harmony=0.0001" in message
    # The two-round average enters the rule, and the message, from five rounds.
    assert ("and its two-round average by" in message) is (rounds >= 5)
    assert result.parameters["converged"] is False


# harmony_peak_bytes


def test_harmony_peak_bytes_validates_its_arguments() -> None:
    assert harmony_peak_bytes(100, 5, 4, [2, 3], n_combinations=6) > 0
    for arguments, options, error, match in (
        ((0, 5, 4, [2]), {}, ValueError, "n_cells must be at least 1"),
        ((100, 0, 4, [2]), {}, ValueError, "dims must be at least 1"),
        ((100, 5, 0, [2]), {}, ValueError, "n_clusters must be at least 1"),
        ((100, 5, 2.0, [2]), {}, TypeError, "n_clusters must be an integer"),
        ((100, 5, 101, [2]), {}, ValueError, "between one and the cell count"),
        ((100, 5, 4, 2), {}, TypeError, "column_levels must be a sequence"),
        ((100, 5, 4, "2"), {}, TypeError, "column_levels must be a sequence"),
        ((100, 5, 4, [2.0]), {}, TypeError, "column_levels must be a sequence"),
        ((100, 5, 4, []), {}, ValueError, "at least one level count"),
        ((100, 5, 4, [0]), {}, ValueError, "each between 1 and n_cells"),
        ((100, 5, 4, [101]), {}, ValueError, "each between 1 and n_cells"),
        ((100, 5, 4, [2, 3]), {"n_combinations": 2}, ValueError, "n_combinations"),
        ((100, 5, 4, [2, 3]), {"n_combinations": 7}, ValueError, "n_combinations"),
        (
            (100, 5, 4, [2]),
            {"block_size": True},
            TypeError,
            "block_size must be a real number",
        ),
        (
            (100, 5, 4, [2]),
            {"block_size": math.inf},
            ValueError,
            "block_size must be finite",
        ),
        (
            (100, 5, 4, [2]),
            {"block_size": 1.5},
            ValueError,
            "block_size must be greater than 0 and at most 1",
        ),
    ):
        with pytest.raises(error, match=match):
            harmony_peak_bytes(*arguments, **options)  # type: ignore[arg-type]


def test_harmony_peak_bytes_grows_with_the_dominant_terms() -> None:
    base = harmony_peak_bytes(10_000, 30, 100, [20])
    # Per cell, the coordinates (three float64 copies) and the assignments dominate.
    per_cell = (harmony_peak_bytes(20_000, 30, 100, [20]) - base) / 10_000
    assert 24 * 30 + 8 * 100 <= per_cell <= 2 * (24 * 30 + 8 * 100)
    # Several columns need the dense Schur complements and the combinations.
    assert harmony_peak_bytes(1_000, 5, 4, [20, 30]) > harmony_peak_bytes(
        1_000, 5, 4, [50]
    )
    assert harmony_peak_bytes(
        1_000, 5, 4, [20, 30], n_combinations=30
    ) <= harmony_peak_bytes(1_000, 5, 4, [20, 30])
