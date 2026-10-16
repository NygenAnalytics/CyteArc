import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from cytearc.regulons.grn import (
    GRNBOOST2_PARAMETERS,
    GrnParameters,
    _EarlyStopMonitor,
    _StageEarlyStopMonitor,
    infer_grn,
)

_FAST = GrnParameters(maxStages=200, earlyStopWindow=10)


def _expression(seed: int = 0, n_cells: int = 120) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    tfs = rng.poisson(2.0, size=(n_cells, 4)).astype(np.float32)
    targets = np.column_stack(
        [
            tfs[:, 0] * 2 + rng.poisson(0.5, n_cells),
            tfs[:, 1] + tfs[:, 2] + rng.poisson(0.5, n_cells),
            rng.poisson(1.0, n_cells),
            np.zeros(n_cells),
        ]
    ).astype(np.float32)
    names = ["TF0", "TF1", "TF2", "TF3", "G0", "G1", "G2", "Gconst"]
    return np.hstack([tfs, targets]), names


def test_grnboost2_defaults_match_arboreto():
    assert GRNBOOST2_PARAMETERS.regressor_kwargs() == {
        "learning_rate": 0.01,
        "n_estimators": 5000,
        "max_features": 0.1,
        "subsample": 0.9,
    }
    assert GRNBOOST2_PARAMETERS.earlyStopWindow == 25
    assert GRNBOOST2_PARAMETERS.earlyStopping == "stage"


@pytest.mark.parametrize("rule", ["stage", "consecutive"])
def test_grn_matches_a_direct_arboreto_style_fit(rule):
    expression, names = _expression()
    tf_names = ["TF0", "TF1", "TF2", "TF3"]
    parameters = GrnParameters(maxStages=200, earlyStopWindow=10, earlyStopping=rule)

    links = infer_grn(
        expression, names, tf_names, targets=["G1"], seed=7, parameters=parameters
    )

    regressor = GradientBoostingRegressor(
        random_state=7, **parameters.regressor_kwargs()
    )
    monitor = _StageEarlyStopMonitor(10) if rule == "stage" else _EarlyStopMonitor(10)
    regressor.fit(expression[:, :4], expression[:, 5], monitor=monitor)
    expected = regressor.feature_importances_ * len(regressor.estimators_)
    observed = links.set_index("TF")["importance"].reindex(tf_names, fill_value=0.0)
    np.testing.assert_allclose(observed.to_numpy(), expected)
    assert (links["target"] == "G1").all()


def test_grn_leaves_each_factor_out_of_its_own_model():
    expression, names = _expression()

    links = infer_grn(
        expression, names, names[:4], targets=["TF1"], seed=1, parameters=_FAST
    )
    # A lone factor has no other factor to predict it from.
    alone = infer_grn(
        expression, names, ["TF1"], targets=["TF1"], seed=1, parameters=_FAST
    )

    assert "TF1" not in set(links["TF"])
    assert set(links["TF"]) <= {"TF0", "TF2", "TF3"}
    assert alone.empty
    assert list(alone.columns) == list(links.columns)


def test_grn_finds_the_simulated_regulators():
    expression, names = _expression(n_cells=400)

    links = infer_grn(expression, names, names[:4], seed=0, parameters=_FAST)

    top = links.sort_values("importance", ascending=False).groupby("target").head(1)
    best = dict(zip(top["target"], top["TF"]))
    assert best["G0"] == "TF0"
    assert best["G1"] in {"TF1", "TF2"}
    # A constant target has no links.
    assert "Gconst" not in set(links["target"])


def test_grn_is_independent_of_thread_count_and_sorted():
    expression, names = _expression(3)

    serial = infer_grn(
        expression, names, names[:4], seed=5, parameters=_FAST, nthreads=1
    )
    threaded = infer_grn(
        expression, names, names[:4], seed=5, parameters=_FAST, nthreads=4
    )

    pd.testing.assert_frame_equal(serial, threaded)
    assert serial["importance"].is_monotonic_decreasing
    assert (serial["importance"] > 0).all()


def test_grn_reports_progress():
    expression, names = _expression()
    seen: list[tuple[int, int]] = []

    infer_grn(
        expression,
        names,
        names[:4],
        targets=["G0", "G2"],
        seed=0,
        parameters=_FAST,
        progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 2), (2, 2)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tf_names": ["nope"]}, "None of tf_names"),
        ({"targets": ["missing"]}, "not expression columns"),
        ({"seed": -1}, "seed"),
        ({"nthreads": 0}, "nthreads"),
        ({"gene_names": ["a"] * 8}, "unique"),
        ({"gene_names": ["a", "b"]}, "align"),
        ({"expression": np.zeros(8)}, "cells-by-genes"),
        ({"expression": np.ones((3, 8), dtype=complex)}, "real-valued"),
        ({"expression": np.full((3, 8), np.nan)}, "finite"),
        ({"expression": np.ones((1, 8))}, "two cells"),
    ],
)
def test_grn_rejects_invalid_requests(kwargs, message):
    expression, names = _expression()
    arguments = {
        "expression": expression,
        "gene_names": names,
        "tf_names": names[:4],
        "seed": 0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        infer_grn(**arguments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subsample": 1.0},
        {"maxFeatures": 0.0},
        {"earlyStopWindow": 0},
        {"learningRate": 0.0},
        {"earlyStopping": "never"},
    ],
)
def test_grn_parameters_reject_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        GrnParameters(**kwargs)


def test_stage_rule_requires_the_boosting_state():
    # scikit-learn could stop passing its fitting loop's variables to monitors.
    with pytest.raises(RuntimeError, match="earlyStopping='consecutive'"):
        _StageEarlyStopMonitor(5)(0, GradientBoostingRegressor(), {})


def test_infer_grn_reports_the_pearson_correlation_of_each_link():
    expression, names = _expression(5)

    links = infer_grn(expression, names, names[:4], seed=0, parameters=_FAST)

    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(expression.astype(np.float64).T)
    position = {name: index for index, name in enumerate(names)}
    expected = [
        correlation[position[tf], position[target]]
        for tf, target in zip(links["TF"], links["target"], strict=True)
    ]
    assert list(links.columns) == ["TF", "target", "importance", "correlation"]
    np.testing.assert_allclose(links["correlation"], expected)


def test_stage_rule_measures_each_stage_on_its_own_out_of_bag_cells():
    expression, _ = _expression(seed=8, n_cells=300)
    predictors = expression[:, :4]
    target = expression[:, 4].astype(np.float64)
    masks = []

    class Recorder(_StageEarlyStopMonitor):
        def __call__(self, current_round, regressor, fit_locals):
            masks.append(np.asarray(fit_locals["sample_mask"]).copy())
            super().__call__(current_round, regressor, fit_locals)
            return False

    recorder = Recorder(5)
    regressor = GradientBoostingRegressor(
        random_state=3,
        learning_rate=0.05,
        n_estimators=40,
        max_features=0.5,
        subsample=0.8,
    )
    regressor.fit(predictors, target, monitor=recorder)

    # scikit-learn before 1.3: MSE of the stage's out-of-bag cells before and
    # after the stage, the predictions replayed with staged_predict.
    staged = [np.full(len(target), float(regressor.init_.constant_.reshape(-1)[0]))]
    staged += list(regressor.staged_predict(predictors))
    expected = [
        np.mean((target[~mask] - staged[i][~mask]) ** 2)
        - np.mean((target[~mask] - staged[i + 1][~mask]) ** 2)
        for i, mask in enumerate(masks)
    ]
    np.testing.assert_allclose(recorder.improvements, expected, rtol=1e-9, atol=1e-12)
    # The rules differ: scikit-learn's improvements now compare different cells.
    assert not np.allclose(recorder.improvements, regressor.oob_improvement_)
