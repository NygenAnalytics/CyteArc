"""Native propeller against speckle::propeller 1.10.0 (limma 3.66.0)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytearc.abundance.propeller import propeller

FIXTURE = Path(__file__).with_name("speckle_1_10_0_propeller_golden.json")


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def _close(actual: Any, expected: Any) -> None:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(
        [np.inf if value == "Inf" else value for value in np.ravel(expected)],
        dtype=np.float64,
    ).reshape(np.shape(expected))
    np.testing.assert_allclose(actual_values, expected_values, rtol=1e-9, atol=1e-12)


def test_fixture_pins_speckle_and_its_inputs():
    fixture = _fixture()
    provenance = fixture["provenance"]
    assert provenance["packageVersion"] == "1.10.0"
    assert provenance["limmaVersion"] == "3.66.0"
    cases = fixture["inputs"]["cases"]
    encoded = json.dumps(cases, separators=(",", ":"), sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == fixture["inputs"]["sha256"]
    assert [case["name"] for case in cases] == [
        result["name"] for result in fixture["speckle"]
    ]


@pytest.mark.parametrize(
    "index", range(len(json.loads(FIXTURE.read_text())["inputs"]["cases"]))
)
def test_propeller_matches_speckle(index):
    fixture = _fixture()
    case = fixture["inputs"]["cases"][index]
    expected = fixture["speckle"][index]
    fit = propeller(
        np.asarray(case["counts"]),
        categories=case["categories"],
        samples=case["samples"],
        sample_groups=case["groups"],
        transform=case["transform"],
        robust=case["robust"],
        trend=case["trend"],
    )
    table = fit.table
    _close(fit.proportions, expected["proportions"])
    _close(fit.transformed, expected["transformed"])
    _close(table["baseline_prop"], expected["baselineProp"])
    for column, values in zip(
        [name for name in table.columns if name.startswith("prop_mean_")],
        expected["propMean"],
        strict=True,
    ):
        _close(table[column], values)
    _close(fit.df_prior, expected["dfPrior"])
    _close(fit.s2_prior, expected["s2Prior"])
    if "tStatistic" in expected:
        assert fit.method == "t_test"
        _close(table["t_statistic"], expected["tStatistic"])
        _close(table["prop_ratio"], expected["propRatio"])
    else:
        assert fit.method == "anova"
        _close(table["f_statistic"], expected["fStatistic"])
    _close(table["p_value"], expected["pValue"])
    _close(table["fdr"], expected["fdr"])
