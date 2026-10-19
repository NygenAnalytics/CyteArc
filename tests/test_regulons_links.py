import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

from cytearc.regulons.links import GBM_PARAMETERS, region_gene_links


def _data():
    rng = np.random.default_rng(0)
    n = 60
    access = rng.gamma(1.0, 1.0, size=(n, 6))
    access[:, 5] = 2.0  # a constant region
    expression = np.column_stack(
        [access[:, 0] * 3 + rng.normal(0, 0.3, n), rng.normal(0, 1, n), np.ones(n)]
    )
    access = pd.DataFrame(access, columns=[f"r{i}" for i in range(6)])
    expression = pd.DataFrame(expression, columns=["A", "B", "C"])
    candidates = pd.DataFrame(
        {
            "region": ["r0", "r1", "r2", "r5", "r3", "r4", "r0", "missing"],
            "gene": ["A", "A", "A", "A", "B", "B", "C", "A"],
        }
    )
    return access, expression, candidates


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_region_gene_links_match_scipy_and_sklearn():
    access, expression, candidates = _data()

    links = region_gene_links(access, expression, candidates, nthreads=2)

    assert links[["region", "gene"]].values.tolist() == [
        ["r0", "A"],
        ["r1", "A"],
        ["r2", "A"],
        ["r5", "A"],
        ["r3", "B"],
        ["r4", "B"],
        ["r0", "C"],
    ]
    for region, gene, rho in links[["region", "gene", "rho"]].itertuples(index=False):
        expected = spearmanr(access[region], expression[gene]).statistic
        if np.isnan(expected):
            assert np.isnan(rho)
        else:
            assert rho == pytest.approx(expected, abs=1e-12)
    gene_a = links[links["gene"] == "A"]
    regressor = GradientBoostingRegressor(random_state=666, **GBM_PARAMETERS)
    regressor.fit(access[["r0", "r1", "r2", "r5"]].to_numpy(), expression["A"])
    np.testing.assert_allclose(gene_a["importance"], regressor.feature_importances_)
    assert gene_a.set_index("region").loc["r0", "importance"] > 0.5
    # A constant gene has no importance.
    assert (links[links["gene"] == "C"]["importance"] == 0).all()


def test_region_gene_links_without_importance():
    access, expression, candidates = _data()

    links = region_gene_links(access, expression, candidates, importance="none")

    assert links["importance"].isna().all()
    assert links.loc[0, "rho"] > 0.8
    with pytest.raises(ValueError, match="unique"):
        region_gene_links(
            access.set_axis(["r0"] * 6, axis=1),
            expression,
            candidates,
            importance="none",
        )
    # Both profiles must list the same samples in the same order.
    with pytest.raises(ValueError, match="same samples"):
        region_gene_links(access, expression.iloc[::-1], candidates)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "accessibility": pd.DataFrame({"r0": [1.0, 2.0]}),
                "expression": pd.DataFrame({"A": [1.0, 2.0]}),
            },
            "three samples",
        ),
        ({"importance": "rf"}, "'gbm' or 'none'"),
        ({"candidates": pd.DataFrame({"region": ["r0"]})}, "region and gene columns"),
    ],
)
def test_region_gene_links_rejects_invalid_requests(kwargs, message):
    access, expression, candidates = _data()
    arguments = {
        "accessibility": access,
        "expression": expression,
        "candidates": candidates,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        region_gene_links(**arguments)


@pytest.mark.parametrize("nthreads", [1, 2])
def test_region_gene_links_report_progress_per_gene(nthreads):
    access, expression, candidates = _data()
    seen = []

    region_gene_links(
        access,
        expression,
        candidates,
        importance="none",
        nthreads=nthreads,
        progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_region_gene_links_without_measured_candidates_are_empty():
    access, expression, _ = _data()

    links = region_gene_links(
        access, expression, pd.DataFrame({"region": ["missing"], "gene": ["A"]})
    )

    assert links.empty
    assert list(links.columns) == ["region", "gene", "importance", "rho"]
