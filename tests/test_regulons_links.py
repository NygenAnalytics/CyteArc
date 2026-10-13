import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

from cytearc.regulons.links import SCENICPLUS_R2G_PARAMETERS, region_gene_links


def _data():
    rng = np.random.default_rng(0)
    n = 60
    access = rng.gamma(1.0, 1.0, size=(n, 6))
    access[:, 5] = 2.0  # a constant region
    expression = np.column_stack(
        [access[:, 0] * 3 + rng.normal(0, 0.3, n), rng.normal(0, 1, n), np.ones(n)]
    )
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
    regions = [f"r{i}" for i in range(6)]

    links = region_gene_links(
        access, regions, expression, ["A", "B", "C"], candidates, nthreads=2
    )

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
        column = int(region[1:])
        expected = spearmanr(
            access[:, column], expression[:, "ABC".index(gene)]
        ).statistic
        if np.isnan(expected):
            assert np.isnan(rho)
        else:
            assert rho == pytest.approx(expected, abs=1e-12)
    gene_a = links[links["gene"] == "A"]
    regressor = GradientBoostingRegressor(random_state=666, **SCENICPLUS_R2G_PARAMETERS)
    regressor.fit(access[:, [0, 1, 2, 5]], expression[:, 0])
    np.testing.assert_allclose(gene_a["importance"], regressor.feature_importances_)
    assert gene_a.set_index("region").loc["r0", "importance"] > 0.5
    # A constant gene has no importance.
    assert (links[links["gene"] == "C"]["importance"] == 0).all()


def test_region_gene_links_without_importance():
    access, expression, candidates = _data()

    links = region_gene_links(
        access,
        [f"r{i}" for i in range(6)],
        expression,
        ["A", "B", "C"],
        candidates,
        importance="none",
    )

    assert links["importance"].isna().all()
    assert links.loc[0, "rho"] > 0.8
