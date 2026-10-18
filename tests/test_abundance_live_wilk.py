"""Live Milo and propeller on the Wilk et al. 2020 COVID-19 PBMC atlas (opt-in).

Set ``CYTEARC_RUN_LIVE_DA=1`` to run. The test mounts the dataset by its
CyteBase URI, which needs network access but not the ``cytebase`` package,
builds the graph with the recipe in ``scripts/export_milo_golden.py``,
and runs ``DataStore.abundance`` on it. Donors are the replicates: 7 COVID-19
and 6 healthy, with the two samples of donor C1 pooled. Setting
``CYTEARC_LIVE_DA_DIR`` keeps the mount there, so later runs reuse the graph.

The comparisons with miloR 2.6.0 and speckle 1.10.0 read a local fixture that
``uv run python scripts/export_milo_golden.py --wilk`` writes to the
git-ignored ``tests/datasets``. It is derived from the dataset and is not
published. miloR tests CyteArc's own kNN graph and refined index cells there,
because the graph comes from an ANN index while miloR's refinement searches
exact neighbours; R's refinement of the same draws is compared separately.
The export takes about 90 minutes. "Live differential-abundance test" in
``docs/source/developers/contributing.md`` lists the run commands, the
comparisons, and their tolerances.
"""

import gzip
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cytearc.metadata.selection import CellField
from scripts.export_milo_golden import (
    WILK,
    WILK_FIXTURE,
    wilk_datastore,
    wilk_fingerprints,
    wilk_inputs,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("CYTEARC_RUN_LIVE_DA") != "1",
        reason="set CYTEARC_RUN_LIVE_DA=1 to run the live Wilk abundance test",
    ),
]

LABELS = CellField(WILK["labelsBy"], kind="categorical")


@cache
def _fixture() -> dict[str, Any]:
    if not WILK_FIXTURE.is_file():
        pytest.skip(
            "No local Wilk fixture; build it with "
            "`uv run python scripts/export_milo_golden.py --wilk`"
        )
    with gzip.open(WILK_FIXTURE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _close(actual: Any, expected: Any, *, rtol: float, atol: float = 0.0) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


@pytest.fixture(scope="module")
def wilk(tmp_path_factory) -> dict[str, Any]:
    pytest.importorskip("httpx")
    pytest.importorskip("edgepython")
    import cytearc

    cytearc.configure_output(level="WARNING", progress=False)
    configured = os.environ.get("CYTEARC_LIVE_DA_DIR")
    directory = Path(configured) if configured else tmp_path_factory.mktemp("wilk_live")
    ds, cytebase_id = wilk_datastore(directory)
    inputs = wilk_inputs(ds)
    neighbors = inputs["refs"]["neighbors"]
    nhoods = ds.abundance.make_neighborhoods(
        neighbors, prop=WILK["prop"], seed=WILK["samplingSeed"]
    )
    counts = ds.abundance.count_neighborhoods(nhoods, sample_by=WILK["sampleBy"])
    tests = ds.abundance.test_neighborhoods(
        counts,
        condition_by=WILK["conditionBy"],
        levels={WILK["conditionBy"]: WILK["conditionLevels"]},
    )
    categories = ds.abundance.test_categories(
        LABELS, sample_by=WILK["sampleBy"], condition_by=WILK["conditionBy"]
    )
    return {
        "ds": ds,
        "cytebase_id": cytebase_id,
        "inputs": inputs,
        "neighborhoods": nhoods,
        "counts": counts,
        "tests": tests,
        "categories": categories,
    }


def test_wilk_design_uses_donors_as_replicates(wilk):
    ds = wilk["ds"]
    counts = ds.abundance.load_neighborhood_counts(wilk["counts"])
    assert len(counts.samples) == 13
    design = wilk["tests"].design
    covid = design[f"{WILK['conditionBy']}COVID-19"]
    assert int(covid.sum()) == 7
    assert int((covid == 0).sum()) == 6
    assert wilk["tests"].contrast == {f"{WILK['conditionBy']}COVID-19": 1.0}
    assert int(counts.counts.sum(axis=1).min()) > 0
    categories = wilk["categories"]
    assert categories.method == "t_test"
    assert categories.groups == ("COVID-19", "normal")
    assert categories.n_cells == ds.cells.N


def test_wilk_reruns_reuse_every_abundance_artifact(wilk):
    ds = wilk["ds"]
    neighbors = wilk["inputs"]["refs"]["neighbors"]
    nhoods = ds.abundance.make_neighborhoods(
        neighbors, prop=WILK["prop"], seed=WILK["samplingSeed"]
    )
    assert nhoods == wilk["neighborhoods"]
    again = ds.abundance.test_neighborhoods(
        wilk["counts"],
        condition_by=WILK["conditionBy"],
        levels={WILK["conditionBy"]: WILK["conditionLevels"]},
    )
    assert again.artifact == wilk["tests"].artifact


def test_wilk_fixture_was_made_from_this_graph(wilk):
    fixture = _fixture()
    provenance = fixture["provenance"]
    assert provenance["packageVersion"] == "2.6.0"
    assert provenance["edgeRVersion"] == "4.8.2"
    assert provenance["speckleVersion"] == "1.10.0"
    assert provenance["published"] is False
    assert provenance["cytebaseId"] == wilk["cytebase_id"]
    assert fixture["recipe"] == WILK
    assert fixture["inputs"]["fingerprints"] == wilk_fingerprints(wilk["inputs"]), (
        "The graph or draws differ from the fixture's; rebuild it with "
        "`uv run python scripts/export_milo_golden.py --wilk`"
    )


def test_wilk_graph_and_refinement_agree_with_milor(wilk):
    expected = _fixture()["milor"]
    assert expected["knnRecall"] > 0.999
    assert all(expected["indexRowExact"])
    native = set(wilk["inputs"]["indexVertices"].tolist())
    milor = set(expected["rRefinedVertices"])
    agreement = len(native & milor) / len(milor)
    assert agreement > 0.99, agreement


def test_wilk_neighborhood_test_matches_milor(wilk):
    ds = wilk["ds"]
    expected = _fixture()["milor"]
    nhoods = ds.abundance.load_neighborhoods(wilk["neighborhoods"])
    np.testing.assert_array_equal(nhoods.index_rows, expected["refinedVertices"])
    np.testing.assert_array_equal(
        nhoods.sizes, [len(members) for members in expected["nhoodMembers"]]
    )
    _close(nhoods.kth_distance, expected["kDistance"], rtol=1e-5)
    counts = ds.abundance.load_neighborhood_counts(wilk["counts"])
    assert list(counts.samples) == expected["sampleOrder"]
    np.testing.assert_array_equal(counts.counts, expected["nhoodCounts"])
    result = wilk["tests"]
    assert list(result.design.columns) == [
        column.replace("condition", WILK["conditionBy"])
        for column in expected["designColumns"]
    ]
    table = result.table
    # On this dataset: p-values and spatial FDR within 1.1e-6 relative of
    # miloR, logFC within 4.2e-5, and the same 931 neighbourhoods called at
    # spatial FDR < 0.1.
    _close(result.norm_factors, expected["normFactors"], rtol=1e-12)
    _close(table["log_cpm"], expected["logCpm"], rtol=1e-8)
    _close(table["log_fc"], expected["logFc"], rtol=0, atol=1e-3)
    _close(table["p_value"], expected["pValue"], rtol=1e-5)
    _close(table["spatial_fdr"], expected["spatialFdr"], rtol=1e-5)
    np.testing.assert_array_equal(
        table["spatial_fdr"].to_numpy() < 0.1,
        np.asarray(expected["spatialFdr"], dtype=np.float64) < 0.1,
    )


def test_wilk_annotation_matches_milor(wilk):
    expected = _fixture()["milor"]
    annotation = wilk["ds"].abundance.annotate_neighborhoods(
        wilk["tests"].artifact, LABELS
    )
    assert list(annotation["label"]) == expected["labelMajority"]
    _close(annotation["fraction"], expected["labelFraction"], rtol=1e-14)


def test_wilk_propeller_matches_speckle(wilk):
    expected = _fixture()["milor"]["propeller"]
    table = wilk["categories"].table.loc[expected["categories"]]
    _close(table["t_statistic"], expected["tStatistic"], rtol=1e-8)
    _close(table["prop_ratio"], expected["propRatio"], rtol=1e-10)
    _close(table["p_value"], expected["pValue"], rtol=1e-8)
    _close(table["fdr"], expected["fdr"], rtol=1e-8)


def test_wilk_abundance_plots_render(wilk):
    ds = wilk["ds"]
    layout = ds.artifacts.find("embedding", imported="X_umap", from_assay="RNA")
    for plot in (
        ds.plots.category_proportions(wilk["categories"], show=False),
        ds.plots.neighborhood_beeswarm(
            wilk["tests"], group_by=LABELS, min_fraction=0.7, show=False
        ),
        ds.plots.neighborhood_graph(wilk["tests"], layout=layout, show=False),
    ):
        try:
            assert plot.tables
        finally:
            plot.close()
