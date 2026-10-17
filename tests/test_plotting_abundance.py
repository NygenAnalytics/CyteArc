"""Differential-abundance plots read saved propeller and Milo tests."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc.embeddings.imported import write_imported_embedding
from cytearc.metadata.selection import CellField
from cytearc.storage.artifacts import fingerprint_array
from tests.test_milo_golden import _build_store, _case

pytest.importorskip("matplotlib")

LABEL = CellField("label", kind="categorical")


@pytest.fixture(scope="module")
def milo_store(tmp_path_factory) -> dict[str, Any]:
    case, _ = _case("condition_refined")
    store, neighbors = _build_store(tmp_path_factory.mktemp("milo_plots"), case)
    nhoods = store.abundance.make_neighborhoods(
        neighbors, prop=case["prop"], seed=case["samplingSeed"]
    )
    counts = store.abundance.count_neighborhoods(nhoods, sample_by="sample")
    tests = store.abundance.test_neighborhoods(counts, condition_by="condition")
    layout_values = np.asarray(case["coordinates"], dtype=np.float32)[:, :2].copy()
    cell_ids = np.asarray([f"cell{idx:04d}" for idx in range(len(layout_values))])
    layout = write_imported_embedding(
        store.zw,
        assay="RNA",
        dimreduc_key="umap",
        role="umap",
        coordinates=layout_values,
        source_digest=hashlib.sha256(layout_values.tobytes()).digest(),
        payload_fingerprints={"values": fingerprint_array(layout_values)},
        source_cell_ids=cell_ids,
        cell_selection=store.snapshot_cell_selection("I"),
    )
    categories = store.abundance.test_categories(
        LABEL, sample_by="sample", condition_by="condition"
    )
    return {
        "store": store,
        "tests": tests,
        "layout": layout,
        "layout_values": layout_values.astype(np.float64),
        "categories": categories,
    }


def test_category_proportions_tables_follow_the_saved_test(milo_store):
    store = milo_store["store"]
    result = milo_store["categories"]
    plot = store.plots.category_proportions(result.artifact, show=False)
    try:
        proportions = plot.tables["proportions"]
        expected = result.proportions()
        assert list(plot.axes) == list(result.fit.categories)
        for row in proportions.itertuples():
            assert row.proportion == expected.loc[row.category, row.sample]
            assert row.group == result.sample_groups()[row.sample]
        transformed = result.proportions(transformed=True)
        np.testing.assert_allclose(
            proportions["transformed"],
            [
                transformed.loc[category, sample]
                for category, sample in zip(
                    proportions["category"], proportions["sample"], strict=True
                )
            ],
        )
        tests = plot.tables["tests"].set_index("category")
        pd.testing.assert_series_equal(tests["fdr"], result.table["fdr"])
        first = result.fit.categories[0]
        assert plot.axes[first].get_title().startswith(f"{first}\nFDR = ")
        assert plot.scales[0].order == result.groups
        assert plot.provenance.n_samples == len(result.fit.samples)
        assert plot.provenance.extras["category_tests"] == result.artifact.to_dict()
    finally:
        plot.close()


def test_category_proportions_selects_categories_and_transformed_values(milo_store):
    store = milo_store["store"]
    result = milo_store["categories"]
    chosen = list(reversed(result.fit.categories))[:2]
    plot = store.plots.category_proportions(
        result, categories=chosen, transformed=True, show=False
    )
    try:
        assert list(plot.axes) == chosen
        assert set(plot.tables["proportions"]["category"]) == set(chosen)
        assert plot.axes[chosen[0]].get_ylabel() == "Logit proportion"
    finally:
        plot.close()
    with pytest.raises(KeyError, match="Unknown categories"):
        store.plots.category_proportions(result, categories=["absent"], show=False)


def test_neighborhood_beeswarm_colors_spatially_significant_neighborhoods(milo_store):
    store = milo_store["store"]
    result = milo_store["tests"]
    alpha = 0.5
    plot = store.plots.neighborhood_beeswarm(
        result.artifact, group_by=LABEL, alpha=alpha, show=False
    )
    try:
        plotted = plot.tables["neighborhoods"]
        annotation = store.abundance.annotate_neighborhoods(result.artifact, LABEL)
        tested = result.table["tested"].to_numpy(dtype=bool)
        assert len(plotted) == int(tested.sum())
        np.testing.assert_array_equal(
            plotted["group"], annotation["label"].to_numpy()[tested]
        )
        spatial_fdr = result.table["spatial_fdr"].to_numpy()[tested]
        np.testing.assert_array_equal(plotted["significant"], spatial_fdr < alpha)
        groups = plot.provenance.extras["groups"]
        rows = plotted["group"].map(
            {group: len(groups) - 1 - index for index, group in enumerate(groups)}
        )
        assert np.all(np.abs(plotted["y"] - rows) <= 0.4 + 1e-12)
        np.testing.assert_array_equal(
            plotted["log_fc"], result.table["log_fc"].to_numpy()[tested]
        )
        limits = plot.legends[0].extras
        assert limits["vmin"] == -limits["vmax"]
        assert plot.provenance.extras["n_significant"] == int(
            plotted["significant"].sum()
        )
    finally:
        plot.close()


def test_neighborhood_beeswarm_marks_mixed_neighborhoods(milo_store):
    store = milo_store["store"]
    result = milo_store["tests"]
    plot = store.plots.neighborhood_beeswarm(
        result, group_by=LABEL, min_fraction=1.0, mixed_label="mixed", show=False
    )
    try:
        plotted = plot.tables["neighborhoods"]
        pure = plotted["label_fraction"] >= 1.0
        assert (plotted.loc[~pure, "group"] == "mixed").all()
        assert (plotted.loc[pure, "group"] != "mixed").all()
        groups = plot.provenance.extras["groups"]
        if "mixed" in groups:
            assert groups[-1] == "mixed"
    finally:
        plot.close()
    plot = store.plots.neighborhood_beeswarm(result, show=False)
    try:
        assert set(plot.tables["neighborhoods"]["group"]) == {"all neighbourhoods"}
    finally:
        plot.close()
    with pytest.raises(ValueError, match="min_fraction requires group_by"):
        store.plots.neighborhood_beeswarm(result, min_fraction=0.5, show=False)
    with pytest.raises(ValueError, match="alpha"):
        store.plots.neighborhood_beeswarm(result, alpha=0.0, show=False)


def test_neighborhood_graph_places_index_cells_on_the_layout(milo_store):
    store = milo_store["store"]
    result = milo_store["tests"]
    layout_values = milo_store["layout_values"]
    plot = store.plots.neighborhood_graph(
        result.artifact, layout=milo_store["layout"], alpha=0.5, show=False
    )
    try:
        table = plot.tables["neighborhoods"]
        index_cells = result.table["index_cell"].to_numpy()
        np.testing.assert_array_equal(table["x"], layout_values[index_cells, 0])
        np.testing.assert_array_equal(table["y"], layout_values[index_cells, 1])
        np.testing.assert_array_equal(table["size"], result.table["size"])
        size_scale = plot.scales[1]
        assert size_scale.vmin == result.table["size"].min()
        assert size_scale.vmax == result.table["size"].max()
        assert plot.legends[1].kind == "size"
        assert plot.provenance.extras["layout"] == milo_store["layout"].to_dict()
    finally:
        plot.close()
    with pytest.raises(TypeError, match="layout"):
        store.plots.neighborhood_graph(result, layout="umap", show=False)


def test_abundance_plots_reject_unrelated_results(milo_store):
    store = milo_store["store"]
    with pytest.raises(TypeError, match="NeighborhoodTestResult"):
        store.plots.neighborhood_beeswarm(milo_store["categories"], show=False)
    with pytest.raises(TypeError, match="CategoryTestResult"):
        store.plots.category_proportions(milo_store["tests"], show=False)


def test_quasirandom_offsets_are_deterministic_and_bounded():
    from cytearc.plotting.abundance import _quasirandom_offsets

    values = np.random.default_rng(0).normal(size=200)
    first = _quasirandom_offsets(values, width=0.4)
    np.testing.assert_array_equal(first, _quasirandom_offsets(values, width=0.4))
    assert np.abs(first).max() <= 0.4
    np.testing.assert_array_equal(_quasirandom_offsets(np.ones(3), 0.4).shape, (3,))
    assert _quasirandom_offsets(np.zeros(1), 0.4).tolist() == [0.0]


def test_plot_module_does_not_need_a_datastore_path(tmp_path: Path):
    from cytearc.plotting import abundance

    assert "datastore" not in abundance.__dict__
