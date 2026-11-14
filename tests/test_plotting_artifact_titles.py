"""Plots title a label artifact by its name, or by its own record without one."""

import json
import warnings
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cytearc.plotting as splt
from cytearc.plotting import NamedCellArtifact
from cytearc.plotting._titles import artifact_title, artifact_titles, kind_title
from tests.clone_helpers import write_selection_embedding
from tests.storage_helpers import open_store, write_count_store

LEIDEN = "Leiden, resolution 1.0"


@pytest.fixture(scope="module")
def titled(tmp_path_factory) -> SimpleNamespace:
    """Two snapshots of columns, two Leiden clusterings at one resolution, clones and a layout."""
    path = tmp_path_factory.mktemp("artifact_titles") / "store.zarr"
    rng = np.random.default_rng(3)
    blocks = np.repeat(np.arange(3), 20)
    rates = np.ones((3, 12))
    for block in range(3):
        rates[block, 4 * block : 4 * block + 4] = 3.0
    write_count_store(str(path), {"RNA": rng.poisson(rates[blocks])}, "uint16")
    store = open_store(path)
    store.cells.insert("cell_type", np.repeat(["A", "B", "C"], 20).astype(object))
    store.cells.insert("donor", np.asarray(["d1", "d2", "d3"] * 20, dtype=object))
    store.cells.insert(
        "lineage", np.asarray([f"l{i % 7}" for i in range(60)], dtype=object)
    )
    store.cells.insert("every_cell", np.ones(60, dtype=bool))
    cells = store.snapshot_cell_selection("every_cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalized = store.features.normalize(
            cells, store.features.universe(from_assay="RNA")
        )
        index = store.graph.ann_index(store.reduction.pca(normalized, dims=3))
        graphs = [
            store.graph.connectivity(store.graph.neighbors(index, k=k)) for k in (5, 8)
        ]
        leiden = [store.clusters.leiden(graph, resolution=1.0) for graph in graphs]
    return SimpleNamespace(
        store=store,
        cell_types=store.snapshot_cluster_labels("cell_type", cell_selection=cells),
        donors=store.snapshot_cluster_labels("donor", cell_selection=cells),
        layout=write_selection_embedding(store, cells, rng.normal(size=(60, 2))),
        clones=store.clones.from_column("lineage", cell_selection=cells),
        graph=graphs[0],
        leiden=leiden,
    )


def _legend_titles(result):
    figure = result.figure
    legends = [*figure.legends, *(ax.get_legend() for ax in figure.axes)]
    return [legend.get_title().get_text() for legend in legends if legend]


def test_an_artifact_without_a_name_takes_a_title_from_its_record(titled, tmp_path):
    from tests.test_plotting_foundation import _imported_plot_store

    store = titled.store
    # A snapshot is titled by the column it copied.
    assert artifact_title(store, titled.cell_types) == "cell_type"
    # A Leiden clustering by its method and resolution.
    assert artifact_title(store, titled.leiden[0]) == LEIDEN
    # Another artifact by its kind in words.
    assert artifact_title(store, titled.clones) == "Clones"
    assert kind_title("cluster_labels") == "Cluster labels"
    assert kind_title("hto_identity") == "HTO identity"
    # A store without a record of the artifact gives its kind.
    assert artifact_title(SimpleNamespace(), titled.cell_types) == "Cluster labels"
    # An H5AD import is titled by the obs key it read.
    x = np.arange(6, dtype=np.float64)
    imported_store, imported, _counts = _imported_plot_store(
        tmp_path, coordinates=np.column_stack((x, x % 3)), clusters=[1, 2, 1, 2, 1, 2]
    )
    assert artifact_title(imported_store, imported.clusterArtifacts["clusters"]) == (
        "clusters"
    )


def test_distinct_artifacts_of_one_title_get_distinct_titles_and_panels(titled):
    first, second = titled.leiden
    suffixed = [f"{LEIDEN} ({ref.artifact_id[:6]})" for ref in (first, second)]
    assert artifact_titles(titled.store, [(first, None), (second, None)]) == suffixed
    # One artifact drawn twice keeps one title, and a name is used as given.
    assert artifact_titles(
        titled.store, [(first, None), (first, None), (second, "Mine")]
    ) == [LEIDEN, LEIDEN, "Mine"]
    result = splt.embedding(
        titled.store, layout=titled.layout, color_by=[first, second], show=False
    )
    try:
        assert list(result.axes) == suffixed
        assert [result.axes[key].get_title() for key in suffixed] == suffixed
        assert _legend_titles(result) == suffixed
    finally:
        result.close()
    # Two named artifacts of one kind keep their names and their own panels.
    named = splt.embedding(
        titled.store,
        layout=titled.layout,
        color_by=[
            NamedCellArtifact("Cell type", titled.cell_types),
            NamedCellArtifact("Donor", titled.donors),
        ],
        show=False,
    )
    try:
        assert list(named.axes) == ["Cell type", "Donor"]
        assert [ax.get_title() for ax in named.axes.values()] == ["Cell type", "Donor"]
        assert _legend_titles(named) == ["Cell type", "Donor"]
    finally:
        named.close()


def test_every_plot_titles_a_named_artifact_by_its_name(titled):
    store, cell_type = titled.store, NamedCellArtifact("Cell type", titled.cell_types)
    try:
        distribution = splt.distribution(
            store, "RNA0", grouping=cell_type, max_points=0, show=False
        )
        assert distribution.axes["RNA0"].get_xlabel() == "Cell type"

        composition = splt.composition(
            store,
            categories=cell_type,
            grouping=NamedCellArtifact("Donor", titled.donors),
            show=False,
        )
        assert composition.axes["composition"].get_xlabel() == "Donor"
        assert _legend_titles(composition) == ["Cell type"]

        dots = splt.dotplot(
            store, features=["RNA0", "RNA4"], groups=cell_type, show=False
        )
        assert dots.axes["dotplot"].get_xlabel() == "Cell type"
        matrix = splt.matrixplot(
            store, features=["RNA0", "RNA4"], groups=cell_type, show=False
        )
        assert matrix.tables["matrix"].columns.tolist() == ["A", "B", "C"]

        connectivity = splt.cluster_connectivity(
            store,
            groups=cell_type,
            layout=titled.layout,
            graph=titled.graph,
            show=False,
        )
        assert connectivity.legends[0].label == "Cell type"

        abundance = splt.clone_abundance(
            store, titled.clones, group_by="donor", groups=cell_type, show=False
        )
        assert abundance.legends[0].label == "donor, Cell type"
        # Bare artifacts take the titles of their records.
        bare = splt.composition(
            store, categories=titled.cell_types, grouping=titled.donors, show=False
        )
        assert bare.axes["composition"].get_xlabel() == "donor"
        assert _legend_titles(bare) == ["cell_type"]
        bare_dots = splt.dotplot(
            store, features=["RNA0"], groups=titled.leiden[0], show=False
        )
        assert bare_dots.axes["dotplot"].get_xlabel() == LEIDEN
    finally:
        plt.close("all")


def test_a_summarys_figure_names_the_summarys_artifact(titled):
    store, clones = titled.store, titled.clones
    cell_type = NamedCellArtifact("Cell type", titled.cell_types)
    diversity = store.clones.diversity(clones, group_by="donor", groups=cell_type)
    # The record keeps the bare artifact and names its column by the kind.
    assert diversity.groups == titled.cell_types
    assert diversity.group_keys.columns.tolist() == ["donor", "cluster_labels"]
    try:
        named = splt.clonal_diversity(store, diversity, groups=cell_type, show=False)
        assert _legend_titles(named) == ["donor, Cell type"]
        bare = splt.clonal_diversity(store, diversity, show=False)
        assert _legend_titles(bare) == ["donor, cell_type"]
        with pytest.raises(
            ValueError, match="groups names the cluster_labels artifact"
        ):
            splt.clonal_diversity(
                store,
                diversity,
                groups=NamedCellArtifact("Donor", titled.donors),
                show=False,
            )
        with pytest.raises(TypeError, match="groups must be a NamedCellArtifact"):
            splt.clonal_diversity(
                store, diversity, groups=titled.cell_types, show=False
            )
        # The expansion plot titles the regrouped cells and its own groups.
        expansion = store.clones.expansion(clones, groups=titled.cell_types)
        regrouped = splt.clonal_expansion(
            store, expansion, groups=cell_type, show=False
        )
        assert regrouped.provenance.extras["grouping"] == "Cell type"
        own = splt.clonal_expansion(store, expansion, show=False)
        assert own.provenance.extras["grouping"] == "cell_type"
    finally:
        plt.close("all")


def test_naming_an_artifact_two_ways_reuses_one_saved_summary(titled):
    store, clones, cell_types = titled.store, titled.clones, titled.cell_types
    bare = store.clones.expansion(clones, groups=cell_types)
    named = store.clones.expansion(
        clones, groups=NamedCellArtifact("Cell type", cell_types)
    )
    renamed = store.clones.expansion(
        clones, groups=NamedCellArtifact("Cluster", cell_types)
    )
    assert named == bare == renamed
    loaded = store.clones.load_expansion(named)
    assert loaded.groups == cell_types
    assert loaded.group_keys.columns.tolist() == ["cluster_labels"]


def test_plot_recipes_name_label_artifacts(titled):
    store = titled.store
    recipe = splt.PlotRecipe.from_json(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "colors",
                        "plot": "embedding",
                        "kwargs": {
                            "colorBy": [
                                {"name": "Cell type", "artifact": "cellTypes"},
                                {"key": "donor"},
                            ]
                        },
                        "artifactKwargs": {"layout": "layout"},
                    },
                    {
                        "name": "violin",
                        "plot": "distribution",
                        "kwargs": {
                            "keys": "RNA0",
                            "grouping": {"name": "Cell type", "artifact": "cellTypes"},
                            "maxPoints": 0,
                        },
                    },
                    {
                        "name": "shares",
                        "plot": "composition",
                        "kwargs": {
                            "categories": {
                                "name": "Cell type",
                                "artifact": "cellTypes",
                            },
                            "grouping": {"name": "Donor", "artifact": "donors"},
                        },
                    },
                    {
                        "name": "dots",
                        "plot": "dotplot",
                        "kwargs": {
                            "features": ["RNA0"],
                            "groups": {"name": "Cell type", "artifact": "cellTypes"},
                        },
                    },
                ]
            }
        )
    )
    artifacts = {
        "cellTypes": titled.cell_types,
        "donors": titled.donors,
        "layout": titled.layout,
    }
    result = splt.run_recipe(store, recipe, artifacts=artifacts)
    try:
        colors, violin, shares, dots = (output.result for output in result.outputs)
        assert list(colors.axes) == ["Cell type", "donor"]
        assert violin.axes["RNA0"].get_xlabel() == "Cell type"
        assert shares.axes["composition"].get_xlabel() == "Donor"
        assert _legend_titles(shares) == ["Cell type"]
        assert dots.axes["dotplot"].get_xlabel() == "Cell type"
    finally:
        plt.close("all")
    # The recipe keeps the names apart from the artifacts the runner supplies.
    with pytest.raises(KeyError, match="cellTypes"):
        splt.run_recipe(store, recipe, artifacts={"layout": titled.layout})


@pytest.mark.parametrize(
    ("grouping", "error", "message"),
    [
        (
            {"name": "Cell type", "artifact": "cellTypes", "kind": "x"},
            ValueError,
            "Unknown grouping key",
        ),
        ({"name": " Cell type", "artifact": "cellTypes"}, ValueError, "whitespace"),
        ({"name": "Cell type", "artifact": 3}, TypeError, "recipe artifact"),
    ],
)
def test_plot_recipes_refuse_malformed_named_artifacts(grouping, error, message):
    with pytest.raises(error, match=message):
        splt.PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "violin",
                        "plot": "distribution",
                        "kwargs": {"keys": "RNA0", "grouping": grouping},
                    }
                ]
            }
        )
