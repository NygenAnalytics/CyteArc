"""Plots title a live column by its CellField label, and a label artifact by its name or by its own record."""

import json
import warnings
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cytearc.plotting as splt
from cytearc.plotting import CellField, NamedCellArtifact
from cytearc.plotting._titles import (
    artifact_title,
    artifact_titles,
    kind_title,
    label_column,
)
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import write_selection_embedding
from tests.storage_helpers import by_cell_id, open_store, write_count_store

LEIDEN = "Leiden, resolution 1.0"
_ARTIFACT = ArtifactRef(scope="datastore", kind="cluster_labels", artifact_id="5" * 64)
_GRAPH = ArtifactRef(
    scope="assay", assay="RNA", kind="connectivity_map", artifact_id="1" * 64
)


@pytest.fixture(scope="module")
def titled(tmp_path_factory) -> SimpleNamespace:
    """Two snapshots of columns, two Leiden clusterings at one resolution, clones and a layout.

    The cells also hold a ``batch`` of two values, and six samples, two of
    each subject, one taken before a treatment and one after it.
    """
    path = tmp_path_factory.mktemp("artifact_titles") / "store.zarr"
    rng = np.random.default_rng(3)
    blocks = np.repeat(np.arange(3), 20)
    rates = np.ones((3, 12))
    for block in range(3):
        rates[block, 4 * block : 4 * block + 4] = 3.0
    write_count_store(str(path), {"RNA": rng.poisson(rates[blocks])}, "uint16")
    store = open_store(path)
    store.cells.insert(
        "cell_type",
        by_cell_id(store.cells, np.repeat(["A", "B", "C"], 20).astype(object)),
    )
    store.cells.insert(
        "donor",
        by_cell_id(store.cells, np.asarray(["d1", "d2", "d3"] * 20, dtype=object)),
    )
    store.cells.insert(
        "lineage",
        by_cell_id(
            store.cells, np.asarray([f"l{i % 7}" for i in range(60)], dtype=object)
        ),
    )
    store.cells.insert("every_cell", by_cell_id(store.cells, np.ones(60, dtype=bool)))
    sample = np.arange(60) % 6
    for name, values in {
        "batch": np.where(np.arange(60) % 2 == 0, "b1", "b2"),
        "sample": np.array([f"s{index + 1}" for index in sample]),
        "subject": np.array([f"p{index // 2 + 1}" for index in sample]),
        "treatment": np.array(["pre", "post"])[sample % 2],
    }.items():
        store.cells.insert(name, by_cell_id(store.cells, values.astype(object)))
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


def _ticks(axis):
    return [tick.get_text() for tick in axis.get_xticklabels()]


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
        assert matrix.axes["matrixplot"].get_xlabel() == "Cell type"

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
        bare_matrix = splt.matrixplot(
            store, features=["RNA0"], groups=titled.leiden[0], show=False
        )
        assert bare_matrix.axes["matrixplot"].get_xlabel() == LEIDEN
    finally:
        plt.close("all")


def test_label_column_reads_the_key_and_title_of_a_grouping_argument():
    assert label_column(None, "sample_by") == (None, None)
    assert label_column("donor", "sample_by") == ("donor", "donor")
    assert label_column(CellField("donor"), "sample_by") == ("donor", "donor")
    assert label_column(CellField("donor", label="Donor"), "sample_by") == (
        "donor",
        "Donor",
    )
    # A categorical field is a grouping too, and an empty label titles nothing.
    assert label_column(
        CellField("donor", kind="categorical", label=""), "sample_by"
    ) == ("donor", "donor")
    with pytest.raises(TypeError) as raised:
        label_column(NamedCellArtifact("Donor", _ARTIFACT), "sample_by")
    assert raised.value.args == (
        "sample_by must be a cell column name or a CellField, not NamedCellArtifact",
    )
    with pytest.raises(ValueError) as raised:
        label_column(CellField("donor", kind="continuous"), "sample_by")
    assert raised.value.args == (
        "sample_by groups cells by category, so its CellField cannot have "
        "kind='continuous'; pass kind='categorical' or 'auto'",
    )


def _assert_same_tables(first, second):
    assert list(first.tables) == list(second.tables)
    for name, table in first.tables.items():
        pd.testing.assert_frame_equal(table, second.tables[name])


def test_every_plot_titles_a_live_column_by_its_cell_field_label(titled):
    store = titled.store
    cell_type = CellField("cell_type", label="Cell type")
    try:
        # Dot plots and matrix plots title their group axis by the label, and
        # by the key of a field without one; two columns join their titles.
        for plot, name in ((splt.dotplot, "dotplot"), (splt.matrixplot, "matrixplot")):
            titles = {
                cell_type: "Cell type",
                (cell_type,): "Cell type",
                (cell_type, "donor"): "Cell type / donor",
                ("cell_type", CellField("donor", label="Donor")): "cell_type / Donor",
                CellField("cell_type"): "cell_type",
                "cell_type": "cell_type",
            }
            plain = plot(
                store, features=["RNA0", "RNA4"], group_by="cell_type", show=False
            )
            # Samples named by a CellField weigh cells as the column's name does.
            by_name, by_field = (
                plot(
                    store,
                    features=["RNA0", "RNA4"],
                    group_by="cell_type",
                    sample_by=sample_by,
                    show=False,
                )
                for sample_by in ("sample", CellField("sample", label="Sample"))
            )
            assert "per_sample" in by_field.tables
            _assert_same_tables(by_field, by_name)
            assert by_field.provenance.extras["sample_by"] == "sample"
            assert by_field.provenance.n_samples == 6
            for group_by, title in titles.items():
                result = plot(
                    store, features=["RNA0", "RNA4"], group_by=group_by, show=False
                )
                assert result.axes[name].get_xlabel() == title, group_by
                if title in ("Cell type", "cell_type"):
                    _assert_same_tables(result, plain)
                    assert result.provenance.extras["group_by"] == ["cell_type"]
            plt.close("all")
        # A composition titles its legend and its bar axis by the labels.
        named = store.plots.composition(
            category_by=cell_type,
            sample_by=CellField("donor", label="Donor"),
            show=False,
        )
        plain = store.plots.composition(
            category_by="cell_type", sample_by="donor", show=False
        )
        assert named.axes["composition"].get_xlabel() == "Donor"
        assert _legend_titles(named) == ["Cell type"]
        assert named.legends[0].label == "Cell type"
        assert plain.axes["composition"].get_xlabel() == "donor"
        assert _legend_titles(plain) == ["cell_type"]
        _assert_same_tables(named, plain)
        assert named.provenance.extras == plain.provenance.extras
        # The conditions' legend takes the label of condition_by, and is
        # titled "Condition" by a column's name.
        paired = {
            condition_by: splt.composition(
                store,
                category_by=cell_type,
                sample_by=CellField("sample"),
                subject_by=CellField("subject", label="Subject"),
                condition_by=condition_by,
                kind="per_sample",
                show=False,
            )
            for condition_by in (
                CellField("treatment", label="Treatment"),
                "treatment",
                CellField("treatment"),
            )
        }
        titled_pairs, plain_pairs, unlabeled = paired.values()
        # PlotResult.legends reports the title that the figure draws.
        assert _legend_titles(titled_pairs) == ["Cell type", "Treatment", "Summary"]
        assert titled_pairs.legends[1].label == "Treatment"
        for result in (plain_pairs, unlabeled):
            assert _legend_titles(result) == ["Cell type", "Condition", "Summary"]
            assert result.legends[1].label == "Condition"
            _assert_same_tables(titled_pairs, result)
            assert result.provenance.extras == titled_pairs.provenance.extras
        assert titled_pairs.provenance.extras["subject_by"] == "subject"
        assert titled_pairs.provenance.extras["n_pair_lines"] > 0
        # A study design's condition column replaces condition_by and its label.
        designed = splt.composition(
            store,
            category_by=cell_type,
            condition_by=CellField("treatment", label="Treatment"),
            study_design=splt.StudyDesign(
                sample_by="sample", subject_by="subject", condition_by="treatment"
            ),
            kind="per_sample",
            show=False,
        )
        assert _legend_titles(designed) == ["Cell type", "Condition", "Summary"]
        assert designed.legends[1].label == "Condition"
        _assert_same_tables(designed, titled_pairs)
        # And its sample column replaces sample_by and its label.
        designed_bars = splt.composition(
            store,
            category_by=cell_type,
            sample_by=CellField("donor", label="Donor"),
            study_design=splt.StudyDesign(sample_by="sample"),
            show=False,
        )
        assert designed_bars.axes["composition"].get_xlabel() == "sample"
        assert _ticks(designed_bars.axes["composition"]) == [
            f"s{index}" for index in range(1, 7)
        ]
        # A split violin titles its split by the label.
        split = {
            split_by: splt.distribution(
                store,
                "RNA0",
                grouping=CellField("cell_type"),
                split_by=split_by,
                sample_by=CellField("donor", label="Donor"),
                max_points=0,
                show=False,
            )
            for split_by in (CellField("batch", label="Batch"), "batch")
        }
        named_split, plain_split = split.values()
        assert named_split.axes["RNA0"].get_legend().get_title().get_text() == "Batch"
        assert named_split.legends[0].label == "Batch"
        assert plain_split.axes["RNA0"].get_legend().get_title().get_text() == "batch"
        _assert_same_tables(named_split, plain_split)
        assert named_split.provenance.extras["split_by"] == "batch"
        assert named_split.provenance.extras["sample_by"] == "donor"
    finally:
        plt.close("all")


def test_cluster_connectivity_titles_its_groups_by_a_cell_field_label():
    from tests.test_plotting_cluster_connectivity import _store, _stored_graph_selection

    store = _store()
    with _stored_graph_selection():
        named, plain = (
            splt.cluster_connectivity(
                store,
                group_by=group_by,
                layout_key="layout",
                graph=_GRAPH,
                minimum_edge_weight=0.0,
                show=False,
            )
            for group_by in (CellField("cluster", label="Cluster"), "cluster")
        )
    try:
        assert named.legends[0].label == "Cluster"
        assert plain.legends[0].label == "cluster"
        _assert_same_tables(named, plain)
        assert named.provenance.extras == plain.provenance.extras
        assert named.provenance.extras["group_by"] == "cluster"
        # The column is read by its key.
        assert ("cluster", "I") in store.cells.fetch_calls
    finally:
        plt.close("all")


# A call of each plot that takes a grouping argument, on a store that fails
# if the plot reads it.
_GROUPING_PLOTS = {
    "dotplot": lambda **options: splt.dotplot(
        object(), features=["RNA0"], show=False, **options
    ),
    "matrixplot": lambda **options: splt.matrixplot(
        object(), features=["RNA0"], show=False, **options
    ),
    "composition": lambda **options: splt.composition(object(), show=False, **options),
    "cluster_connectivity": lambda **options: splt.cluster_connectivity(
        object(), graph=_GRAPH, show=False, **options
    ),
    "distribution": lambda **options: splt.distribution(
        object(), "RNA0", show=False, **options
    ),
}
_GROUPING_ARGUMENTS = [
    ("dotplot", "sample_by"),
    ("matrixplot", "sample_by"),
    ("composition", "category_by"),
    ("composition", "sample_by"),
    ("composition", "subject_by"),
    ("composition", "pair_by"),
    ("composition", "condition_by"),
    ("cluster_connectivity", "group_by"),
    ("distribution", "split_by"),
    ("distribution", "sample_by"),
]


@pytest.mark.parametrize(("plot", "argument"), _GROUPING_ARGUMENTS)
@pytest.mark.parametrize("value", [3, NamedCellArtifact("Donor", _ARTIFACT)])
def test_grouping_arguments_refuse_another_type_before_reading_the_store(
    plot, argument, value
):
    with pytest.raises(TypeError) as raised:
        _GROUPING_PLOTS[plot](**{argument: value})
    assert raised.value.args == (
        f"{argument} must be a cell column name or a CellField, "
        f"not {type(value).__name__}",
    )


@pytest.mark.parametrize(("plot", "argument"), _GROUPING_ARGUMENTS)
def test_grouping_arguments_refuse_a_continuous_field(plot, argument):
    with pytest.raises(ValueError) as raised:
        _GROUPING_PLOTS[plot](**{argument: CellField("donor", kind="continuous")})
    assert raised.value.args == (
        f"{argument} groups cells by category, so its CellField cannot have "
        "kind='continuous'; pass kind='categorical' or 'auto'",
    )


def test_composition_refuses_a_sample_column_beside_a_grouping_artifact():
    with pytest.raises(ValueError) as raised:
        splt.composition(
            object(),
            categories=_ARTIFACT,
            grouping=_ARTIFACT,
            sample_by=CellField("donor", label="Donor"),
            show=False,
        )
    assert raised.value.args == (
        "grouping is mutually exclusive with sample_by or StudyDesign",
    )


@pytest.mark.parametrize("plot", ["dotplot", "matrixplot"])
@pytest.mark.parametrize(
    ("group_by", "error", "message"),
    [
        (
            3,
            TypeError,
            "group_by must be a cell column name, a CellField, or a tuple of "
            "them, not int",
        ),
        (
            ("donor", 3),
            TypeError,
            "each column of group_by must be a cell column name or a CellField, "
            "not int",
        ),
        (
            ("donor", None),
            TypeError,
            "each column of group_by must be a cell column name or a CellField, "
            "not None",
        ),
        (
            (CellField("donor", kind="continuous"),),
            ValueError,
            "each column of group_by groups cells by category, so its CellField "
            "cannot have kind='continuous'; pass kind='categorical' or 'auto'",
        ),
    ],
)
def test_summary_plots_refuse_group_by_columns_of_another_type(
    plot, group_by, error, message
):
    with pytest.raises(error) as raised:
        _GROUPING_PLOTS[plot](group_by=group_by)
    assert raised.value.args == (message,)


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
                    # Live columns are named as CellField values.
                    {
                        "name": "column dots",
                        "plot": "matrixplot",
                        "kwargs": {
                            "features": ["RNA0"],
                            "groupBy": [
                                {"key": "cell_type", "label": "Cell type"},
                                "donor",
                            ],
                            "sampleBy": {"key": "sample", "label": "Sample"},
                        },
                    },
                    {
                        "name": "column shares",
                        "plot": "composition",
                        "kwargs": {
                            "categoryBy": {"key": "cell_type", "label": "Cell type"},
                            "sampleBy": {"key": "donor", "label": "Donor"},
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
        colors, violin, shares, dots, column_matrix, column_shares = (
            output.result for output in result.outputs
        )
        assert list(colors.axes) == ["Cell type", "donor"]
        assert violin.axes["RNA0"].get_xlabel() == "Cell type"
        assert shares.axes["composition"].get_xlabel() == "Donor"
        assert _legend_titles(shares) == ["Cell type"]
        assert dots.axes["dotplot"].get_xlabel() == "Cell type"
        assert column_matrix.axes["matrixplot"].get_xlabel() == "Cell type / donor"
        assert column_matrix.provenance.extras["sample_by"] == "sample"
        assert column_shares.axes["composition"].get_xlabel() == "Donor"
        assert _legend_titles(column_shares) == ["Cell type"]
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
