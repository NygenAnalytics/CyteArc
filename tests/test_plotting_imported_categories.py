"""Plots and exports of imported cluster labels, of label snapshots and of frozen run columns keep their stored categories."""

import shutil
import warnings
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import cytearc
import cytearc.plotting as splt
from tests.storage_helpers import by_cell_id

anndata = pytest.importorskip("anndata")

# The file's categories, neither sorted nor in CyteArc's colors.
FILE_ORDER = ["T", "B", "NK"]
FILE_COLORS = ["#ff0000", "#00aa00", "#0000ff"]
FILE_LEGEND = list(zip(FILE_ORDER, FILE_COLORS, strict=True))
# The first four colors of CyteArc's default palette, as plots assign them in
# order.
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#279e68", "#d62728"]


@pytest.fixture(scope="module")
def imported(tmp_path_factory) -> SimpleNamespace:
    """One categorical imported as an obs column and as cluster labels.

    The store also holds a categorical column whose categories are ordered,
    one of them unused, an unordered one whose natural order differs from
    its stored order, and ordered integer categories out of natural order.
    """
    folder = tmp_path_factory.mktemp("imported_categories")
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(3), 60)
    rates = np.ones((3, 90))
    for block in range(3):
        rates[block, 30 * block : 30 * block + 30] = 8.0
    labels = pd.Categorical(np.array(FILE_ORDER)[blocks], categories=FILE_ORDER)
    adata = anndata.AnnData(
        X=rng.poisson(rates[blocks]).astype(np.float32),
        obs=pd.DataFrame(
            {"cell_type": labels, "clusters": labels},
            index=[f"c{index}" for index in range(len(blocks))],
        ),
        var=pd.DataFrame(index=[f"g{index}" for index in range(90)]),
    )
    adata.uns["cell_type_colors"] = np.array(FILE_COLORS)
    adata.uns["clusters_colors"] = np.array(FILE_COLORS)
    adata.obsm["X_umap"] = rng.normal(size=(len(blocks), 2))
    adata.write_h5ad(folder / "cells.h5ad")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = cytearc.H5adReader.from_inspect(
            cytearc.inspect_h5ad(str(folder / "cells.h5ad")),
            embedding_roles={"X_umap": "umap"},
            cluster_keys=("clusters",),
        )
        try:
            result = cytearc.H5adToZarr(
                reader,
                zarr_loc=str(folder / "cells.zarr"),
                analysis_assay="RNA",
                nthreads=1,
            ).dump()
        finally:
            reader.h5.close()
    store = cytearc.DataStore(
        str(folder / "cells.zarr"), min_features_per_cell=0, zarr_mode="r+"
    )
    store.cells.insert(
        "stage",
        by_cell_id(
            store.cells,
            pd.Categorical(
                np.array(["late", "early", "mid"])[blocks],
                categories=["early", "mid", "late", "unused"],
                ordered=True,
            ),
        ),
    )
    store.cells.insert(
        "cluster_codes",
        by_cell_id(
            store.cells,
            pd.Categorical(
                np.array(["10", "2", "1"])[blocks], categories=["10", "2", "1"]
            ),
        ),
    )
    store.cells.insert(
        "grade",
        by_cell_id(
            store.cells,
            pd.Categorical(
                np.array([2, 3, 1])[blocks], categories=[3, 1, 2], ordered=True
            ),
        ),
    )
    return SimpleNamespace(
        store=store,
        layout=result.embeddingArtifacts["X_umap"],
        clusters=result.clusterArtifacts["clusters"],
    )


@pytest.fixture(scope="module")
def frozen_run(imported) -> SimpleNamespace:
    """A run that freezes the imported, ordered, unordered, and integer columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = imported.store.pipeline.rna(
            label="frozen",
            filtering=False,
            cell_cycle=False,
            hvg={"top_n": 60},
            pca={"dims": 5},
            neighbors={"k": 8},
            umap={"n_epochs": 30},
            leiden={"partitions": [1.0]},
            doublets=False,
            markers=False,
            snapshot_columns=("cell_type", "stage", "cluster_codes", "grade"),
        )
    return SimpleNamespace(**vars(imported), run=run)


def _legend(result):
    """The text and color of each entry of a figure's side legend."""
    legend = result.figure.legends[0]
    entries = [
        (
            text.get_text(),
            to_hex(
                handle.get_markerfacecolor()
                if hasattr(handle, "get_markerfacecolor")
                else handle.get_facecolor()
            ),
        )
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    ]
    result.close()
    return entries


def test_plots_of_imported_cluster_labels_draw_the_files_order_and_colors(imported):
    from cytearc.plotting._groups import _colors as group_colors

    store, layout, clusters = imported.store, imported.layout, imported.clusters
    try:
        embedding = store.plots.embedding(
            layout=layout, color_by=clusters, legend_loc="right", show=False
        )
        # Each cell, found by its coordinates, takes its label's color from
        # the file.
        (points,) = embedding.axes["clusters"].collections
        coordinates = np.asarray(store.artifacts.load_values(layout).values)
        labels = np.asarray(store.artifacts.load_values(clusters).values)
        row = {tuple(point): index for index, point in enumerate(coordinates)}
        drawn = [row[tuple(offset)] for offset in np.asarray(points.get_offsets())]
        assert sorted(drawn) == list(range(len(labels)))
        assert [to_hex(color) for color in points.get_facecolors()] == [
            dict(FILE_LEGEND)[labels[cell]] for cell in drawn
        ]
        assert _legend(embedding) == FILE_LEGEND

        assert _legend(store.plots.composition(categories=clusters, show=False)) == (
            FILE_LEGEND
        )

        violin = store.plots.distribution(
            "RNA_nCounts", grouping=clusters, max_points=0, show=False
        )
        assert [
            tick.get_text() for tick in violin.axes["RNA_nCounts"].get_xticklabels()
        ] == FILE_ORDER
        assert {
            value: to_hex(color) for value, color in violin.scales[0].palette.items()
        } == dict(FILE_LEGEND)
        violin.close()

        dots = store.plots.dotplot(features=["g0"], groups=clusters, show=False)
        assert [
            tick.get_text() for tick in dots.axes["dotplot"].get_xticklabels()
        ] == FILE_ORDER
        dots.close()
        matrix = store.plots.matrixplot(features=["g0"], groups=clusters, show=False)
        assert matrix.tables["matrix"].columns.tolist() == FILE_ORDER
        matrix.close()

        # Clone and variant figures color the groups of the labels the same way.
        groups = group_colors(
            store, ["NK", "T", "B"], [clusters.kind], group_by=(), groups=clusters
        )
        assert list(groups.palette.items()) == FILE_LEGEND

        # A scale of labels alone renames a category and keeps the rest.
        renamed = store.plots.embedding(
            layout=layout,
            color_by=clusters,
            categorical_scale=splt.CategoricalScale(labels={"T": "T cells"}),
            legend_loc="right",
            show=False,
        )
        assert _legend(renamed) == [("T cells", "#ff0000"), *FILE_LEGEND[1:]]
    finally:
        plt.close("all")


def test_label_artifacts_and_exports_read_one_record_of_categories(imported):
    from cytearc.metadata.artifacts import artifact_categories
    from cytearc.writers.export import _stored_labels

    root = imported.store.zw
    recorded = artifact_categories(root, imported.clusters)
    assert recorded.levels == tuple(FILE_ORDER)
    assert recorded.ordered is False
    assert recorded.ordered_levels is None
    assert [category["color"] for category in recorded.display["categories"]] == (
        FILE_COLORS
    )
    exported = _stored_labels(root, imported.clusters, "values")
    assert exported.levels == tuple(FILE_ORDER)
    assert dict(exported.colors) == dict(FILE_LEGEND)


def _exported_categorical(adata, column):
    """The categories, order flag and colors of an exported categorical."""
    values = adata.obs[column]
    colors = adata.uns.get(f"{column}_colors")
    return (
        values.cat.categories.tolist(),
        bool(values.cat.ordered),
        None if colors is None else list(colors),
    )


def test_snapshots_draw_and_export_the_categories_of_their_sources(imported):
    store, layout = imported.store, imported.layout
    cells = store.snapshot_cell_selection()
    sources = {
        "cell_type": "cell_type",
        "stage": "stage",
        "cluster_codes": "cluster_codes",
        "grade": "grade",
        "clusters": imported.clusters,
    }
    snapshots = {
        name: store.snapshot_cluster_labels(source, cell_selection=cells)
        for name, source in sources.items()
    }
    legends = {
        # The file's order and colors.
        "cell_type": FILE_LEGEND,
        "clusters": FILE_LEGEND,
        # The stored order of ordered categories, with the default colors
        # over every stored category.
        "stage": list(zip(["early", "mid", "late"], DEFAULT_COLORS, strict=False)),
        # Unordered categories in natural order, as plots of the column draw
        # them.
        "cluster_codes": list(zip(["1", "2", "10"], DEFAULT_COLORS, strict=False)),
        "grade": list(zip(["3", "1", "2"], DEFAULT_COLORS, strict=False)),
    }
    try:
        for name, legend in legends.items():
            for color_by in (sources[name], snapshots[name]):
                drawn = store.plots.embedding(
                    layout=layout, color_by=color_by, legend_loc="right", show=False
                )
                assert _legend(drawn) == legend, (name, color_by)
    finally:
        plt.close("all")

    exported = store.to_anndata(
        obs={f"{name}_snapshot": ref for name, ref in snapshots.items()}
    )
    categoricals = {
        "cell_type": (FILE_ORDER, False, FILE_COLORS),
        "clusters": (FILE_ORDER, False, FILE_COLORS),
        # Every stored category, unused ones included, and the ordered flag.
        "stage": (["early", "mid", "late", "unused"], True, None),
        "cluster_codes": (["10", "2", "1"], False, None),
        "grade": ([3, 1, 2], True, None),
    }
    live = store.to_anndata()
    for name, categorical in categoricals.items():
        assert _exported_categorical(exported, f"{name}_snapshot") == categorical
        if name != "clusters":
            assert _exported_categorical(live, name) == categorical, name
        # Each cell keeps its label.
        assert exported.obs[f"{name}_snapshot"].astype(str).tolist() == (
            exported.obs["cell_type" if name == "clusters" else name]
            .astype(str)
            .tolist()
        ), name


def test_a_snapshot_of_some_clusters_keeps_every_category_and_color(imported):
    from cytearc.metadata.artifacts import artifact_categories

    store, clusters = imported.store, imported.clusters
    some = store.select_cells(clusters, include=["NK", "T"])
    narrowed = store.snapshot_cluster_labels(clusters, cell_selection=some)

    recorded = artifact_categories(store.zw, narrowed)
    assert recorded.levels == tuple(FILE_ORDER)
    assert [category["color"] for category in recorded.display["categories"]] == (
        FILE_COLORS
    )
    violin = store.plots.distribution(
        "RNA_nCounts", grouping=narrowed, max_points=0, show=False
    )
    try:
        # The file's order, where natural order puts NK first, and its colors.
        ticks = violin.axes["RNA_nCounts"].get_xticklabels()
        assert [tick.get_text() for tick in ticks] == ["T", "NK"]
        palette = {
            value: to_hex(color) for value, color in violin.scales[0].palette.items()
        }
        assert {label: palette[label] for label in ("T", "NK")} == {
            "T": "#ff0000",
            "NK": "#0000ff",
        }
    finally:
        violin.close()
    exported = store.to_anndata(cell_selection=some, obs={"types": narrowed})
    # B, which no exported cell holds, stays a category with its color.
    assert _exported_categorical(exported, "types") == (FILE_ORDER, False, FILE_COLORS)
    assert set(exported.obs["types"].astype(str)) == {"NK", "T"}
    # A snapshot of the snapshot keeps them too.
    t_cells = store.select_cells(narrowed, include=["T"])
    again = store.snapshot_cluster_labels(narrowed, cell_selection=t_cells)
    assert artifact_categories(store.zw, again) == recorded


def test_only_a_store_with_a_root_reads_an_artifacts_categories(imported):
    from cytearc.plotting._display import stored_artifact_scale

    stored = stored_artifact_scale(imported.store, imported.clusters)
    assert stored.order == tuple(FILE_ORDER)
    assert stored_artifact_scale(SimpleNamespace(), imported.clusters) is None


@pytest.fixture(scope="module")
def paris_tree(imported, tmp_path_factory) -> SimpleNamespace:
    """A Paris clustering of the cells of codes 2 and 10, on a copy of the store.

    The copy keeps the graph and the column that selects the cells out of
    the store that the other tests read.
    """
    from tests.fixtures_datastore import _graph_coordinates, build_neighborhood_graph

    path = tmp_path_factory.mktemp("imported_paris") / "cells.zarr"
    shutil.copytree(imported.store.zarr_loc, path)
    store = cytearc.DataStore(str(path), min_features_per_cell=0, zarr_mode="r+")
    codes = np.asarray(store.cells.fetch_all("cluster_codes"))
    store.cells.insert("tree_cells", by_cell_id(store.cells, codes != "1"))
    cells = store.snapshot_cell_selection("tree_cells")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = build_neighborhood_graph(
            store,
            cell_selection=cells,
            features=store.features.detected(cells, min_cells=1),
            dims=5,
            k=10,
        )
        clusters = store.clusters.paris(graph, n_clusters=3)
        initialization = store.embeddings.initialization(
            _graph_coordinates(store, graph)
        )
        layout = store.embeddings.umap(graph, initialization, n_epochs=10)
    return SimpleNamespace(store=store, layout=layout, graph=graph, clusters=clusters)


def test_cluster_trees_color_categories_as_embeddings_do(paris_tree):
    store = paris_tree.store

    def scale_colors(result):
        (scale,) = [
            scale for scale in result.scales if isinstance(scale, splt.CategoricalScale)
        ]
        return [(value, to_hex(color)) for value, color in scale.palette.items()]

    def tree(**options):
        result = store.plots.cluster_tree(
            graph=paris_tree.graph,
            clusters=paris_tree.clusters,
            show_labels=False,
            show=False,
            **options,
        )
        wedges = {
            to_hex(color)
            for collection in result.axes["tree"].collections
            if collection.get_gid() == "cluster-tree-pie"
            for color in collection.get_facecolors()
        }
        colors = scale_colors(result)
        result.close()
        return colors, wedges

    def embedding(color_by):
        result = store.plots.embedding(
            layout=paris_tree.layout, color_by=color_by, show=False
        )
        colors = scale_colors(result)
        result.close()
        return colors

    # The tree holds the cells of codes 2 and 10, whose stages are early and
    # late. The imported cell types take the file's order and colors; the
    # ordered stage takes default colors over every stage in its stored
    # order, early, mid, late and unused; and the unordered codes take
    # default colors over every code in natural order, 1, 2 and 10, though
    # the tree holds no code 1.
    expected = {
        "cell_type": FILE_LEGEND,
        "stage": [("early", "#1f77b4"), ("late", "#279e68")],
        "cluster_codes": [("2", "#ff7f0e"), ("10", "#279e68")],
    }
    for column, colors in expected.items():
        palette, wedges = tree(fill_by_value=column)
        assert palette == colors
        # The pies draw those colors, and an embedding of the same cells
        # colors the column alike.
        assert wedges == {color for _, color in colors}
        assert embedding(column) == colors
    # The Paris groups take the default colors in their order, as an
    # embedding colored by the clustering draws them.
    clusters, _ = tree()
    assert clusters == [(1, "#1f77b4"), (2, "#ff7f0e"), (3, "#279e68")]
    assert embedding(paris_tree.clusters) == clusters


@pytest.mark.slow
def test_run_plots_of_frozen_columns_draw_them_as_plots_of_the_live_columns(
    frozen_run,
):
    store, run, layout = frozen_run.store, frozen_run.run, frozen_run.layout
    expected = {
        # The file's order and colors.
        "cell_type": FILE_LEGEND,
        # The stored order of ordered categories, with the default colors
        # over every stored category.
        "stage": list(zip(["early", "mid", "late"], DEFAULT_COLORS, strict=False)),
        # Unordered categories in natural order.
        "cluster_codes": list(zip(["1", "2", "10"], DEFAULT_COLORS, strict=False)),
    }
    try:
        for column, legend in expected.items():
            live = store.plots.embedding(
                layout=layout, color_by=column, legend_loc="right", show=False
            )
            assert _legend(live) == legend, column
            frozen = store.plots.embedding(
                run=run, color_by=column, legend_loc="right", show=False
            )
            assert _legend(frozen) == legend, column
        # The #319 merge applies to a frozen display too.
        renamed = store.plots.embedding(
            run=run,
            color_by="stage",
            categorical_scale=splt.CategoricalScale(labels={"mid": "middle"}),
            legend_loc="right",
            show=False,
        )
        assert _legend(renamed) == [
            ("early", DEFAULT_COLORS[0]),
            ("middle", DEFAULT_COLORS[1]),
            ("late", DEFAULT_COLORS[2]),
        ]
    finally:
        plt.close("all")


@pytest.mark.slow
def test_a_run_records_each_frozen_columns_display_in_its_field(frozen_run):
    store = frozen_run.store
    # The display persists with the run record, which a reopened run reads.
    cells = store.pipeline.open(label="frozen").cells
    live_display = store.zw["cellData"]["cell_type"].attrs["display"]
    assert cells._field_display("cell_type") == live_display
    stage = cells._field_display("stage")
    assert [category["value"] for category in stage["categories"]] == [
        "early",
        "mid",
        "late",
        "unused",
    ]
    assert [category["color"] for category in stage["categories"]] == DEFAULT_COLORS
    assert cells._field_display("cluster_codes") is None
    # The snapshot keeps only the labels, as releases before this record read it.
    snapshot = store.artifacts.load(cells._descriptor("cell_type").artifact)
    assert "display" not in dict(snapshot["cell_type"].attrs)


@pytest.mark.slow
def test_cluster_connectivity_draws_imported_groups_in_the_files_colors(frozen_run):
    run = frozen_run.run
    result = frozen_run.store.plots.cluster_connectivity(
        groups=frozen_run.clusters,
        layout=run["umap"],
        graph=run["connectivity_map"],
        minimum_edge_weight=0.0,
        show=False,
    )
    try:
        nodes = result.tables["nodes"]
        assert nodes["category"].tolist() == FILE_ORDER
        assert result.scales[0].palette == dict(FILE_LEGEND)
    finally:
        result.close()


@pytest.mark.slow
def test_a_run_export_writes_frozen_columns_in_the_order_and_colors_of_run_plots(
    frozen_run, tmp_path
):
    store, run = frozen_run.store, frozen_run.run
    path = tmp_path / "run.h5ad"
    cytearc.to_h5ad(store.RNA, str(path), run=run)
    columns = ["cell_type", "stage", "cluster_codes", "grade"]
    labels = store.cells.to_pandas_dataframe(columns)
    labels.index = store.cells.fetch_all("ids")
    for adata in (store.to_anndata(run=run), anndata.read_h5ad(path)):
        # The file's order and colors.
        assert _exported_categorical(adata, "cell_type") == (
            FILE_ORDER,
            False,
            FILE_COLORS,
        )
        # Every stored category in its stored order, with the default colors
        # that run plots draw; the run records no ordered flag.
        assert _exported_categorical(adata, "stage") == (
            ["early", "mid", "late", "unused"],
            False,
            DEFAULT_COLORS,
        )
        assert _exported_categorical(adata, "grade") == (
            [3, 1, 2],
            False,
            DEFAULT_COLORS[:3],
        )
        # A column whose run records no display stays in natural order.
        assert _exported_categorical(adata, "cluster_codes") == (
            ["1", "2", "10"],
            False,
            None,
        )
        assert "clusters_colors" in adata.uns
        # Each cell keeps its label.
        for column in columns:
            assert (
                adata.obs[column].astype(str).tolist()
                == labels.loc[adata.obs_names, column].astype(str).tolist()
            ), column


@pytest.mark.slow
def test_a_run_that_records_no_display_exports_its_frozen_columns_as_before(
    frozen_run, monkeypatch
):
    from cytearc.datastore._pipeline import fields

    # Runs saved before CyteArc recorded the displays of frozen columns.
    monkeypatch.setattr(fields, "_frozen_column_display", lambda root, column: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        earlier = frozen_run.store.pipeline.rna(
            label="frozen_without_displays",
            filtering=False,
            cell_cycle=False,
            hvg={"top_n": 60},
            pca={"dims": 5},
            neighbors={"k": 8},
            umap={"n_epochs": 30},
            leiden={"partitions": [1.0]},
            doublets=False,
            markers=False,
            snapshot_columns=("cell_type", "stage", "cluster_codes", "grade"),
        )
    assert earlier.cells._field_display("cell_type") is None

    exported = frozen_run.store.to_anndata(run=earlier)
    assert _exported_categorical(exported, "cell_type") == (
        ["B", "NK", "T"],
        False,
        None,
    )
    assert _exported_categorical(exported, "stage") == (
        ["early", "late", "mid"],
        False,
        None,
    )
    assert "clusters_colors" in exported.uns


def test_label_artifacts_refuse_display_fields_that_the_display_kind_does_not_hold(
    imported,
):
    from cytearc.plotting._display import stored_artifact_scale
    from cytearc.storage.artifacts import artifact_group

    values = artifact_group(imported.store.zw, imported.clusters)["values"]
    display = dict(values.attrs["display"])
    try:
        values.attrs["display"] = {**display, "addedFlag": True}
        with pytest.raises(ValueError, match="has unknown fields"):
            stored_artifact_scale(imported.store, imported.clusters)
    finally:
        values.attrs["display"] = display
    assert stored_artifact_scale(imported.store, imported.clusters).order == tuple(
        FILE_ORDER
    )


def _what_run_plots_and_exports_show(store, run):
    """The legends and color scales of a run's plots, and its exported categoricals."""
    shown = {}
    for column in ("cell_type", "stage", "cluster_codes", "grade", "clusters"):
        shown[column] = _legend(
            store.plots.embedding(
                run=run, color_by=column, legend_loc="right", show=False
            )
        )
    for name, plot in (
        ("embedding", store.plots.embedding),
        ("raster", store.plots.embedding_raster),
    ):
        result = plot(run=run, color_by="umap_1", show=False)
        shown[f"umap_1 {name}"] = result.scales
        result.close()
    adata = store.to_anndata(run=run)
    for column in ("cell_type", "stage", "cluster_codes", "grade", "clusters"):
        shown[f"{column} export"] = _exported_categorical(adata, column)
    return shown


@pytest.mark.slow
def test_run_plots_and_exports_ignore_display_fields_that_a_later_release_adds(
    frozen_run,
):
    store, run = frozen_run.store, frozen_run.run
    group = store.zw[f"pipeline/runs/{run.run_id}"]
    recorded = group.attrs["fields"]

    def with_later_fields(display):
        # Keys that this release does not know, as a later release could add
        # them, one of them a list of the display's values in reverse order.
        if display["kind"] == "continuous":
            return {**display, "addedText": "counts"}
        values = [category["value"] for category in display["categories"]]
        return {**display, "addedFlag": True, "addedList": values[::-1]}

    before = _what_run_plots_and_exports_show(store, run)
    try:
        group.attrs["fields"] = [
            field
            if field["display"] is None
            else {**field, "display": with_later_fields(field["display"])}
            for field in recorded
        ]
        later = store.pipeline.open(label="frozen")
        assert later.cells._field_display("stage")["addedFlag"] is True
        assert later.cells._field_display("umap_1")["addedText"] == "counts"
        after = _what_run_plots_and_exports_show(store, later)
    finally:
        group.attrs["fields"] = recorded
    plt.close("all")
    assert after == before
    # The display's order and colors, unordered, as the run draws and exports
    # them without the added fields.
    assert after["stage"] == list(
        zip(["early", "mid", "late"], DEFAULT_COLORS, strict=False)
    )
    assert after["stage export"] == (
        ["early", "mid", "late", "unused"],
        False,
        DEFAULT_COLORS,
    )
    assert after["cell_type"] == FILE_LEGEND
