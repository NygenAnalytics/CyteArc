"""Plots of imported cluster labels and of frozen run columns keep their stored categories."""

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
    one of them unused, and an unordered one whose natural order differs
    from its stored order.
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
        pd.Categorical(
            np.array(["late", "early", "mid"])[blocks],
            categories=["early", "mid", "late", "unused"],
            ordered=True,
        ),
    )
    store.cells.insert(
        "cluster_codes",
        pd.Categorical(np.array(["10", "2", "1"])[blocks], categories=["10", "2", "1"]),
    )
    return SimpleNamespace(
        store=store,
        layout=result.embeddingArtifacts["X_umap"],
        clusters=result.clusterArtifacts["clusters"],
    )


@pytest.fixture(scope="module")
def frozen_run(imported) -> SimpleNamespace:
    """A run that freezes the imported, ordered, and unordered columns."""
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
            snapshot_columns=("cell_type", "stage", "cluster_codes"),
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
        (points,) = embedding.axes["cluster_labels"].collections
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


def test_only_a_store_with_a_root_reads_an_artifacts_categories(imported):
    from cytearc.plotting._display import stored_artifact_scale

    stored = stored_artifact_scale(imported.store, imported.clusters)
    assert stored.order == tuple(FILE_ORDER)
    assert stored_artifact_scale(SimpleNamespace(), imported.clusters) is None


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
def test_a_run_export_writes_frozen_columns_as_before(frozen_run):
    # The display of a frozen column serves plots; the run export still
    # writes the column as other frozen text, and gives colors only to the
    # run's own label fields.
    exported = frozen_run.store.to_anndata(run=frozen_run.run)
    assert exported.obs["cell_type"].cat.categories.tolist() == ["B", "NK", "T"]
    assert exported.obs["stage"].cat.categories.tolist() == ["early", "late", "mid"]
    assert "cell_type_colors" not in exported.uns
    assert "stage_colors" not in exported.uns
    assert "clusters_colors" in exported.uns
