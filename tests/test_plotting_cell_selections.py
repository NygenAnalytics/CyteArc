"""Plots of inputs whose cell selections differ, on a store of 60 cells.

Inputs over another selection of the same cells, which another operation
made, draw as inputs over one selection do. Labels of more cells are read for
the cells of the input they describe. Inputs of other cells are refused with
a message that names the cells to compute them on.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from cytearc.metadata.selection import cell_selection_indices
from cytearc.plotting import (
    cluster_connectivity,
    composition,
    embedding,
    modality_weights,
)
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import write_selection_embedding
from tests.storage_helpers import open_store, write_count_store

pytestmark = pytest.mark.usefixtures("figures_closed")

N_CELLS = 60
# Every tenth cell is left out of the graphs, which hold the 54 kept cells.
LEFT_OUT = np.arange(N_CELLS) % 10 == 0
KEPT = np.flatnonzero(~LEFT_OUT)
# Three blocks of 20 cells, labeled A, B and C, except the cells left out,
# which carry AB: it sorts between A and B, and no kept cell carries it.
CLUSTERS = np.where(LEFT_OUT, "AB", np.repeat(["A", "B", "C"], 20)).astype(object)
SAMPLES = np.asarray(["s1", "s2", "s3"] * 20, dtype=object)


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    """A store of 60 cells, graphs of the kept cells, and layouts of four selections.

    The RNA graph and the WNN graph of RNA and ADT hold the kept cells, which
    two selections hold: "kept", a snapshot of a cell column, and "kept
    again", which select_cells makes. "half" holds the first 30 cells and
    "every" all 60. Each selection has a layout, an imported embedding that
    places the cell at store position p at (p, p % 7), so a drawn point
    names its cell.
    """
    rng = np.random.default_rng(3)
    blocks = np.repeat(np.arange(3), 20)
    rates = np.ones((3, 12))
    for block in range(3):
        rates[block, 4 * block : 4 * block + 4] = 3.0
    counts = {
        "RNA": rng.poisson(rates[blocks]),
        "ADT": rng.poisson(2 * rates[blocks][:, ::2]) + 1,
    }
    path = tmp_path_factory.mktemp("cell_selections") / "store.zarr"
    write_count_store(str(path), counts, "uint16")
    store = open_store(path)
    store.cells.insert("cluster", CLUSTERS)
    store.cells.insert("sample", SAMPLES)
    store.cells.insert("kept", ~LEFT_OUT)
    store.cells.insert(
        "membership", np.where(LEFT_OUT, "left out", "kept").astype(object)
    )
    store.cells.insert("half", np.arange(N_CELLS) < N_CELLS // 2)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    every = store.snapshot_cell_selection("every_cell")
    selections = {
        "kept": store.snapshot_cell_selection("kept"),
        "kept again": store.select_cells(
            store.snapshot_cluster_labels("membership", cell_selection=every),
            include=["kept"],
        ),
        "half": store.snapshot_cell_selection("half"),
        "every": every,
    }
    with warnings.catch_warnings():
        # The few cells of the store draw warnings from the neighbor search.
        warnings.simplefilter("ignore")
        neighbors = []
        for assay in ("RNA", "ADT"):
            normalized = store.features.normalize(
                selections["kept"], store.features.universe(from_assay=assay)
            )
            reduction = store.reduction.pca(normalized, dims=3)
            neighbors.append(
                store.graph.neighbors(store.graph.ann_index(reduction), k=5)
            )
        graph = store.graph.connectivity(neighbors[0])
        wnn = store.integration.modalities(neighbors)
    layouts = {}
    for name, selection in selections.items():
        positions = cell_selection_indices(store.zw, selection)
        layouts[name] = write_selection_embedding(
            store, selection, np.column_stack([positions, positions % 7])
        )
    return SimpleNamespace(
        store=store,
        selections=selections,
        graph=graph,
        wnn=wnn,
        layouts=layouts,
    )


def _labels(cells, column, selection):
    """The labels of a cell column over the cells of a named selection."""
    return cells.store.snapshot_cluster_labels(
        column, cell_selection=cells.selections[selection]
    )


def test_the_two_selections_of_the_kept_cells_are_different_artifacts(cells):
    kept, again = cells.selections["kept"], cells.selections["kept again"]

    assert kept != again
    np.testing.assert_array_equal(cell_selection_indices(cells.store.zw, kept), KEPT)
    np.testing.assert_array_equal(cell_selection_indices(cells.store.zw, again), KEPT)


# Composition: the grouping is read for the cells of categories.


def _per_group(result):
    """The share and count of each category among each group's cells."""
    table = result.tables["per_group"].set_index(["sample", "category"])
    return table[["proportion", "n_cells"]].sort_index()


def _expected_per_group(cells_in):
    """Each sample's share and count of each cluster among some cells, by hand."""
    counts = pd.crosstab(SAMPLES[cells_in], CLUSTERS[cells_in])
    shares = counts.div(counts.sum(axis=1), axis=0)
    table = pd.DataFrame(
        {"proportion": shares.stack(), "n_cells": counts.stack()}
    ).rename_axis(["sample", "category"])
    return table.sort_index()


@pytest.mark.parametrize("grouping", ["kept", "kept again", "every"])
def test_composition_reads_the_grouping_for_the_cells_of_categories(cells, grouping):
    categories = _labels(cells, "cluster", "kept")

    result = composition(
        cells.store,
        categories=categories,
        grouping=_labels(cells, "sample", grouping),
        show=False,
    )

    try:
        assert result.provenance.n_cells == len(KEPT)
        expected = _expected_per_group(KEPT)
        actual = _per_group(result)
        pd.testing.assert_index_equal(actual.index, expected.index)
        np.testing.assert_allclose(actual["proportion"], expected["proportion"])
        np.testing.assert_array_equal(actual["n_cells"], expected["n_cells"])
    finally:
        result.close()


def test_composition_refuses_a_grouping_without_labels_for_some_categories(cells):
    with pytest.raises(ValueError) as raised:
        composition(
            cells.store,
            categories=_labels(cells, "cluster", "kept"),
            grouping=_labels(cells, "sample", "half"),
            show=False,
        )

    assert raised.value.args == (
        "The cluster_labels artifact of grouping holds no label for some cells "
        "of categories; compute it on cells that include those of categories, "
        "such as the cell selection of categories",
    )


def test_composition_draws_a_grouping_beside_a_column_on_the_groupings_cells(
    cells,
):
    result = composition(
        cells.store,
        category_by="cluster",
        grouping=_labels(cells, "sample", "half"),
        show=False,
    )

    try:
        expected = _expected_per_group(np.arange(N_CELLS // 2))
        actual = _per_group(result)
        pd.testing.assert_index_equal(actual.index, expected.index)
        np.testing.assert_allclose(actual["proportion"], expected["proportion"])
    finally:
        result.close()


# Cluster connectivity: the layout holds the graph's cells, and the groups
# are read for them.


def _connectivity(cells, *, groups, layout):
    return cluster_connectivity(
        cells.store,
        graph=cells.graph,
        groups=_labels(cells, "cluster", groups),
        layout=cells.layouts[layout],
        minimum_edge_weight=0.0,
        show=False,
    )


def test_cluster_connectivity_places_each_group_at_its_kept_cells(cells):
    result = _connectivity(cells, groups="kept", layout="kept")

    try:
        nodes = result.tables["nodes"].set_index("category")
        assert nodes.index.tolist() == ["A", "B", "C"]
        for label in ("A", "B", "C"):
            # Each kept cell stands at x equal to its store position.
            positions = KEPT[CLUSTERS[KEPT] == label]
            assert nodes.loc[label, "nCells"] == len(positions)
            assert nodes.loc[label, "x"] == np.median(positions)
            assert nodes.loc[label, "y"] == np.median(positions % 7)
        assert len(result.tables["edges"]) > 0
    finally:
        result.close()


@pytest.mark.parametrize(
    ("groups", "layout"),
    [("kept again", "kept"), ("kept", "kept again"), ("every", "kept again")],
)
def test_cluster_connectivity_draws_inputs_on_the_graphs_cells_alike(
    cells, groups, layout
):
    one_selection = _connectivity(cells, groups="kept", layout="kept")
    try:
        result = _connectivity(cells, groups=groups, layout=layout)
        try:
            for table in ("nodes", "edges"):
                pd.testing.assert_frame_equal(
                    result.tables[table], one_selection.tables[table]
                )
            assert result.provenance.n_cells == len(KEPT)
        finally:
            result.close()
    finally:
        one_selection.close()


def test_cluster_connectivity_colors_groups_of_more_cells_as_an_embedding_does(
    cells,
):
    groups = _labels(cells, "cluster", "every")
    drawn = embedding(
        cells.store, layout=cells.layouts["every"], color_by=groups, show=False
    )
    try:
        (scale,) = [found for found in drawn.scales if hasattr(found, "palette")]
        every_color = dict(scale.palette)
    finally:
        drawn.close()
    result = cluster_connectivity(
        cells.store,
        graph=cells.graph,
        groups=groups,
        layout=cells.layouts["kept"],
        show=False,
    )

    try:
        # AB, which only the cells outside the graph carry, sorts between A
        # and B and keeps its place among the default colors.
        assert list(every_color) == ["A", "AB", "B", "C"]
        palette = result.scales[0].palette
        assert {label: palette[label] for label in "ABC"} == {
            label: every_color[label] for label in "ABC"
        }
        nodes = result.axes["cluster_connectivity"].collections[-1]
        np.testing.assert_allclose(
            nodes.get_facecolors(), [to_rgba(every_color[label]) for label in "ABC"]
        )
    finally:
        result.close()


def test_cluster_connectivity_refuses_groups_without_labels_for_some_graph_cells(
    cells,
):
    with pytest.raises(ValueError) as raised:
        _connectivity(cells, groups="half", layout="kept")

    assert raised.value.args == (
        "The cluster_labels artifact of groups holds no label for some cells of "
        "the graph; compute it on cells that include the graph's, such as the "
        "graph's cell selection",
    )


@pytest.mark.parametrize("layout", ["half", "every"])
def test_cluster_connectivity_refuses_a_layout_of_other_cells(cells, layout):
    with pytest.raises(ValueError) as raised:
        _connectivity(cells, groups="kept", layout=layout)

    assert raised.value.args == (
        "The layout holds other cells than the graph; draw an embedding of the "
        "graph's cells, such as a UMAP of the graph",
    )


# Modality weights: the layout holds the WNN graph's cells.


@pytest.mark.parametrize("layout", ["kept", "kept again"])
def test_modality_weights_draws_a_layout_of_the_graphs_cells(cells, layout):
    stored = np.asarray(
        artifact_group(cells.store.zw, cells.wnn)["modality_weights"][:],
        dtype=np.float64,
    )

    result = modality_weights(
        cells.store, graph=cells.wnn, layout=cells.layouts[layout], show=False
    )

    try:
        weights = result.tables["weights"]
        assert weights.columns.tolist() == ["RNA", "ADT"]
        np.testing.assert_array_equal(weights.index, KEPT)
        np.testing.assert_allclose(weights.to_numpy(), stored)
        for column, assay in enumerate(("RNA", "ADT")):
            points = result.axes[assay].collections[0]
            np.testing.assert_array_equal(
                points.get_offsets(), np.column_stack([KEPT, KEPT % 7])
            )
            np.testing.assert_allclose(points.get_array(), stored[:, column])
    finally:
        result.close()


@pytest.mark.parametrize("layout", ["half", "every"])
def test_modality_weights_refuses_a_layout_of_other_cells(cells, layout):
    with pytest.raises(ValueError) as raised:
        modality_weights(
            cells.store, graph=cells.wnn, layout=cells.layouts[layout], show=False
        )

    assert raised.value.args == (
        "The layout holds other cells than the WNN graph; draw an embedding of "
        "the graph's cells, such as a UMAP of the graph",
    )
