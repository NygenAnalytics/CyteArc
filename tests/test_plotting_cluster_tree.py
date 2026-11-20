"""Cluster tree drawing from a prepared hierarchy."""

import io
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import zarr
from matplotlib.colors import ListedColormap, to_hex, to_rgb
from zarr.storage import MemoryStore

import cytearc.plotting as splt
from cytearc.plotting.cluster_tree import _draw_tree_pie
from cytearc.storage import ArtifactRef
from cytearc.storage.artifacts import artifact_path

_GRAPH = ArtifactRef(
    scope="assay", assay="RNA", kind="connectivity_map", artifact_id="1" * 64
)
_CLUSTERS = ArtifactRef(
    scope="assay", assay="RNA", kind="cluster_cut", artifact_id="2" * 64
)


# The first colors of CyteArc's default palette, which plots assign over a
# column's labels in natural order.
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#279e68", "#d62728"]


def _tree_store(color_values, color_missing=None, *, column=None):
    """A root over clusters 1 and 2, each holding three of six cells.

    ``column`` holds every label of the fill column, the tree's cells and
    others; it is ``color_values`` by default.
    """
    graph = nx.DiGraph([(0, 1), (0, 2)])
    nx.set_node_attributes(graph, {0: 6, 1: 3, 2: 3}, "nleaves")
    nx.set_node_attributes(graph, {1: 1, 2: 2}, "partition_id")
    prepared = {
        "graph": graph,
        "clusters": np.array([1, 1, 1, 2, 2, 2]),
        "color_values": None if color_values is None else np.asarray(color_values),
        "color_missing": None if color_missing is None else np.asarray(color_missing),
        "coalesced_location": "tree",
        "from_assay": "RNA",
        "graph_ref": _GRAPH,
        "clusters_ref": _CLUSTERS,
        "cell_selection": ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="3" * 64
        ),
    }
    labels = np.asarray(color_values if column is None else column)
    return SimpleNamespace(
        _prepare_cluster_tree=lambda **_kwargs: prepared,
        cells=SimpleNamespace(fetch_all=lambda _column: labels),
    )


def _node_colors(result):
    """Face colors of the hierarchy nodes in graph node order (0, 1, 2)."""
    (nodes,) = [
        collection
        for collection in result.axes["tree"].collections
        if len(collection.get_offsets()) == 3
    ]
    return [to_hex(color) for color in nodes.get_facecolors()]


def _pies(ax):
    """The value pies drawn on a tree axes, in drawing order."""
    return [
        collection
        for collection in ax.collections
        if collection.get_gid() == "cluster-tree-pie"
    ]


def _wedge_covering(pie, degrees):
    """Index of the wedge covering each direction, in degrees from +x.

    Probes sit halfway to the rim in the pie's own marker coordinates, and
    every direction must fall in exactly one wedge: no gaps, no overlaps.
    """
    paths = pie.get_paths()
    reach = 0.5 * max(np.hypot(*path.vertices.T).max() for path in paths)
    radians = np.deg2rad(degrees)
    probes = reach * np.column_stack((np.cos(radians), np.sin(radians)))
    covering = np.array([path.contains_points(probes) for path in paths])
    np.testing.assert_array_equal(covering.sum(axis=0), np.ones(radians.size))
    return covering.argmax(axis=0)


def _render(figure, dpi):
    """RGB pixels of the figure saved by Agg at ``dpi``, top row first."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="rgba", dpi=dpi, facecolor="white")
    width, height = (round(side * dpi) for side in figure.get_size_inches())
    pixels = np.frombuffer(buffer.getvalue(), np.uint8).reshape(height, width, 4)
    return pixels[..., :3] / 255


def test_cluster_tree_draws_value_pies_and_skips_clusters_without_values():
    result = splt.cluster_tree(
        _tree_store(
            ["A", "B", "A", "C", "C", "C"],
            color_missing=[False, False, False, True, True, True],
        ),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="phase",
        color_key={"A": "#ff0000", "B": "#0000ff"},
        show_labels=False,
        show=False,
    )

    ax = result.axes["tree"]
    positions = result.tables["positions"].set_index("node")
    # Cluster 1 holds two A cells and one B cell; every value of cluster 2 is
    # missing, so it draws no pie.
    (pie,) = _pies(ax)
    assert [to_hex(color) for color in pie.get_facecolors()] == [
        "#ff0000",
        "#0000ff",
    ]
    np.testing.assert_allclose(
        pie.get_offset_transform().transform(pie.get_offsets()),
        ax.transData.transform(positions.loc[[1], ["x", "y"]].to_numpy(float)),
    )
    # A fills two thirds of the turn counter-clockwise from +x, then B.
    directions = np.arange(0.05, 360, 0.1)
    np.testing.assert_array_equal(
        _wedge_covering(pie, directions), np.where(directions < 240, 0, 1)
    )
    # Value pies replace the cluster fills, which stay white and hidden.
    assert _node_colors(result)[1:] == ["#ffffff", "#ffffff"]
    # The pies leave the limits to the tree, which still frames every node.
    result.figure.canvas.draw()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert positions["x"].between(x0, x1).all()
    assert positions["y"].between(y0, y1).all()
    result.close()


def test_tree_pie_is_sqrt_size_points_across_and_stays_circular():
    figure, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_axis_off()
    # Nine decades apart, these scales would flatten a pie drawn in data units.
    ax.set(xlim=(0, 1e6), ylim=(0, 1e-3))
    center, size = (5e5, 5e-4), 2500.0
    colors = ["#ff0000", "#00ff00", "#0000ff"]
    _draw_tree_pie(ax, np.array([1, 2, 3]), colors, x=center[0], y=center[1], size=size)

    def assert_marker(dpi):
        rgb = _render(figure, dpi)
        # Saturated colors on white: the darkest channel gives the coverage.
        ink = 1 - rgb.min(axis=-1)
        diameter = np.sqrt(size) * dpi / 72
        covered_rows, covered_columns = np.nonzero(ink > 0.5)
        assert np.ptp(covered_columns) + 1 == pytest.approx(diameter, abs=1.5)
        assert np.ptp(covered_rows) + 1 == pytest.approx(diameter, abs=1.5)
        assert 2 * np.sqrt(ink.sum() / np.pi) == pytest.approx(diameter, rel=0.01)
        # The ink centers on the data position, in pixels from the bottom left.
        x, y = ax.transData.transform(center) * dpi / figure.dpi
        weights = ink / ink.sum()
        rows, columns = np.indices(ink.shape)
        np.testing.assert_allclose(
            [
                (weights * (columns + 0.5)).sum(),
                (weights * (len(ink) - rows - 0.5)).sum(),
            ],
            [x, y],
            atol=0.5,
        )
        # The wedges run counter-clockwise from +x: sample each one's middle.
        for color, angle in zip(colors, np.deg2rad([30, 120, 270]), strict=True):
            column = int(x + 0.3 * diameter * np.cos(angle))
            row = int(len(ink) - y - 0.3 * diameter * np.sin(angle))
            np.testing.assert_allclose(rgb[row, column], to_rgb(color), atol=0.02)

    try:
        assert_marker(dpi=100)
        # Saved at another dpi, then zoomed off-center, then resized.
        assert_marker(dpi=250)
        ax.set(xlim=(4.5e5, 6.5e5), ylim=(4.5e-4, 5.3e-4))
        assert_marker(dpi=100)
        figure.set_size_inches(6, 2)
        assert_marker(dpi=100)
    finally:
        plt.close(figure)


def test_tree_pies_leave_the_axes_limits_to_the_nodes():
    figure, (ax, reference) = plt.subplots(1, 2)
    nodes = np.array([[0.0, 0.0], [3.0, 1.0], [1.0, -2.0]])
    for target in (ax, reference):
        target.scatter(nodes[:, 0], nodes[:, 1], s=100)
    data_limits = ax.dataLim.get_points().copy()
    try:
        with patch.object(ax, "autoscale_view", wraps=ax.autoscale_view) as autoscale:
            for x, y in nodes:
                _draw_tree_pie(
                    ax, np.array([1, 1]), ["#ff0000", "#0000ff"], x=x, y=y, size=100.0
                )
        # The node markers already hold every pie position, so the pies
        # neither autoscale on insertion nor move the limits.
        assert autoscale.call_count == 0
        np.testing.assert_array_equal(ax.dataLim.get_points(), data_limits)
        figure.canvas.draw()
        assert ax.get_xlim() == reference.get_xlim()
        assert ax.get_ylim() == reference.get_ylim()
    finally:
        plt.close(figure)


def test_cluster_tree_shows_owned_results_by_default(monkeypatch):
    shown = []
    monkeypatch.setattr(splt.PlotResult, "show", lambda result: shown.append(result))

    result = splt.cluster_tree(
        _tree_store(None), graph=_GRAPH, clusters=_CLUSTERS, show_labels=False
    )

    assert shown == [result]
    assert result.tables["cluster_summary"]["n_cells"].tolist() == [3, 3]
    result.close()
    assert not plt.fignum_exists(result.figure.number)


def test_cluster_tree_keeps_missing_clusters_gray_under_a_uniform_fill():
    def drawn(values, missing):
        result = splt.cluster_tree(
            _tree_store(values, color_missing=missing),
            graph=_GRAPH,
            clusters=_CLUSTERS,
            fill_by_value="score",
            force_ints_as_cats=False,
            show_labels=False,
            show=False,
        )
        try:
            return _node_colors(result), result.scales[0]
        finally:
            result.close()

    uniform, _ = drawn([5.0] * 6, [False] * 6)
    one_missing, _ = drawn(
        [5.0, 5.0, 5.0, 0.0, 0.0, 0.0], [False, False, False, True, True, True]
    )
    # Cluster 1's third cell is missing too, and its stored 1.0 is no value.
    partly, partly_scale = drawn(
        [5.0, 5.0, 1.0, 0.0, 0.0, 0.0], [False, False, True, True, True, True]
    )

    # Both clusters share the one observed value's color when nothing is
    # missing; a cluster whose values are all missing shows in the missing
    # gray, and the observed cluster keeps its color.
    assert uniform[1] == uniform[2] != "#bdbdbd"
    assert one_missing[1] == uniform[1]
    assert one_missing[2] == "#bdbdbd"
    # A cluster's mean leaves its missing cells out, so cluster 1 still
    # means 5, and the limits span that one mean.
    assert partly[1] == uniform[1]
    assert partly[2] == "#bdbdbd"
    assert (partly_scale.vmin, partly_scale.vmax) == (5.0, 6.0)


def test_cluster_tree_keeps_a_constant_fill_value():
    result = splt.cluster_tree(
        _tree_store([5.0] * 6),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="score",
        force_ints_as_cats=False,
        cmap="viridis",
        show_labels=False,
        show=False,
    )
    try:
        (scale, _sizes) = result.scales
        assert isinstance(scale, splt.ColorScale)
        assert (scale.vmin, scale.vmax) == (5.0, 6.0)
        (legend,) = result.legends
        assert legend.kind == "colorbar"
        assert (legend.extras["vmin"], legend.extras["vmax"]) == (5.0, 6.0)
        assert result.axes["colorbar"].get_ylim() == pytest.approx((5.0, 6.0))
        # The shared policy places a constant value at the low end of the map.
        low = to_hex(matplotlib.colormaps["viridis"](0.0))
        assert _node_colors(result)[1:] == [low, low]
    finally:
        result.close()


def test_cluster_tree_colors_clusters_as_embeddings_do():
    result = splt.cluster_tree(
        _tree_store(None),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        show_labels=False,
        show=False,
    )
    try:
        # Clusters 1 and 2 take the first two default colors, in their order.
        assert _node_colors(result)[1:] == DEFAULT_COLORS[:2]
        assert result.scales[0] == splt.CategoricalScale(
            order=(1, 2), palette={1: DEFAULT_COLORS[0], 2: DEFAULT_COLORS[1]}
        )
    finally:
        result.close()


def test_cluster_tree_colors_clusters_in_the_colors_their_clustering_stores():
    # A clustering that records its categories, as an H5AD import records
    # those of the clusterings it imports, orders and colors the clusters.
    root = zarr.open_group(store=MemoryStore(), mode="w")
    labels = root.create_group(artifact_path(_CLUSTERS)).create_array(
        "labels", data=np.array([1, 1, 1, 2, 2, 2])
    )
    labels.attrs["display"] = {
        "kind": "categorical",
        "categories": [
            {"value": 2, "label": "2", "color": "#00aa00"},
            {"value": 1, "label": "1", "color": "#ff0000"},
        ],
    }
    store = _tree_store(None)
    store.zw = root

    result = splt.cluster_tree(
        store, graph=_GRAPH, clusters=_CLUSTERS, show_labels=False, show=False
    )
    try:
        assert _node_colors(result)[1:] == ["#ff0000", "#00aa00"]
        assert result.scales[0] == splt.CategoricalScale(
            order=(2, 1), palette={2: "#00aa00", 1: "#ff0000"}
        )
    finally:
        result.close()


def test_cluster_tree_colors_categories_over_every_label_of_the_column():
    # Cell type A labels only cells outside the tree, and still takes the
    # first default color, so B and C keep the colors of a plot of all cells.
    result = splt.cluster_tree(
        _tree_store(
            ["B", "B", "C", "C", "C", "B"],
            column=["A", "C", "B", "B", "C", "C", "C", "B", "A"],
        ),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="cell_type",
        show_labels=False,
        show=False,
    )
    try:
        assert result.scales[0] == splt.CategoricalScale(
            order=("B", "C"), palette={"B": DEFAULT_COLORS[1], "C": DEFAULT_COLORS[2]}
        )
        pies = _pies(result.axes["tree"])
        assert {to_hex(color) for pie in pies for color in pie.get_facecolors()} == set(
            DEFAULT_COLORS[1:3]
        )
    finally:
        result.close()


@pytest.mark.parametrize(
    "cmap", ["viridis", ListedColormap(["#440154", "#fde725"], name="ends")]
)
def test_cluster_tree_samples_a_colormap_over_the_categories_in_order(cmap):
    result = splt.cluster_tree(
        _tree_store(["B", "B", "A", "A", "B", "B"]),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="cell_type",
        cmap=cmap,
        show_labels=False,
        show=False,
    )
    try:
        # A and B, in natural order, take the colormap at 0 and 1.
        assert result.scales[0].palette == {"A": "#440154", "B": "#fde725"}
    finally:
        result.close()


@pytest.mark.parametrize(
    ("cmap", "low", "high", "name"),
    [
        # viridis at 0 and 1.
        (None, "#440154", "#fde725", "viridis"),
        (
            ListedColormap(["#ff0000", "#0000ff"], name="red_blue"),
            "#ff0000",
            "#0000ff",
            "red_blue",
        ),
    ],
)
def test_cluster_tree_fills_numbers_from_a_colormap(cmap, low, high, name):
    # Cluster 1 means 1 and cluster 2 means 3, the ends of the color range.
    result = splt.cluster_tree(
        _tree_store([0.0, 1.0, 2.0, 3.0, 3.0, 3.0]),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="score",
        force_ints_as_cats=False,
        cmap=cmap,
        show_labels=False,
        show=False,
    )
    try:
        assert _node_colors(result)[1:] == [low, high]
        assert result.scales[0] == splt.ColorScale(cmap=name, vmin=1.0, vmax=3.0)
    finally:
        result.close()


def test_cluster_tree_keeps_a_single_category_categorical():
    result = splt.cluster_tree(
        _tree_store(np.asarray(["A"] * 6)),
        graph=_GRAPH,
        clusters=_CLUSTERS,
        fill_by_value="phase",
        show_labels=False,
        show=False,
    )
    try:
        (scale, _sizes) = result.scales
        assert isinstance(scale, splt.CategoricalScale)
        assert scale.order == ("A",)
        assert list(scale.palette) == ["A"]
        (legend,) = result.legends
        assert legend.kind == "categorical"
        assert "colorbar" not in result.axes
        pies = _pies(result.axes["tree"])
        # Each cluster draws one pie of a single full wedge in the category's
        # color.
        assert [[to_hex(color) for color in pie.get_facecolors()] for pie in pies] == [
            [scale.palette["A"]]
        ] * 2
        directions = np.arange(0.05, 360, 0.1)
        for pie in pies:
            np.testing.assert_array_equal(
                _wedge_covering(pie, directions), np.zeros(directions.size)
            )
    finally:
        result.close()
