"""Embedding labels, drawing order, layout, and color edge cases."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import LogNorm, to_hex, to_rgba

import cytearc.plotting as splt
from cytearc.plotting._style import (
    LEGEND_SIDE_ENTRIES_PER_COLUMN,
    default_point_size,
    panel_area_inches,
)
from tests.test_plotting_category_colors import _Cells
from tests.test_plotting_foundation import _ArrayAssay


def _store(columns, *, genes=None, displays=None, missing=None):
    """A store of these cell columns and, optionally, gene values."""
    n = len(next(iter(columns.values())))
    columns = {"I": np.ones(n, dtype=bool), **columns}
    values = (
        np.zeros((n, 1)) if genes is None else np.column_stack(list(genes.values()))
    )
    names = ["GeneA"] if genes is None else list(genes)
    store = SimpleNamespace(
        cells=_Cells(columns, missing),
        RNA=_ArrayAssay(values, names),
        _defaultAssay="RNA",
        nthreads=1,
        zw=None,
        _stored_display_metadata=lambda column: (displays or {}).get(column),
    )
    store._get_assay = lambda name: store.RNA
    return store


def _draw_order(n, cells=None, seed=0):
    """Positions of the drawn cells in drawing order: each cell at its rank."""
    rank = np.random.default_rng(seed).permutation(n)
    if cells is None:
        cells = np.ones(n, dtype=bool)
    return np.argsort(rank[cells], kind="stable")


def _offsets(result, key):
    return np.asarray(result.axes[key].collections[0].get_offsets())


# On-data labels


def _clustered_store(centres, n_per, *, spread=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(cx, spread, n_per) for cx, _ in centres])
    y = np.concatenate([rng.normal(cy, spread, n_per) for _, cy in centres])
    labels = np.repeat(
        np.asarray([f"c{index:02d}" for index in range(len(centres))], dtype=object),
        n_per,
    )
    return _store({"U1": x, "U2": y, "cl": labels}), x, y, labels


def _label_geometry(result, x, y, labels):
    """Each label's text, display position, own-centre offset, and height."""
    figure = result.figure
    ax = next(iter(result.axes.values()))
    renderer = figure.canvas.get_renderer()
    names = sorted(set(labels))
    centres = np.asarray(
        [(np.median(x[labels == name]), np.median(y[labels == name])) for name in names]
    )
    centre_pixels = ax.transData.transform(centres)
    rows = []
    for text in ax.texts:
        position = ax.transData.transform(text.get_position())
        own = names.index(text.get_text())
        distances = np.hypot(*(centre_pixels - position).T)
        rows.append(
            (
                text.get_text(),
                int(np.argmin(distances)) == own,
                float(distances[own]),
                float(text.get_window_extent(renderer).height),
            )
        )
    return rows


def test_on_data_labels_stay_on_their_clusters_in_a_grid():
    # 30 clusters on a 6 x 5 grid, where labels sorted by height used to
    # climb onto the clusters above them.
    centres = [(cx * 10.0, cy * 10.0) for cy in range(5) for cx in range(6)]
    store, x, y, labels = _clustered_store(centres, 200)
    result = splt.embedding(
        store, layout_key="U", color_by="cl", legend_loc="on_data", show=False
    )
    try:
        rows = _label_geometry(result, x, y, labels)
        assert len(rows) == 30
        for name, nearest_own, distance, height in rows:
            assert nearest_own, name
            # Well-separated clusters keep their labels at the medians.
            assert distance < 0.5, name
    finally:
        result.close()


def test_on_data_labels_stay_nearest_their_clusters_in_a_dense_layout():
    from scipy.cluster.vq import kmeans2

    # Three touching, elongated populations cut into 30 adjacent clusters, as
    # fine clustering does to a UMAP.
    rng = np.random.default_rng(5)
    parts = [
        rng.normal((0.0, 0.0), (6.0, 1.5), size=(2_000, 2)),
        rng.normal((4.0, 5.0), (1.5, 5.0), size=(1_500, 2)),
        rng.normal((12.0, -2.0), (3.0, 3.0), size=(1_500, 2)),
    ]
    points = np.concatenate(parts)
    _, assignment = kmeans2(points, 30, seed=1, minit="++")
    labels = np.asarray([f"k{value}" for value in assignment], dtype=object)
    store = _store({"U1": points[:, 0], "U2": points[:, 1], "km": labels})
    result = splt.embedding(store, layout_key="U", color_by="km", show=False)
    try:
        assert result.legends[0].extras["legend_loc"] == "on_data"
        rows = _label_geometry(result, points[:, 0], points[:, 1], labels)
        assert len(rows) == len(set(labels))
        ax = next(iter(result.axes.values()))
        box = ax.get_window_extent(result.figure.canvas.get_renderer())
        moved = 0
        for name, nearest_own, distance, height in rows:
            assert nearest_own, name
            # A label moves at most about one label height, with its halo.
            assert distance <= height + 2.0 * result.figure.dpi / 72.0, name
            moved += distance > 0.5
        assert moved > 0
        for text in ax.texts:
            extent = text.get_window_extent(result.figure.canvas.get_renderer())
            assert box.x0 <= extent.x0 and extent.x1 <= box.x1
            assert box.y0 <= extent.y0 and extent.y1 <= box.y1
    finally:
        result.close()


def test_separate_labels_moves_overlaps_apart_within_limits():
    from cytearc.plotting._labels import separate_labels

    bounds = (0.0, 0.0, 100.0, 100.0)
    sizes = np.full((2, 2), 10.0)
    # Two labels side by side overlap by 6 in x and 10 in y: each moves 3 in
    # x, away from the other's centre.
    moved = separate_labels(
        np.array([[50.0, 50.0], [54.0, 50.0]]),
        sizes,
        np.array([[50.0, 50.0], [54.0, 50.0]]),
        bounds=bounds,
        max_shift=np.full(2, 100.0),
    )
    np.testing.assert_allclose(moved, [[47.0, 50.0], [57.0, 50.0]])

    # A third cluster centred 6 below the lower label stops it 0.9 of the way
    # to the halfway point, 2.7 below its anchor, so it stays nearer its own
    # centre; the upper label then moves the rest of the way alone.
    held = separate_labels(
        np.array([[50.0, 50.0], [50.0, 52.0]]),
        sizes,
        np.array([[50.0, 50.0], [50.0, 52.0], [50.0, 44.0]]),
        bounds=bounds,
        max_shift=np.full(2, 100.0),
    )
    np.testing.assert_allclose(held, [[50.0, 47.3], [50.0, 57.3]], atol=1e-3)

    # Coincident anchors separate by index; the overlap of 10 in y splits
    # evenly, within the default shift of one label height.
    free = separate_labels(
        np.array([[50.0, 50.0], [50.0, 50.0]]),
        sizes,
        np.array([[50.0, 50.0], [50.0, 50.0], [90.0, 90.0]]),
        bounds=bounds,
    )
    np.testing.assert_allclose(free, [[50.0, 45.0], [50.0, 55.0]])

    # The shift bound holds a label within its limit of its anchor.
    bounded = separate_labels(
        np.array([[50.0, 50.0], [50.0, 50.0]]),
        np.array([[30.0, 10.0], [30.0, 10.0]]),
        np.array([[50.0, 50.0], [50.0, 50.0]]),
        bounds=bounds,
        max_shift=np.array([4.0, 4.0]),
    )
    np.testing.assert_allclose(bounded, [[50.0, 46.0], [50.0, 54.0]])

    # A label at the panel edge does not leave the panel.
    edge = separate_labels(
        np.array([[50.0, 95.0], [50.0, 90.0]]),
        sizes,
        np.array([[50.0, 95.0], [50.0, 90.0]]),
        bounds=bounds,
        max_shift=np.full(2, 100.0),
    )
    assert edge[0, 1] == pytest.approx(95.0)
    assert edge[1, 1] < 90.0
    # A label whose box already crosses the edge at its anchor is not held by
    # that edge.
    crossing = separate_labels(
        np.array([[50.0, 98.0], [50.0, 93.0]]),
        sizes,
        np.array([[50.0, 98.0], [50.0, 93.0]]),
        bounds=bounds,
        max_shift=np.full(2, 100.0),
    )
    np.testing.assert_allclose(crossing, [[50.0, 100.5], [50.0, 90.5]])

    single = separate_labels(
        np.array([[1.0, 2.0]]), np.ones((1, 2)), np.zeros((1, 2)), bounds=bounds
    )
    np.testing.assert_allclose(single, [[1.0, 2.0]])


def test_on_data_labels_separate_on_a_vector_canvas():
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    # Two clusters whose medians nearly coincide, on a figure that only an
    # SVG canvas can draw.
    centres = [(0.0, 0.0), (0.05, 0.0), (8.0, 8.0)]
    store, _, _, _ = _clustered_store(centres, 40, spread=0.05)
    figure = Figure(figsize=(3.0, 3.0), layout="constrained")
    FigureCanvasSVG(figure)
    axis = figure.add_subplot()
    result = splt.embedding(
        store,
        layout_key="U",
        color_by="cl",
        legend_loc="on_data",
        point_size=4.0,
        target=axis,
        show=False,
    )
    positions = {text.get_text(): text.get_position() for text in axis.texts}
    # The overlapping labels moved apart; the distant one stayed put.
    assert positions["c00"] != pytest.approx(positions["c01"], abs=0.05)
    assert positions["c02"] == pytest.approx((8.0, 8.0), abs=0.05)
    result.close()


def test_placing_labels_again_gives_the_same_positions():
    from cytearc.plotting._labels import place_layout_labels

    centres = [(0.0, 0.0), (1.0, 0.2), (0.4, 0.5), (8.0, 8.0)]
    store, _, _, _ = _clustered_store(centres, 50, spread=0.2)
    result = splt.embedding(
        store, layout_key="U", color_by="cl", legend_loc="on_data", show=False
    )
    try:
        ax = result.axes["cl"]
        first = [text.get_position() for text in ax.texts]
        place_layout_labels(result.figure)
        assert [text.get_position() for text in ax.texts] == first
        # Labels sit on the data and never reshape the layout.
        assert not any(text.get_in_layout() for text in ax.texts)
    finally:
        result.close()


# Rendering cost


def _count_renders(monkeypatch):
    """Record each scatter or marker drawn on an Agg canvas.

    Agg binds its drawing methods to each renderer, and a layout pass without
    rendering replaces them with no-ops, so only real drawing is recorded.
    Matplotlib probes ``draw_path_collection`` with no paths before drawing,
    and the probe draws nothing.
    """
    calls = []
    update = RendererAgg._update_methods

    def counting_update(renderer):
        update(renderer)
        for name in ("draw_path_collection", "draw_markers"):

            def counting(*args, _method=getattr(renderer, name), _name=name, **kwargs):
                if _name == "draw_markers" or len(args[2]):
                    calls.append(_name)
                return _method(*args, **kwargs)

            setattr(renderer, name, counting)

    monkeypatch.setattr(RendererAgg, "_update_methods", counting_update)
    return calls


def test_embedding_settles_layout_and_sizes_without_rendering_points(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(0)
    n = 2_000
    labels = np.asarray([f"c{index}" for index in range(20)], dtype=object)[
        rng.integers(0, 20, n)
    ]
    store = _store({"U1": rng.normal(size=n), "U2": rng.normal(size=n), "cl": labels})
    calls = _count_renders(monkeypatch)

    result = splt.embedding(store, layout_key="U", color_by="cl", show=False)
    # The layout and the point sizes settle without drawing a point.
    assert calls == []
    ax = result.axes["cl"]
    size = float(ax.collections[0].get_sizes()[0])
    position = ax.get_position().bounds
    result.save(tmp_path / "plot.png")
    # Saving draws the scatter once.
    assert calls.count("draw_path_collection") == 1
    # The settled layout is the rendered one, and sizes fit it.
    assert ax.get_position().bounds == pytest.approx(position)
    assert size == pytest.approx(
        default_point_size(n, panel_area=panel_area_inches(ax), size_max=28.0)
    )
    result.close()


# Drawing order


def test_categorical_cells_draw_in_a_seeded_shuffle():
    rng = np.random.default_rng(2)
    n = 60
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 20)
    store = _store(
        {"U1": np.arange(n, dtype=float), "U2": rng.normal(size=n), "cl": labels}
    )

    default = splt.embedding(store, layout_key="U", color_by="cl", show=False)
    again = splt.embedding(store, layout_key="U", color_by="cl", show=False)
    other = splt.embedding(store, layout_key="U", color_by="cl", seed=7, show=False)
    stored = splt.embedding(store, layout_key="U", color_by="cl", seed=None, show=False)
    try:
        drawn = _offsets(default, "cl")[:, 0].astype(int)
        np.testing.assert_array_equal(drawn, _draw_order(n))
        np.testing.assert_array_equal(_offsets(again, "cl"), _offsets(default, "cl"))
        np.testing.assert_array_equal(
            _offsets(other, "cl")[:, 0].astype(int), _draw_order(n, seed=7)
        )
        np.testing.assert_array_equal(
            _offsets(stored, "cl")[:, 0].astype(int), np.arange(n)
        )
        # Colors stay with their cells.
        palette = default.scales[-1].palette
        assert [
            to_hex(color)
            for color in default.axes["cl"].collections[0].get_facecolors()
        ] == [palette[labels[cell]] for cell in drawn]
        assert default.provenance.extras["draw_order"] == {"cl": "shuffled"}
        assert default.provenance.extras["seed"] == 0
        assert stored.provenance.extras["draw_order"] == {"cl": "stored"}
    finally:
        for result in (default, again, other, stored):
            result.close()


def test_facet_panels_draw_their_cells_at_the_shared_ranks():
    n = 30
    store = _store(
        {
            "U1": np.arange(n, dtype=float),
            "U2": np.zeros(n),
            "cl": np.tile(np.asarray(["a", "b"], dtype=object), 15),
            "side": np.repeat(np.asarray(["left", "right"], dtype=object), 15),
        }
    )
    result = splt.embedding(
        store, layout_key="U", color_by="cl", facet_by="side", show=False
    )
    try:
        side = store.cells.fetch_all("side")
        for facet in ("left", "right"):
            cells = np.flatnonzero(side == facet)
            np.testing.assert_array_equal(
                _offsets(result, ("cl", facet))[:, 0].astype(int),
                cells[_draw_order(n, side == facet)],
            )
    finally:
        result.close()


def test_gene_colors_draw_high_values_on_top_by_default():
    n = 40
    rng = np.random.default_rng(4)
    gene = np.where(rng.random(n) < 0.6, 0.0, rng.uniform(1, 5, n))
    metadata = rng.normal(size=n)
    store = _store(
        {"U1": np.arange(n, dtype=float), "U2": np.zeros(n), "score": metadata},
        genes={"GeneA": gene},
    )
    result = splt.embedding(
        store, layout_key="U", color_by=["GeneA", "score"], show=False
    )
    unsorted = splt.embedding(
        store, layout_key="U", color_by="GeneA", sort_values=False, show=False
    )
    every = splt.embedding(
        store, layout_key="U", color_by="score", sort_values=True, show=False
    )
    stored = splt.embedding(
        store, layout_key="U", color_by="GeneA", seed=None, show=False
    )
    try:
        rank = np.random.default_rng(0).permutation(n)
        drawn = _offsets(result, "GeneA")[:, 0].astype(int)
        # Values rise, and the many zeros draw in shuffle order.
        np.testing.assert_array_equal(drawn, np.lexsort((rank, gene)))
        # Without a seed, ties keep the stored order.
        np.testing.assert_array_equal(
            _offsets(stored, "GeneA")[:, 0].astype(int),
            np.argsort(gene, kind="stable"),
        )
        assert stored.provenance.extras["draw_order"] == {"GeneA": "sorted"}
        np.testing.assert_array_equal(
            _offsets(result, "score")[:, 0].astype(int), _draw_order(n)
        )
        np.testing.assert_array_equal(
            _offsets(unsorted, "GeneA")[:, 0].astype(int), _draw_order(n)
        )
        np.testing.assert_array_equal(
            _offsets(every, "score")[:, 0].astype(int), np.lexsort((rank, metadata))
        )
        assert result.provenance.extras["draw_order"] == {
            "GeneA": "sorted",
            "score": "shuffled",
        }
        assert result.provenance.extras["sort_values"] is None
    finally:
        for plot in (result, unsorted, every, stored):
            plot.close()


# Facets


def test_cells_without_a_facet_value_get_a_labelled_panel():
    n = 9
    facet = np.asarray(["x", "y", None] * 3, dtype=object)
    store = _store(
        {
            "U1": np.arange(n, dtype=float),
            "U2": np.zeros(n),
            "score": np.arange(n, dtype=float),
            "facet": facet,
        }
    )
    result = splt.embedding(
        store,
        layout_key="U",
        color_by=splt.CellField("score", kind="continuous"),
        facet_by="facet",
        show=False,
    )
    try:
        assert list(result.axes) == [("score", "x"), ("score", "y"), ("score", None)]
        assert [ax.get_title() for ax in result.axes.values()] == [
            "score | facet=x",
            "score | facet=y",
            "score | facet=NA",
        ]
        assert sorted(_offsets(result, ("score", None))[:, 0]) == [2.0, 5.0, 8.0]
    finally:
        result.close()

    display = {
        "kind": "categorical",
        "categories": [
            {"value": "x", "label": "x", "color": "#111111"},
            {"value": "y", "label": "y", "color": "#222222"},
        ],
        "missing_label": "unassigned",
        "missing_color": "#cccccc",
    }
    labelled = _store(
        {
            "U1": np.arange(n, dtype=float),
            "U2": np.zeros(n),
            "facet": facet,
            "other1": np.arange(n, dtype=float),
            "other2": np.ones(n),
        },
        displays={"facet": display},
    )
    layouts = splt.embedding(
        labelled, layout_key=["U", "other"], facet_by="facet", show=False
    )
    try:
        assert ("U", "cells", None) in layouts.axes
        assert layouts.axes[("other", "cells", None)].get_title() == (
            "other | cells | facet=unassigned"
        )
    finally:
        layouts.close()


def _grid_slots(result):
    """Each panel's (row, column) in its figure grid."""
    slots = {}
    for key, ax in result.axes.items():
        spec = ax.get_subplotspec()
        slots[key] = (spec.rowspan.start, spec.colspan.start)
    return slots


def test_facets_wrap_at_four_columns_and_start_each_color_on_a_new_row():
    n = 36
    facet = np.repeat(np.asarray([f"f{index}" for index in range(6)], dtype=object), 6)
    store = _store(
        {
            "U1": np.arange(n, dtype=float),
            "U2": np.zeros(n),
            "facet": facet,
            "a": np.arange(n, dtype=float),
            "b": -np.arange(n, dtype=float),
        }
    )
    options = {"layout_key": "U", "facet_by": "facet", "show": False}
    wrapped = splt.embedding(
        store,
        color_by=[
            splt.CellField("a", kind="continuous"),
            splt.CellField("b", kind="continuous"),
        ],
        **options,
    )
    explicit = splt.embedding(
        store, color_by=splt.CellField("a", kind="continuous"), n_columns=6, **options
    )
    two_layouts = _store(
        {
            "U1": np.arange(n, dtype=float),
            "U2": np.zeros(n),
            "V1": np.zeros(n),
            "V2": np.arange(n, dtype=float),
            "facet": facet,
        }
    )
    layouts = splt.embedding(
        two_layouts, layout_key=["U", "V"], facet_by="facet", show=False
    )
    # An explicit column count fills the grid row by row.
    filled = splt.embedding(
        two_layouts, layout_key=["U", "V"], facet_by="facet", n_columns=5, show=False
    )
    try:
        filled_slots = _grid_slots(filled)
        assert [filled_slots[("V", "cells", f)] for f in ("f0", "f3", "f4")] == [
            (1, 1),
            (1, 4),
            (2, 0),
        ]
        slots = _grid_slots(wrapped)
        facets = [f"f{index}" for index in range(6)]
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
        assert [slots[("a", f)] for f in facets] == positions
        assert [slots[("b", f)] for f in facets] == [
            (row + 2, column) for row, column in positions
        ]
        # Unused grid cells are removed, and the figure is four panels wide.
        assert len(wrapped.figure.axes) == 12 + 2
        assert wrapped.figure.get_figwidth() == pytest.approx(4 * 3.2 + 0.25)
        assert {column for _, column in _grid_slots(explicit).values()} == set(range(6))
        layout_slots = _grid_slots(layouts)
        assert [layout_slots[("V", "cells", f)] for f in facets] == [
            (row + 2, column) for row, column in positions
        ]
    finally:
        for result in (wrapped, explicit, layouts, filled):
            result.close()


# Continuous colors


def test_clip_fraction_uses_only_the_drawn_cells():
    rng = np.random.default_rng(0)
    n = 10_000
    group = np.where(np.arange(n) < 9_000, "A", "B").astype(object)
    score = np.where(group == "A", rng.uniform(0, 1, n), rng.uniform(100, 200, n))
    store = _store(
        {
            "U1": rng.normal(size=n),
            "U2": rng.normal(size=n),
            "score": score,
            "grp": group,
        }
    )
    result = splt.embedding(
        store,
        layout_key="U",
        color_by="score",
        facet_by="grp",
        groups=["B"],
        clip_fraction=0.05,
        show=False,
    )
    try:
        expected = tuple(np.percentile(score[group == "B"], [5, 95]))
        assert result.provenance.extras["color_limits"]["score"] == pytest.approx(
            expected
        )
    finally:
        result.close()


def test_missing_color_replaces_the_active_scale_missing_color():
    n = 20
    score = np.arange(n, dtype=float)
    score[:5] = np.nan
    category = np.asarray(["a", "b"] * 10, dtype=object)
    category[:3] = None
    store = _store(
        {
            "U1": np.arange(n, dtype=float),
            "U2": np.zeros(n),
            "score": score,
            "cat": category,
        }
    )
    red = to_rgba("#ff0000")
    plots = {
        "score": splt.embedding(
            store,
            layout_key="U",
            color_by="score",
            color_scale=splt.ColorScale(cmap="magma"),
            missing_color="#ff0000",
            show=False,
        ),
        "cat": splt.embedding(
            store,
            layout_key="U",
            color_by="cat",
            categorical_scale=splt.CategoricalScale(
                palette={"a": "#0000ff", "b": "#00ff00"}
            ),
            missing_color="#ff0000",
            show=False,
        ),
    }
    try:
        for key, values in (("score", score), ("cat", category)):
            points = plots[key].axes[key].collections[0]
            cells = np.asarray(points.get_offsets())[:, 0].astype(int)
            missing = np.asarray([value is None or value != value for value in values])
            colors = points.get_facecolors()
            assert np.allclose(colors[missing[cells]], red)
            assert not np.any(np.all(np.isclose(colors[~missing[cells]], red), axis=1))
        assert plots["score"].scales[0].missing_color == "#ff0000"
        assert plots["cat"].scales[-1].missing_color == "#ff0000"
    finally:
        for result in plots.values():
            result.close()


def test_log_scale_draws_values_that_are_not_positive_as_missing():
    values = np.array([0.0, 1.0, 10.0, -2.0, 100.0])
    store = _store(
        {
            "U1": np.arange(5, dtype=float),
            "U2": np.zeros(5),
            "counts": values,
            "zeros": np.zeros(5),
        }
    )
    result = splt.embedding(
        store,
        layout_key="U",
        color_by=["counts", "zeros"],
        color_scale=splt.ColorScale(scale="log", missing_color="#ff0000"),
        sort_values=False,
        show=False,
    )
    try:
        assert result.provenance.extras["color_limits"] == {
            "counts": (1.0, 100.0),
            "zeros": None,
        }
        points = result.axes["counts"].collections[0]
        cells = np.asarray(points.get_offsets())[:, 0].astype(int)
        expected = plt.get_cmap("viridis")(
            LogNorm(1.0, 100.0)(np.clip(values, 1, None))
        )
        expected[values <= 0] = to_rgba("#ff0000")
        np.testing.assert_allclose(points.get_facecolors(), expected[cells])
        # A color with no positive value draws no colorbar.
        assert [legend.label for legend in result.legends] == ["counts"]
    finally:
        result.close()


def test_log_limits_widen_a_tie_by_a_decade():
    from cytearc.plotting._style import resolve_color_limits

    log = {"scale": "log"}
    # A single positive value spans a decade upward, and a lone explicit
    # upper limit below the data a decade downward.
    assert resolve_color_limits([0.0, 4.0], splt.ColorScale(**log)) == (4.0, 40.0)
    assert resolve_color_limits([5.0, 6.0], splt.ColorScale(vmax=2.0, **log)) == (
        pytest.approx(0.2),
        2.0,
    )
    # Without a positive value, a log scale spans one to ten.
    assert resolve_color_limits([0.0, -1.0], splt.ColorScale(**log)) == (1.0, 10.0)


# Side legends


@pytest.mark.parametrize("theme", ["notebook", "paper"])
def test_side_legend_heights_and_rows_agree(theme):
    from cytearc.plotting._style import (
        legend_side_height,
        legend_side_rows,
        theme_context,
    )

    with theme_context(theme):
        for rows in range(1, LEGEND_SIDE_ENTRIES_PER_COLUMN + 1):
            for title in (False, True):
                height = legend_side_height(rows, title=title)
                assert legend_side_rows(height, title=title) == rows
                assert legend_side_rows(height - 0.01, title=title) == max(rows - 1, 1)
        # Rows never exceed one column of entries or fall below one.
        assert legend_side_rows(100.0, title=True) == LEGEND_SIDE_ENTRIES_PER_COLUMN
        assert legend_side_rows(0.1, title=True) == 1


def _legend_inside_figure(result):
    figure = result.figure
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    box = figure.bbox
    for legend in [*figure.legends, *(ax.get_legend() for ax in figure.axes)]:
        if legend is None:
            continue
        extent = legend.get_window_extent(renderer)
        assert box.y0 - 1 <= extent.y0 and extent.y1 <= box.y1 + 1


def test_side_legends_fit_their_figure():
    rng = np.random.default_rng(0)
    n = 3_000
    labels = np.asarray(
        [f"k{index}" for index in rng.integers(0, 150, n)], dtype=object
    )
    store = _store({"U1": rng.normal(size=n), "U2": rng.normal(size=n), "many": labels})
    auto = splt.embedding(store, layout_key="U", color_by="many", show=False)
    small = splt.embedding(
        store, layout_key="U", color_by="many", figsize=(6.0, 3.0), show=False
    )
    figure, axis = plt.subplots(figsize=(6.0, 2.5), layout="constrained")
    target = splt.embedding(
        store, layout_key="U", color_by="many", target=axis, show=False
    )
    try:
        legend = auto.figure.legends[0]
        assert legend.get_title().get_text() == "many (80 of 150)"
        # The figure grows to hold four columns of twenty entries.
        assert legend._ncols == 4
        assert auto.figure.get_figheight() > 3.4
        _legend_inside_figure(auto)

        legend = small.figure.legends[0]
        rows = int(np.ceil(len(legend.legend_handles) / legend._ncols))
        assert rows < LEGEND_SIDE_ENTRIES_PER_COLUMN
        assert legend.get_title().get_text() == f"many ({4 * rows} of 150)"
        _legend_inside_figure(small)
        _legend_inside_figure(target)
    finally:
        auto.close()
        small.close()
        plt.close(figure)
