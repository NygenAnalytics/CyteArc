import warnings
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from cytearc.plotting import DensityOverlay, Highlight
from cytearc.plotting._style import scatter_edges
from cytearc.plotting.embedding import (
    _color_labels,
    _density_selection_mask,
    _draw_density_overlay,
    _draw_highlight,
    _embedding_panel_keys,
    _multi_layout_facets,
    _resolve_highlight_mask,
    _retain_strongest_hotspots,
    _smoothed_local_mean,
    _soft_clip,
    _weighted_quantiles,
)

embedding_module = import_module("cytearc.plotting.embedding")


def test_embedding_helper_labels_duplicate_panels_and_explicit_facets(monkeypatch):
    store = SimpleNamespace(cells=SimpleNamespace(columns=("group",)))
    monkeypatch.setattr(
        embedding_module,
        "resolve_feature_batch",
        lambda _store, items, **_kwargs: [
            SimpleNamespace(label="resolved feature") for _ in items
        ],
    )

    assert _color_labels(store, [None, "group", "gene"], from_assay=None) == [
        "cells",
        "group",
        "resolved feature",
    ]
    assert _embedding_panel_keys(["same", "same"], [None]) == [
        (0, "same"),
        (1, "same"),
    ]
    assert _multi_layout_facets(
        store,
        facet_by="group",
        facet_order=("b", "a"),
        groups=None,
        subset_by=None,
        cell_key="I",
    ) == ["b", "a"]


def test_highlight_and_density_masks_validate_selected_metadata(monkeypatch):
    store = object()
    with pytest.raises(IndexError, match="outside the selected cell range"):
        _resolve_highlight_mask(
            store,
            Highlight(indices=(3,)),
            cell_key="I",
            n_cells=3,
        )

    monkeypatch.setattr(
        embedding_module,
        "_selected_metadata_column",
        lambda *_args, **_kwargs: np.array([True]),
    )
    with pytest.raises(ValueError, match="highlight metadata length"):
        _resolve_highlight_mask(
            store,
            Highlight(by="selected"),
            cell_key="I",
            n_cells=2,
        )
    with pytest.raises(ValueError, match="density metadata length"):
        _density_selection_mask(
            store,
            DensityOverlay(group_by="group"),
            cell_key="I",
            n_cells=2,
        )

    monkeypatch.setattr(
        embedding_module,
        "_selected_metadata_column",
        lambda *_args, **_kwargs: np.array([1, 0]),
    )
    with pytest.raises(TypeError, match="requires a boolean metadata column"):
        _resolve_highlight_mask(
            store,
            Highlight(by="selected"),
            cell_key="I",
            n_cells=2,
        )
    np.testing.assert_array_equal(
        _resolve_highlight_mask(
            store,
            Highlight(by="group", groups=(1,)),
            cell_key="I",
            n_cells=2,
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        _density_selection_mask(
            store,
            DensityOverlay(group_by="group"),
            cell_key="I",
            n_cells=2,
        ),
        [True, True],
    )


def test_embedding_numeric_helpers_cover_degenerate_inputs():
    assert scatter_edges("black", 0) == ("none", 0.0)
    np.testing.assert_allclose(
        _weighted_quantiles(
            np.array([3.0, 1.0, 2.0]),
            np.zeros(3),
            np.array([0.5]),
        ),
        [2.0],
    )
    with pytest.raises(ValueError, match="contour values must match"):
        _smoothed_local_mean(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            grid_pixels=16,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            sigma=1.0,
            min_support=0.25,
        )

    values = np.array([1.0, np.nan, 3.0])
    assert _soft_clip(values, 0) is values
    np.testing.assert_array_equal(
        np.isnan(_soft_clip(np.array([np.nan, np.inf]), 0.1)),
        [True, False],
    )


def test_hotspot_filter_retains_only_the_strongest_component():
    surface = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    support = np.ones_like(surface)

    unchanged = _retain_strongest_hotspots(
        surface,
        support,
        level=1.0,
        max_hotspots=2,
    )
    np.testing.assert_array_equal(unchanged, surface)

    filtered = _retain_strongest_hotspots(
        surface,
        support,
        level=1.0,
        max_hotspots=1,
    )
    assert filtered[2, 2] == 3.0
    assert filtered[0, 0] < 1.0


def test_density_and_highlight_draw_helpers_return_early_without_points():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.contour import ContourSet

    density = {
        "overlay": DensityOverlay(),
        "values": None,
        "xlim": (0.0, 1.0),
        "ylim": (0.0, 1.0),
        "theme": "light",
    }
    highlight = {
        "highlight": Highlight(indices=(0,)),
        "edgecolor": "black",
        "rasterized": False,
    }
    # Both helpers return before they use this axes, which cannot draw.
    ax = SimpleNamespace()
    _draw_density_overlay(ax, np.array([0.0, 1.0]), np.array([0.0, 1.0]), **density)
    assert (
        _draw_highlight(ax, np.array([]), np.array([]), np.array([]), **highlight)
        is None
    )

    # Three points are enough to contour, and one point to highlight.
    figure, axis = plt.subplots()
    try:
        _draw_density_overlay(
            axis, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), **density
        )
        (contours,) = axis.collections
        assert isinstance(contours, ContourSet)
        drawn = _draw_highlight(
            axis, np.array([0.5]), np.array([0.25]), np.array([4.0]), **highlight
        )
        np.testing.assert_array_equal(drawn.get_offsets(), [[0.5, 0.25]])
        # The default highlight is red, 1.5 times the point size, black haloed.
        np.testing.assert_array_equal(drawn.get_sizes(), [6.0])
        np.testing.assert_array_equal(drawn.get_facecolors(), [to_rgba("#d62728")])
        np.testing.assert_array_equal(drawn.get_edgecolors(), [to_rgba("black")])
    finally:
        plt.close(figure)


class _Cells:
    def __init__(self, **columns):
        self._columns = {name: np.asarray(value) for name, value in columns.items()}
        self.columns = tuple(self._columns)
        self.N = len(next(iter(self._columns.values())))

    def fetch(self, column, key="I"):
        return self._columns[column][self.active_index(key)]

    def fetch_all(self, column):
        return self._columns[column]

    def active_index(self, key="I"):
        return np.flatnonzero(self._columns[key])


def _plot_store(n=40, seed=0, **extra):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        cells=_Cells(
            I=np.ones(n, dtype=bool),
            none=np.zeros(n, dtype=bool),
            umap1=rng.normal(size=n),
            umap2=rng.normal(size=n),
            other1=rng.normal(size=n),
            other2=rng.normal(size=n),
            score=rng.normal(size=n),
            **extra,
        ),
        _defaultAssay="RNA",
        zw=None,
        _stored_display_metadata=lambda _column: None,
    )


def _clustered_points():
    rng = np.random.default_rng(4)
    x = np.concatenate((rng.normal(-2, 0.3, 60), rng.normal(2, 0.3, 60)))
    y = rng.normal(0, 0.3, 120)
    return x, y


def test_density_overlay_draws_filled_bands_and_haloed_lines():
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.contour import ContourSet

    x, y = _clustered_points()
    figure, (filled_ax, line_ax) = plt.subplots(1, 2)
    limits = {"xlim": (-4.0, 4.0), "ylim": (-4.0, 4.0)}

    # Levels outside (0, 1) are absolute surface values, not quantiles.
    _draw_density_overlay(
        filled_ax,
        x,
        y,
        overlay=DensityOverlay(kind="filled", levels=(1.0,), sigma=1.0, pixels=32),
        values=None,
        theme="notebook",
        **limits,
    )
    (filled,) = filled_ax.collections
    assert isinstance(filled, ContourSet)
    assert filled.filled is True
    assert filled.levels[0] == 1.0
    assert len(filled.levels) == 2

    _draw_density_overlay(
        line_ax,
        x,
        y,
        overlay=DensityOverlay(
            levels=(1.0,), sigma=1.0, pixels=32, halo_width=1.5, linewidth=0.8
        ),
        values=None,
        theme="dark",
        **limits,
    )
    (lines,) = line_ax.collections
    stroke, normal = lines.get_path_effects()
    assert isinstance(stroke, patheffects.withStroke)
    assert isinstance(normal, patheffects.Normal)
    # The halo widens the line by the halo width on both sides.
    assert stroke._gc == {"linewidth": 0.8 + 2 * 1.5, "foreground": "#202020"}
    plt.close(figure)


def test_density_overlay_skips_levels_outside_the_surface_and_needs_values():
    import matplotlib.pyplot as plt

    x, y = _clustered_points()
    figure, ax = plt.subplots()
    limits = {"xlim": (-4.0, 4.0), "ylim": (-4.0, 4.0), "theme": "notebook"}

    _draw_density_overlay(
        ax,
        x,
        y,
        overlay=DensityOverlay(levels=(1e9,)),
        values=None,
        **limits,
    )
    assert len(ax.collections) == 0

    with pytest.raises(ValueError, match="requires a continuous color panel"):
        _draw_density_overlay(
            ax,
            x,
            y,
            overlay=DensityOverlay(statistic="mean"),
            values=None,
            **limits,
        )
    plt.close(figure)


def test_categorical_legend_lists_missing_cells_last():
    from matplotlib.colors import to_hex

    groups = np.array(["b", "a", None, "b"] * 5, dtype=object)
    result = splt_embedding(
        _plot_store(n=20, group=groups),
        layout_key="umap",
        color_by="group",
        legend_loc="right",
        show=False,
    )

    legend = result.figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == ["a", "b", "NA"]
    assert to_hex(legend.legend_handles[-1].get_markerfacecolor()) == "#bdbdbd"
    result.close()


def test_on_data_labels_skip_categories_absent_from_a_facet():
    groups = np.repeat(["a", "b", "c"], 10).astype(object)
    facet = np.where(np.arange(30) < 10, "left", "right")
    result = splt_embedding(
        _plot_store(n=30, group=groups, facet=facet),
        layout_key="umap",
        color_by="group",
        facet_by="facet",
        legend_loc="on_data",
        show=False,
    )

    # The left facet holds only "a"; the right facet holds "b" and "c".
    labels = [
        sorted(text.get_text() for text in axis.texts) for axis in result.axes.values()
    ]
    assert labels == [["a"], ["b", "c"]]
    result.close()


def test_seeded_continuous_embedding_draws_cells_in_a_reproducible_shuffle():
    store = _plot_store(n=40)

    def drawn_order(seed):
        result = splt_embedding(
            store, layout_key="umap", color_by="score", seed=seed, show=False
        )
        offsets = np.asarray(result.axes["score"].collections[0].get_offsets())
        result.close()
        return offsets

    first = drawn_order(5)
    coordinates = np.column_stack(
        (store.cells.fetch_all("umap1"), store.cells.fetch_all("umap2"))
    )
    # Every cell is drawn once, at its rank in the seeded permutation.
    rank = np.random.default_rng(5).permutation(40)
    np.testing.assert_array_equal(first, coordinates[np.argsort(rank, kind="stable")])
    np.testing.assert_array_equal(first, drawn_order(5))
    assert not np.array_equal(first, drawn_order(6))


def test_stored_continuous_display_sets_the_color_scale_unless_clipped():
    from matplotlib.colors import Normalize

    store = _plot_store(n=40)
    store._stored_display_metadata = lambda column: (
        {
            "kind": "continuous",
            "colormap": "magma",
            "minimum": -5.0,
            "maximum": 5.0,
            "scale": "linear",
        }
        if column == "score"
        else None
    )
    score = store.cells.fetch_all("score")

    stored = splt_embedding(store, layout_key="umap", color_by="score", show=False)
    points = stored.axes["score"].collections[0]
    assert stored.provenance.extras["color_limits"]["score"] == (-5.0, 5.0)
    # Metadata colors draw in the seeded shuffle: each cell at its rank.
    drawn = np.argsort(np.random.default_rng(0).permutation(40), kind="stable")
    np.testing.assert_allclose(
        points.get_facecolors(),
        _colormap("magma")(Normalize(-5.0, 5.0)(score[drawn])),
        atol=1e-6,
    )
    stored.close()

    clipped = splt_embedding(
        store, layout_key="umap", color_by="score", clip_fraction=0.1, show=False
    )
    low, high = clipped.provenance.extras["color_limits"]["score"]
    # Clipping replaces the stored limits with the clipped value range.
    assert -5.0 < low < high < 5.0
    clipped.close()


def test_clipping_leaves_categorical_colors_unchanged():
    from matplotlib.colors import to_hex

    groups = np.repeat(["a", "b"], 20).astype(object)
    store = _plot_store(n=40, group=groups)
    plain = splt_embedding(store, layout_key="umap", color_by="group", show=False)
    clipped = splt_embedding(
        store, layout_key="umap", color_by="group", clip_fraction=0.2, show=False
    )

    assert [
        to_hex(color) for color in clipped.axes["group"].collections[0].get_facecolors()
    ] == [
        to_hex(color) for color in plain.axes["group"].collections[0].get_facecolors()
    ]
    plain.close()
    clipped.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Provide exactly one of layout_key or layout"),
        (
            {"layout_key": "umap", "layout": "artifact"},
            "Provide exactly one of layout_key or layout",
        ),
        (
            {"layout": "artifact", "cell_key": "filtered"},
            "cell_key cannot override an artifact's stored cell selection",
        ),
        (
            {"layout_key": "umap", "cell_key": "none"},
            "No cells selected by cell_key 'none'",
        ),
        (
            {"layout_key": "umap", "color_by": "score", "groups": ["a"]},
            "groups requires a categorical color_by column, or facet_by",
        ),
        (
            {"layout_key": ["umap", "other"], "color_by": "artifact"},
            "ArtifactRef color_by requires an explicit layout ArtifactRef",
        ),
    ],
)
def test_embedding_rejects_ambiguous_layouts_and_selections(kwargs, message):
    from cytearc.storage import ArtifactRef

    artifact = ArtifactRef(
        scope="assay", assay="RNA", kind="cluster_labels", artifact_id="1" * 64
    )
    layout = ArtifactRef(
        scope="assay", assay="RNA", kind="embedding", artifact_id="2" * 64
    )
    options = {
        key: {"artifact": layout if key == "layout" else artifact}.get(value, value)
        if isinstance(value, str)
        else value
        for key, value in kwargs.items()
    }

    with pytest.raises(ValueError) as raised:
        splt_embedding(_plot_store(), show=False, **options)

    assert raised.value.args == (message,)


SCREEN_CELLS = 120
# The smaller layout leaves out every tenth cell of the screen.
LEFT_OUT = np.arange(SCREEN_CELLS) % 10 == 0
LAYOUT_CELLS = np.flatnonzero(~LEFT_OUT)


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    """A screen of 120 cells that each carry one guide, and two layouts.

    The layouts are imported embeddings: "every" of every cell, and "kept" of
    all but every tenth cell. The cells left out carry the one guide of target
    A1, and the others one guide of A, B or the NT controls. Each cell stands
    at x equal to its store position, so a drawn point names its cell.
    """
    from cytearc.embeddings.imported import write_imported_embedding
    from cytearc.storage.artifacts import fingerprint_array
    from tests.perturbation_helpers import write_screen

    rng = np.random.default_rng(0)
    guides = np.zeros((SCREEN_CELLS, 7), dtype=np.int64)
    guide = np.where(LEFT_OUT, 6, np.arange(SCREEN_CELLS) % 6)
    guides[np.arange(SCREEN_CELLS), guide] = 30
    store = write_screen(
        tmp_path_factory.mktemp("screen") / "screen.zarr",
        rng.poisson(2.0, size=(SCREEN_CELLS, 20)),
        guides,
        targets=["A", "A", "B", "B", "NT", "NT", "A1"],
        cell_columns={"kept": ~LEFT_OUT, "half": np.arange(SCREEN_CELLS) < 60},
    )
    ids = np.asarray(store.cells.fetch_all("ids"))
    layouts = {}
    for name, cells in (("every", "I"), ("kept", "kept")):
        positions = np.flatnonzero(np.asarray(store.cells.fetch_all(cells)))
        coordinates = np.column_stack([positions, positions % 7]).astype(np.float32)
        layouts[name] = write_imported_embedding(
            store.zw,
            assay="RNA",
            dimreduc_key=f"X_umap_{name}",
            role="umap",
            coordinates=coordinates,
            source_digest=bytes([len(layouts)]) * 32,
            payload_fingerprints={"values": fingerprint_array(coordinates)},
            source_cell_ids=ids[positions],
            cell_selection=store.snapshot_cell_selection(cells),
        )
    return store, layouts


def _screen_labels(store, selection):
    """Label the cells of ``selection`` by the guide called in each."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calls = store.perturbation.call_guides(selection, method="fixed", min_umi=10)
        return store.perturbation.labels(calls, controls=["NT"])


def _palette(store, layout, labels):
    """The colour of each label drawn on ``layout``."""
    result = splt_embedding(store, layout=layout, color_by=labels, show=False)
    try:
        (scale,) = [found for found in result.scales if hasattr(found, "palette")]
        return dict(scale.palette)
    finally:
        result.close()


@pytest.mark.parametrize(
    "source", ["every cell", "the layout's cells, selected again", "the layout's"]
)
def test_labels_of_cells_that_hold_the_layouts_colour_each_cell(screen, source):
    from matplotlib.colors import to_rgba

    from cytearc.plotting._data import _artifact_cell_selection

    store, layouts = screen
    layout_selection = _artifact_cell_selection(store, layouts["kept"])
    selection = {
        "every cell": lambda: store.select_measured_cells("CRISPR"),
        # Another selection artifact that holds exactly the layout's cells.
        "the layout's cells, selected again": lambda: store.select_cells(
            _screen_labels(store, store.select_measured_cells("CRISPR")),
            include=["A", "B", "control"],
        ),
        "the layout's": lambda: layout_selection,
    }[source]()
    assert (selection == layout_selection) == (source == "the layout's")
    labels = _screen_labels(store, selection)
    result = splt_embedding(store, layout=layouts["kept"], color_by=labels, show=False)

    try:
        (scale,) = [found for found in result.scales if hasattr(found, "palette")]
        label_of = store.perturbation.load_labels(labels).cells["label"]
        ids = np.asarray(store.cells.fetch_all("ids"))
        (points,) = next(iter(result.axes.values())).collections[:1]
        offsets = np.asarray(points.get_offsets())
        drawn = np.rint(offsets[:, 0]).astype(np.int64)
        # Each cell of the layout is drawn once, at its own place, in the
        # colour of its own label.
        np.testing.assert_array_equal(np.sort(drawn), LAYOUT_CELLS)
        np.testing.assert_array_equal(offsets[:, 1], drawn % 7)
        colors = [to_rgba(scale.palette[label_of[ids[cell]]]) for cell in drawn]
        np.testing.assert_allclose(points.get_facecolors(), colors)
    finally:
        result.close()


def test_labels_of_more_cells_keep_their_colours_on_a_layout_of_fewer(screen):
    store, layouts = screen
    labels = _screen_labels(store, store.select_measured_cells("CRISPR"))

    every = _palette(store, layouts["every"], labels)
    kept = _palette(store, layouts["kept"], labels)

    # A1, which only the cells left out of "kept" carry, sorts between A and
    # B, and still takes its place among the default colours.
    assert list(every) == ["A", "A1", "B", "control"]
    assert kept == {label: every[label] for label in ("A", "B", "control")}


def test_labels_of_cells_that_miss_some_of_the_layouts_are_refused(screen):
    store, layouts = screen
    labels = _screen_labels(store, store.snapshot_cell_selection("half"))

    with pytest.raises(ValueError) as raised:
        splt_embedding(store, layout=layouts["kept"], color_by=labels, show=False)

    assert raised.value.args == (
        "The perturbation_labels artifact of color_by holds no value for some "
        "cells of the layout; compute it on cells that include the layout's, "
        "such as the run's analysis_cell_selection",
    )


def test_multi_layout_facets_follow_groups_and_show_by_default(monkeypatch):
    from cytearc.plotting._figure import PlotResult

    shown = []
    monkeypatch.setattr(PlotResult, "show", lambda result: shown.append(result))
    facet = np.tile(["x", "y", "z"], 10).astype(object)

    result = splt_embedding(
        _plot_store(n=30, facet=facet),
        layout_key=["umap", "other"],
        color_by="score",
        facet_by="facet",
        groups=["z", "x"],
    )

    # Requested groups choose and order the facet panels of each layout.
    assert list(result.axes) == [
        ("umap", "score", "z"),
        ("umap", "score", "x"),
        ("other", "score", "z"),
        ("other", "score", "x"),
    ]
    assert all(
        len(axis.collections[0].get_offsets()) == 10 for axis in result.axes.values()
    )
    assert shown == [result]
    result.close()


def splt_embedding(store, **kwargs):
    import cytearc.plotting as splt

    return splt.embedding(store, **kwargs)


def _colormap(name):
    import matplotlib

    return matplotlib.colormaps[name]
