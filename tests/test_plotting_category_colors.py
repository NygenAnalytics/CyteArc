"""Category colors and labels stay the same in every plot of a column."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgba
from seaborn.utils import desaturate

import cytearc.plotting as splt
from cytearc.plotting._style import resolve_category_scale
from tests.test_plotting_foundation import _ArrayAssay, _imported_plot_store

CATEGORIES = (
    "Alpha",
    "Beta",
    "Delta",
    "Ductal",
    "Epsilon",
    "Ngn3 high EP",
    "Ngn3 low EP",
    "Pre-endocrine",
)
# Scanpy's default palette, which CyteArc assigns over up to ten categories in
# natural order: the colors every plot of the column must reuse.
FULL_COLORS = dict(
    zip(
        CATEGORIES,
        (
            "#1f77b4",
            "#ff7f0e",
            "#279e68",
            "#d62728",
            "#aa40fc",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ),
        strict=True,
    )
)
CELLS_PER_CATEGORY = 6


class _Cells:
    """Cell metadata with cell keys, row reads and linked missing masks."""

    def __init__(self, columns, missing=None):
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        self._missing = dict(missing or {})
        self.columns = tuple(self._columns)
        self.N = len(next(iter(self._columns.values())))

    def active_index(self, key="I"):
        return np.flatnonzero(self._columns[key])

    def fetch(self, column, key="I"):
        return self._columns[column][self.active_index(key)]

    def fetch_all(self, column):
        return self._columns[column]

    def _get_array(self, column):
        return self._columns[column]

    def _get_missing_mask_array(self, column):
        return self._missing.get(column)

    def get_dtype(self, column):
        return self._columns[column].dtype


def _store(displays=None, **extra):
    """Six cells of each category, shuffled, on distinct coordinates."""
    rng = np.random.default_rng(3)
    labels = np.repeat(np.asarray(CATEGORIES, dtype=object), CELLS_PER_CATEGORY)
    labels = labels[rng.permutation(len(labels))]
    n = len(labels)
    columns = {
        "I": np.ones(n, dtype=bool),
        "no_beta": labels != "Beta",
        "not_alpha": labels != "Alpha",
        "umap1": np.arange(n, dtype=np.float64),
        "umap2": rng.normal(size=n),
        "cluster": labels,
        "batch": np.where(np.arange(n) % 2 == 0, "b1", "b2").astype(object),
        "metric": rng.gamma(2.0, 1.0, size=n),
        **extra,
    }
    return _bind_store(columns, rng, displays)


def _bind_store(columns, rng, displays):
    """A store of these cell columns and a two-gene RNA assay."""
    store = SimpleNamespace(
        cells=_Cells(columns),
        RNA=_ArrayAssay(rng.random((len(columns["I"]), 2)), ["GeneA", "GeneB"]),
        _defaultAssay="RNA",
        nthreads=1,
        zw=None,
        _stored_display_metadata=lambda column: (displays or {}).get(column),
    )
    store._get_assay = lambda name: store.RNA
    return store


def _category_scale(result):
    return next(
        scale for scale in result.scales if isinstance(scale, splt.CategoricalScale)
    )


def _drawn_colors(result, store):
    """Each drawn cell's color, keyed by its category, from the point offsets."""
    (points,) = next(iter(result.axes.values())).collections
    cells = np.asarray(points.get_offsets())[:, 0].astype(int)
    labels = store.cells.fetch_all("cluster")[cells]
    colors: dict[str, set[str]] = {}
    for label, color in zip(labels, points.get_facecolors(), strict=True):
        colors.setdefault(label, set()).add(to_hex(color))
    return colors


def test_embedding_keeps_category_colors_across_groups_subsets_and_cell_keys():
    store = _store()
    options = {"layout_key": "umap", "color_by": "cluster", "show": False}
    plots = {
        "full": splt.embedding(store, **options),
        "groups": splt.embedding(store, groups=["Beta", "Alpha", "Delta"], **options),
        "subset": splt.embedding(store, subset_by="not_alpha", **options),
        "cell_key": splt.embedding(store, cell_key="no_beta", **options),
    }
    try:
        shown = {
            "full": CATEGORIES,
            "groups": ("Beta", "Alpha", "Delta"),
            "subset": tuple(value for value in CATEGORIES if value != "Alpha"),
            "cell_key": tuple(value for value in CATEGORIES if value != "Beta"),
        }
        for name, result in plots.items():
            scale = _category_scale(result)
            # Requested groups order the legend; colors never move.
            assert scale.order == shown[name]
            assert scale.palette == {value: FULL_COLORS[value] for value in shown[name]}
            assert _drawn_colors(result, store) == {
                value: {FULL_COLORS[value]} for value in shown[name]
            }
        legend = plots["groups"].figure.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "Beta",
            "Alpha",
            "Delta",
        ]
        assert [
            to_hex(handle.get_markerfacecolor()) for handle in legend.legend_handles
        ] == [
            FULL_COLORS["Beta"],
            FULL_COLORS["Alpha"],
            FULL_COLORS["Delta"],
        ]
    finally:
        for result in plots.values():
            result.close()


def test_categories_of_hidden_rows_and_masked_rows_set_the_color_universe():
    # The category "Alpha" exists only in a row outside the cell key, and the
    # masked row's placeholder "Aa" is not a category.
    labels = np.asarray(["Gamma", "Beta", "Alpha", "Beta", "Aa", "Gamma"], dtype=object)
    store = _bind_store(
        {
            "I": np.array([True, True, False, True, True, True]),
            "umap1": np.arange(6, dtype=np.float64),
            "umap2": np.zeros(6),
            "cluster": labels,
        },
        np.random.default_rng(0),
        None,
    )
    store.cells._missing["cluster"] = np.array(
        [False, False, False, False, True, False]
    )

    result = splt.embedding(store, layout_key="umap", color_by="cluster", show=False)
    try:
        scale = _category_scale(result)
        assert scale.order == ("Beta", "Gamma")
        # Three categories in natural order: Alpha, Beta, Gamma.
        assert scale.palette == {"Beta": "#ff7f0e", "Gamma": "#279e68"}
    finally:
        result.close()


def test_a_frozen_run_view_ranges_colors_over_its_own_cells():
    from cytearc.plotting._data import _category_universe

    class FrozenCells:
        # A frozen run view holds only the run's cells; its full axis would
        # hold placeholders for the others.
        _selection_ref = object()

        def fetch(self, column, key="I"):
            assert (column, key) == ("cluster", "I")
            return np.asarray(["b", None, "a"], dtype=object)

        def fetch_all(self, column):
            raise AssertionError("a frozen view's full axis holds placeholders")

    universe = _category_universe(SimpleNamespace(cells=FrozenCells()), "cluster")
    assert universe.tolist() == ["b", None, "a"]


def test_distribution_keeps_group_colors_in_every_kind():
    store = _store()
    grouping = splt.CellField("cluster")
    groups = ["Beta", "Alpha"]
    try:
        violin = splt.distribution(
            store, "metric", grouping=grouping, groups=groups, max_points=0, show=False
        )
        assert violin.scales[0].order == ("Beta", "Alpha")
        assert violin.scales[0].palette == {
            group: FULL_COLORS[group] for group in groups
        }
        bodies = violin.axes["metric"].collections[:2]
        for body, group in zip(bodies, groups, strict=True):
            np.testing.assert_allclose(
                body.get_facecolor()[0],
                to_rgba(desaturate(FULL_COLORS[group], 0.9), 0.9),
            )

        subset = splt.distribution(
            store, "metric", grouping=grouping, subset_by="not_alpha", show=False
        )
        assert subset.scales[0].palette == {
            group: FULL_COLORS[group] for group in CATEGORIES if group != "Alpha"
        }

        hist = splt.distribution(
            store, "metric", grouping=grouping, groups=groups, kind="hist", show=False
        )
        handles = hist.axes["metric"].get_legend().legend_handles
        assert [to_hex(handle.get_facecolor()) for handle in handles] == [
            FULL_COLORS[group] for group in groups
        ]

        ecdf = splt.distribution(
            store, "metric", grouping=grouping, groups=groups, kind="ecdf", show=False
        )
        assert [to_hex(line.get_color()) for line in ecdf.axes["metric"].lines] == [
            FULL_COLORS[group] for group in groups
        ]
    finally:
        plt.close("all")


def test_distribution_split_colors_range_over_the_split_column():
    rng = np.random.default_rng(1)
    labels = np.repeat(np.asarray(["a", "b"], dtype=object), 12)
    side = np.tile(np.asarray(["left", "middle", "right"], dtype=object), 8)
    store = _bind_store(
        {
            "I": np.ones(24, dtype=bool),
            "group": labels,
            "side": side,
            "no_middle": side != "middle",
            "metric": rng.normal(size=24),
        },
        rng,
        None,
    )

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        split_by="side",
        subset_by="no_middle",
        max_points=0,
        show=False,
    )
    try:
        # Three sides in natural order: left, middle, right.
        assert result.scales[0].order == ("left", "right")
        assert result.scales[0].palette == {"left": "#1f77b4", "right": "#279e68"}
    finally:
        result.close()


def test_composition_keeps_category_colors_for_a_cell_key_subset():
    store = _store()
    full = splt.composition(store, category_by="cluster", show=False)
    subset = splt.composition(
        store, category_by="cluster", cell_key="no_beta", show=False
    )
    try:
        assert _category_scale(full).palette == FULL_COLORS
        shown = [value for value in CATEGORIES if value != "Beta"]
        assert _category_scale(subset).palette == {
            value: FULL_COLORS[value] for value in shown
        }
        bars = subset.axes["composition"].patches
        assert [to_hex(bar.get_facecolor()) for bar in bars] == [
            FULL_COLORS[value] for value in shown
        ]
    finally:
        full.close()
        subset.close()


def test_label_artifact_colors_range_over_all_labelled_cells(tmp_path):
    from matplotlib.colors import to_hex as hex_color

    x = np.arange(12, dtype=np.float64)
    labels = np.repeat([10, 2, 1], 4)
    store, imported, _counts = _imported_plot_store(
        tmp_path,
        coordinates=np.column_stack((x, (x % 4) * 1.5)),
        clusters=labels,
    )
    store.cells.insert("no_one", labels != 1)
    layout = imported.embeddingArtifacts["X_umap"]
    clusters = imported.clusterArtifacts["clusters"]
    # Labels 1, 2 and 10 take the first three default colors.
    colors = {1: "#1f77b4", 2: "#ff7f0e", 10: "#279e68"}
    try:
        subset = splt.embedding(
            store, layout=layout, color_by=clusters, subset_by="no_one", show=False
        )
        assert _category_scale(subset).palette == {2: colors[2], 10: colors[10]}
        (points,) = subset.axes["cluster_labels"].collections
        cells = np.asarray(points.get_offsets())[:, 0].astype(int)
        assert [hex_color(color) for color in points.get_facecolors()] == [
            colors[labels[cell]] for cell in cells
        ]
        subset.close()

        grouped = splt.distribution(
            store, "RNA_nCounts", grouping=clusters, groups=[10, 2], show=False
        )
        assert grouped.scales[0].palette == {10: colors[10], 2: colors[2]}
        grouped.close()
    finally:
        plt.close("all")


def test_explicit_scale_without_colors_keeps_colors_and_applies_labels():
    store = _store()
    scale = splt.CategoricalScale(labels={value: f"<{value}>" for value in CATEGORIES})
    result = splt.embedding(
        store,
        layout_key="umap",
        color_by="cluster",
        groups=["Delta", "Beta"],
        categorical_scale=scale,
        show=False,
    )
    try:
        resolved = _category_scale(result)
        assert resolved.palette == {
            "Delta": FULL_COLORS["Delta"],
            "Beta": FULL_COLORS["Beta"],
        }
        assert [text.get_text() for text in result.figure.legends[0].get_texts()] == [
            "<Delta>",
            "<Beta>",
        ]
    finally:
        result.close()


def test_shared_legend_beside_a_dotplot_keeps_the_full_plot_colors():
    store = _store()
    figure, (left, right) = plt.subplots(1, 2, layout="constrained")
    try:
        children = [
            splt.embedding(
                store,
                layout_key="umap",
                color_by="cluster",
                groups=["Beta", "Alpha"],
                target=left,
                show=False,
            ),
            splt.dotplot(
                store, features=["GeneA"], group_by="cluster", target=right, show=False
            ),
        ]
        splt.compose_results(figure, children, panel_labels=False)
        (legend,) = figure.legends[:1]
        assert [text.get_text() for text in legend.get_texts()] == ["Beta", "Alpha"]
        assert [
            to_hex(handle.get_markerfacecolor()) for handle in legend.legend_handles
        ] == [FULL_COLORS["Beta"], FULL_COLORS["Alpha"]]
    finally:
        plt.close(figure)


def test_category_scale_assigns_colors_over_the_universe_and_orders_requests():
    universe = np.asarray(["c", "a", None, "b", "a"], dtype=object)
    subset = resolve_category_scale(["b"], None, universe=universe)
    assert subset.order == ("b",)
    # "b" is second of a, b, c.
    assert subset.palette == {"b": "#ff7f0e"}

    requested = resolve_category_scale(
        ["a", "c"], None, universe=universe, order=["c", "b", "a"]
    )
    assert requested.order == ("c", "a")
    assert requested.palette == {"c": "#279e68", "a": "#1f77b4"}
    with pytest.raises(ValueError, match="Requested groups are missing observed"):
        resolve_category_scale(["a", "c"], None, universe=universe, order=["c"])

    # A palette needs only the shown categories; an explicit order outranks
    # the universe.
    palette = resolve_category_scale(
        ["a"], splt.CategoricalScale(palette={"a": "#000000"}), universe=universe
    )
    assert palette.palette == {"a": "#000000"}
    explicit = resolve_category_scale(
        ["b"], splt.CategoricalScale(order=("b", "a", "c")), universe=universe
    )
    assert explicit.palette == {"b": "#1f77b4"}

    # Typed arrays give the same categories as object arrays.
    typed = resolve_category_scale(
        np.asarray(["b", "a"]), None, universe=np.asarray(["a", "b", "c"])
    )
    assert typed.order == ("a", "b")
    assert typed.palette == {"a": "#1f77b4", "b": "#ff7f0e"}
    numeric = resolve_category_scale(
        np.asarray([3, 1]), None, universe=np.asarray([1, 2, 3, 3])
    )
    assert numeric.palette == {1: "#1f77b4", 3: "#279e68"}
    # A plain list of tuples holds tuple categories, not rows of an array.
    pairs = resolve_category_scale([("b", 2), None, ("a", 1)], None)
    assert pairs.order == (("a", 1), ("b", 2))


def _labelled_scale(values):
    return splt.CategoricalScale(labels={value: f"<{value}>" for value in values})


def test_distribution_ticks_and_legends_show_category_labels():
    store = _store()
    grouping = splt.CellField("cluster")
    groups = ["Beta", "Alpha"]
    scale = _labelled_scale(CATEGORIES)
    expected = ["<Beta>", "<Alpha>"]
    try:
        for kind in ("violin", "box", "stacked_violin"):
            result = splt.distribution(
                store,
                "metric",
                grouping=grouping,
                groups=groups,
                categorical_scale=scale,
                kind=kind,
                max_points=0,
                show=False,
            )
            ticks = result.axes["metric"].get_xticklabels()
            assert [tick.get_text() for tick in ticks] == expected, kind
        horizontal = splt.distribution(
            store,
            "metric",
            grouping=grouping,
            groups=groups,
            categorical_scale=scale,
            orientation="horizontal",
            max_points=0,
            show=False,
        )
        ticks = horizontal.axes["metric"].get_yticklabels()
        assert [tick.get_text() for tick in ticks] == expected
        for kind in ("hist", "ecdf"):
            result = splt.distribution(
                store,
                "metric",
                grouping=grouping,
                groups=groups,
                categorical_scale=scale,
                kind=kind,
                show=False,
            )
            legend = result.axes["metric"].get_legend()
            assert [text.get_text() for text in legend.get_texts()] == expected, kind

        split = splt.distribution(
            store,
            "metric",
            grouping=grouping,
            split_by="batch",
            split_scale=_labelled_scale(["b1", "b2"]),
            max_points=0,
            show=False,
        )
        legend = split.axes["metric"].get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["<b1>", "<b2>"]
    finally:
        plt.close("all")


def test_summary_plot_ticks_show_category_labels():
    display = {
        "kind": "categorical",
        "categories": [
            {"value": value, "label": value.upper(), "color": FULL_COLORS[value]}
            for value in CATEGORIES
        ],
    }
    store = _store(displays={"cluster": display})
    try:
        for swap_axes in (False, True):
            dots = splt.dotplot(
                store,
                features=["GeneA"],
                group_by="cluster",
                categorical_scale=_labelled_scale(CATEGORIES),
                swap_axes=swap_axes,
                show=False,
            )
            axis = dots.axes["dotplot"]
            ticks = axis.get_yticklabels() if swap_axes else axis.get_xticklabels()
            assert [tick.get_text() for tick in ticks] == [
                f"<{value}>" for value in CATEGORIES
            ]
        # The stored display labels the matrix columns, and an explicit
        # scale replaces them.
        stored = splt.matrixplot(
            store, features=["GeneA"], group_by="cluster", show=False
        )
        assert [
            tick.get_text() for tick in stored.axes["matrixplot"].get_xticklabels()
        ] == [value.upper() for value in CATEGORIES]
        # The matrix table keeps the group values.
        assert stored.tables["matrix"].columns.tolist() == list(CATEGORIES)
        explicit = splt.matrixplot(
            store,
            features=["GeneA"],
            group_by="cluster",
            categorical_scale=splt.CategoricalScale(
                order=tuple(reversed(CATEGORIES)),
                labels={"Alpha": "first"},
            ),
            show=False,
        )
        assert [
            tick.get_text() for tick in explicit.axes["matrixplot"].get_xticklabels()
        ] == [*reversed(CATEGORIES[1:]), "first"]
    finally:
        plt.close("all")
