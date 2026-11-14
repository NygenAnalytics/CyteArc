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
# A stored display in developmental order with colors of its own, as an H5AD
# import of the pancreas stores it: neither its order nor its colors are the
# ones CyteArc would generate.
STORED_ORDER = (
    "Ductal",
    "Ngn3 low EP",
    "Ngn3 high EP",
    "Pre-endocrine",
    "Beta",
    "Alpha",
    "Delta",
    "Epsilon",
)
STORED_COLORS = {
    "Ductal": "#8fbc8f",
    "Ngn3 low EP": "#f4a460",
    "Ngn3 high EP": "#fdbf6f",
    "Pre-endocrine": "#ff7f00",
    "Beta": "#b2df8a",
    "Alpha": "#1f78b4",
    "Delta": "#6a3d9a",
    "Epsilon": "#cab2d6",
}


def _stored_display(**missing):
    """The stored display: each category labelled in capitals."""
    return {
        "kind": "categorical",
        "categories": [
            {"value": value, "label": value.upper(), "color": STORED_COLORS[value]}
            for value in STORED_ORDER
        ],
        **missing,
    }


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
        # The panel takes the obs key of the imported clusters as its title.
        (points,) = subset.axes["clusters"].collections
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
    store = _store(displays={"cluster": _stored_display()})
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
            # A scale of labels alone keeps the stored order.
            assert [tick.get_text() for tick in ticks] == [
                f"<{value}>" for value in STORED_ORDER
            ]
        # The stored display orders and labels the matrix columns.
        stored = splt.matrixplot(
            store, features=["GeneA"], group_by="cluster", show=False
        )
        assert [
            tick.get_text() for tick in stored.axes["matrixplot"].get_xticklabels()
        ] == [value.upper() for value in STORED_ORDER]
        # The matrix table keeps the group values.
        assert stored.tables["matrix"].columns.tolist() == list(STORED_ORDER)
        # An explicit order reorders the columns, and an explicit label
        # renames its category while the others keep their stored labels.
        explicit = splt.matrixplot(
            store,
            features=["GeneA"],
            group_by="cluster",
            categorical_scale=splt.CategoricalScale(
                order=tuple(reversed(STORED_ORDER)),
                labels={"Alpha": "first"},
            ),
            show=False,
        )
        assert [
            tick.get_text() for tick in explicit.axes["matrixplot"].get_xticklabels()
        ] == [
            "first" if value == "Alpha" else value.upper()
            for value in reversed(STORED_ORDER)
        ]
    finally:
        plt.close("all")


def _scale_plots(store, scale, **options):
    """The order, colors and labels that each plot of the column draws under ``scale``."""
    drawn = {}
    embedding = splt.embedding(
        store,
        layout_key="umap",
        color_by="cluster",
        categorical_scale=scale,
        legend_loc="right",
        show=False,
        **options,
    )
    legend = embedding.figure.legends[0]
    drawn["embedding"] = [
        (text.get_text(), to_hex(handle.get_markerfacecolor()))
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    ]
    composition = splt.composition(
        store, category_by="cluster", categorical_scale=scale, show=False, **options
    )
    legend = composition.figure.legends[0]
    drawn["composition"] = [
        (text.get_text(), to_hex(handle.get_facecolor()))
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    ]
    violin = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("cluster"),
        categorical_scale=scale,
        max_points=0,
        show=False,
        **options,
    )
    resolved = violin.scales[0]
    drawn["distribution"] = [
        (tick.get_text(), to_hex(resolved.palette[value]))
        for tick, value in zip(
            violin.axes["metric"].get_xticklabels(), resolved.order, strict=True
        )
    ]
    matrix = splt.matrixplot(
        store,
        features=["GeneA"],
        group_by="cluster",
        categorical_scale=scale,
        show=False,
        **options,
    )
    drawn["matrixplot"] = [
        (tick.get_text(), None) for tick in matrix.axes["matrixplot"].get_xticklabels()
    ]
    plt.close("all")
    return drawn


def _expected(order, colors, labels):
    return {
        "embedding": [(labels[value], colors[value]) for value in order],
        "composition": [(labels[value], colors[value]) for value in order],
        "distribution": [(labels[value], colors[value]) for value in order],
        "matrixplot": [(labels[value], None) for value in order],
    }


def test_a_partial_scale_keeps_the_stored_fields_it_leaves_unset():
    store = _store(displays={"cluster": _stored_display()})
    stored_labels = {value: value.upper() for value in STORED_ORDER}

    # Labels alone rename their categories and keep the stored order, colors
    # and the labels of the other categories.
    renamed = splt.CategoricalScale(labels={"Alpha": "Alpha (Gcg)"})
    assert _scale_plots(store, renamed) == _expected(
        STORED_ORDER, STORED_COLORS, {**stored_labels, "Alpha": "Alpha (Gcg)"}
    )

    # An order alone keeps the stored colors and labels.
    order = ("Alpha", "Beta", "Delta", "Epsilon", *STORED_ORDER[:4])
    assert _scale_plots(store, splt.CategoricalScale(order=order)) == _expected(
        order, STORED_COLORS, stored_labels
    )

    # A palette alone keeps the stored order and labels.
    grays = {
        value: f"#{index:02x}{index:02x}{index:02x}"
        for index, value in enumerate(STORED_ORDER, 16)
    }
    assert _scale_plots(store, splt.CategoricalScale(palette=grays)) == _expected(
        STORED_ORDER, grays, stored_labels
    )

    # A palette name replaces the stored colors in the stored order.
    colorblind = dict(
        zip(
            STORED_ORDER,
            (
                "#0072b2",
                "#e69f00",
                "#009e73",
                "#d55e00",
                "#cc79a7",
                "#56b4e9",
                "#f0e442",
                "#000000",
            ),
            strict=True,
        )
    )
    assert _scale_plots(
        store, splt.CategoricalScale(palette_name="colorblind")
    ) == _expected(STORED_ORDER, colorblind, stored_labels)

    # A palette that leaves out a shown category still raises.
    partial = {value: grays[value] for value in STORED_ORDER if value != "Beta"}
    with pytest.raises(KeyError, match="'Beta' missing from palette"):
        splt.composition(
            store,
            category_by="cluster",
            categorical_scale=splt.CategoricalScale(palette=partial),
            show=False,
        )
    plt.close("all")


def test_a_palette_names_only_the_categories_a_plot_shows():
    # The stored order lists Beta, which the cell key leaves out, so a palette
    # of the shown categories colors the plot.
    store = _store(displays={"cluster": _stored_display()})
    shown = [value for value in STORED_ORDER if value != "Beta"]
    palette = {value: STORED_COLORS[value] for value in shown}
    result = splt.composition(
        store,
        category_by="cluster",
        cell_key="no_beta",
        categorical_scale=splt.CategoricalScale(palette=palette),
        show=False,
    )
    try:
        assert _category_scale(result).order == tuple(shown)
        assert [
            to_hex(bar.get_facecolor()) for bar in result.axes["composition"].patches
        ] == [STORED_COLORS[value] for value in shown]
    finally:
        result.close()


def test_requested_groups_order_the_legend_unless_the_caller_sets_an_order():
    store = _store(displays={"cluster": _stored_display()})
    groups = ["Delta", "Ductal", "Alpha"]
    options = {"layout_key": "umap", "color_by": "cluster", "groups": groups}
    try:
        # The stored order does not outrank the requested groups.
        renamed = splt.embedding(
            store,
            categorical_scale=splt.CategoricalScale(labels={"Alpha": "A"}),
            show=False,
            **options,
        )
        assert _category_scale(renamed).order == tuple(groups)
        assert [text.get_text() for text in renamed.figure.legends[0].get_texts()] == [
            "DELTA",
            "DUCTAL",
            "A",
        ]
        assert _category_scale(renamed).palette == {
            value: STORED_COLORS[value] for value in groups
        }
        # A caller's order does.
        ordered = splt.embedding(
            store,
            categorical_scale=splt.CategoricalScale(order=STORED_ORDER),
            show=False,
            **options,
        )
        assert _category_scale(ordered).order == ("Ductal", "Alpha", "Delta")
    finally:
        plt.close("all")


def test_missing_cells_take_the_stored_missing_color_and_label_unless_set():
    labels = np.asarray(["Alpha", None, "Beta", "Alpha", None, "Beta"], dtype=object)
    display = {
        "kind": "categorical",
        "categories": [
            {"value": "Beta", "label": "Beta", "color": "#00aa00"},
            {"value": "Alpha", "label": "Alpha", "color": "#aa0000"},
        ],
        "missing_label": "unassigned",
        "missing_color": "#cccccc",
    }
    store = _bind_store(
        {
            "I": np.ones(6, dtype=bool),
            "umap1": np.arange(6, dtype=np.float64),
            "umap2": np.zeros(6),
            "cluster": labels,
        },
        np.random.default_rng(0),
        {"cluster": display},
    )

    def legend(scale):
        result = splt.composition(
            store, category_by="cluster", categorical_scale=scale, show=False
        )
        try:
            entries = result.figure.legends[0]
            return [
                (text.get_text(), to_hex(handle.get_facecolor()))
                for text, handle in zip(
                    entries.get_texts(), entries.legend_handles, strict=True
                )
            ]
        finally:
            result.close()

    stored = [("Beta", "#00aa00"), ("Alpha", "#aa0000"), ("unassigned", "#cccccc")]
    assert legend(None) == stored
    assert legend(splt.CategoricalScale(labels={"Beta": "B"})) == [
        ("B", "#00aa00"),
        *stored[1:],
    ]
    assert legend(splt.CategoricalScale(missing_label="none")) == [
        *stored[:2],
        ("none", "#cccccc"),
    ]
    # Without a stored display, missing cells take the defaults.
    store._stored_display_metadata = lambda column: None
    assert legend(splt.CategoricalScale(labels={"Beta": "B"}))[-1] == ("NA", "#bdbdbd")


def test_merge_categorical_scales_merges_each_field():
    from cytearc.plotting._display import merge_categorical_scales

    stored = splt.CategoricalScale(
        order=("b", "a"),
        palette={"b": "#000001", "a": "#000002"},
        labels={"b": "Bee", "a": "Ay"},
        missing_color="#000003",
        missing_label="none",
    )
    assert merge_categorical_scales(None, None) is None
    assert merge_categorical_scales(stored, None) is stored
    explicit = splt.CategoricalScale(order=("a", "b"))
    assert merge_categorical_scales(None, explicit) is explicit
    assert merge_categorical_scales(stored, splt.CategoricalScale()) == stored
    assert merge_categorical_scales(
        stored, splt.CategoricalScale(labels={"a": "A"}, missing_color="#000004")
    ) == splt.CategoricalScale(
        order=("b", "a"),
        palette={"b": "#000001", "a": "#000002"},
        labels={"b": "Bee", "a": "A"},
        missing_color="#000004",
        missing_label="none",
    )
    # A palette replaces the stored one whole, and a palette name drops it.
    assert merge_categorical_scales(
        stored, splt.CategoricalScale(palette={"a": "#000005"})
    ).palette == {"a": "#000005"}
    named = merge_categorical_scales(
        stored, splt.CategoricalScale(palette_name="default")
    )
    assert (named.order, named.palette, named.palette_name) == (
        ("b", "a"),
        None,
        "default",
    )
    # A stored scale without labels takes the caller's.
    assert merge_categorical_scales(
        splt.CategoricalScale(order=("b", "a")),
        splt.CategoricalScale(labels={"a": "A"}),
    ) == splt.CategoricalScale(order=("b", "a"), labels={"a": "A"})


def test_categorical_scale_leaves_unset_fields_to_the_plot():
    scale = splt.CategoricalScale()
    assert (scale.missing_color, scale.missing_label, scale.palette_name) == (
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="palette_name must be"):
        splt.CategoricalScale(palette_name="tab20")
    resolved = resolve_category_scale(["b", "a", None], scale)
    assert resolved.order == ("a", "b")
    assert resolved.palette == {"a": "#1f77b4", "b": "#ff7f0e"}
    assert (resolved.missing_color, resolved.missing_label, resolved.palette_name) == (
        "#bdbdbd",
        "NA",
        "default",
    )


def _plotted_orders(store, layout, column, scale=None):
    """The categories and colors that each plot of a live column draws, in order."""
    drawn = {}
    embedding = splt.embedding(
        store,
        layout=layout,
        color_by=column,
        categorical_scale=scale,
        legend_loc="right",
        show=False,
    )
    legend = embedding.figure.legends[0]
    drawn["embedding"] = {
        text.get_text(): to_hex(handle.get_markerfacecolor())
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    }
    composition = splt.composition(
        store, category_by=column, categorical_scale=scale, show=False
    )
    drawn["composition"] = dict(
        zip(
            [text.get_text() for text in composition.figure.legends[0].get_texts()],
            [
                to_hex(handle.get_facecolor())
                for handle in composition.figure.legends[0].legend_handles
            ],
            strict=True,
        )
    )
    violin = splt.distribution(
        store,
        "RNA_nCounts",
        grouping=splt.CellField(column),
        categorical_scale=scale,
        max_points=0,
        show=False,
    )
    drawn["distribution"] = {
        tick.get_text(): to_hex(violin.scales[0].palette[value])
        for tick, value in zip(
            violin.axes["RNA_nCounts"].get_xticklabels(),
            violin.scales[0].order,
            strict=True,
        )
    }
    matrix = splt.matrixplot(
        store, features=["CD3E"], group_by=column, categorical_scale=scale, show=False
    )
    drawn["matrixplot"] = {
        tick.get_text(): None for tick in matrix.axes["matrixplot"].get_xticklabels()
    }
    plt.close("all")
    return {name: list(colors.items()) for name, colors in drawn.items()}


def _in_every_plot(order, colors):
    pairs = list(zip(order, colors, strict=True))
    return {
        "embedding": pairs,
        "composition": pairs,
        "distribution": pairs,
        "matrixplot": [(value, None) for value in order],
    }


def test_a_column_stored_as_ordered_keeps_its_order_in_plots(tmp_path):
    import pandas as pd

    from cytearc.plotting._groups import _colors as _group_colors

    x = np.arange(12, dtype=np.float64)
    store, imported, _counts = _imported_plot_store(
        tmp_path,
        coordinates=np.column_stack((x, (x % 4) * 1.5)),
        clusters=np.repeat([1, 2, 3], 4),
    )
    layout = imported.embeddingArtifacts["X_umap"]
    stages = np.repeat(["late", "early", "mid"], 4)
    # A categorical column whose categories are ordered, one of them unused.
    store.cells.insert(
        "stage",
        pd.Categorical(
            stages, categories=["early", "mid", "late", "post"], ordered=True
        ),
    )
    # A column of labels that records an ordered source categorical, as a
    # Seurat import of an ordered factor writes it.
    store.cells.insert("grade", stages)
    store.zw["cellData"]["grade"].attrs.update(
        {"levels": ["early", "mid", "late"], "ordered": True}
    )
    # The same layouts, unordered, keep the natural order: "2" before "10".
    clusters = np.repeat(["10", "2", "1"], 4)
    store.cells.insert(
        "cluster", pd.Categorical(clusters, categories=["10", "2", "1"], ordered=False)
    )
    store.cells.insert("unordered_grade", stages)
    store.zw["cellData"]["unordered_grade"].attrs.update(
        {"levels": ["early", "mid", "late"], "ordered": False}
    )
    try:
        # Colors follow the stored order, over every stored category.
        stored_colors = ("#1f77b4", "#ff7f0e", "#279e68")
        for column in ("stage", "grade"):
            assert _plotted_orders(store, layout, column) == _in_every_plot(
                ("early", "mid", "late"), stored_colors
            ), column
        assert _plotted_orders(store, layout, "cluster") == _in_every_plot(
            ("1", "2", "10"), stored_colors
        )
        assert _plotted_orders(store, layout, "unordered_grade") == _in_every_plot(
            ("early", "late", "mid"), stored_colors
        )
        # Clone and variant figures color groups of the column in that order.
        groups = _group_colors(
            store, ["late", "mid", "early"], ["grade"], group_by=["grade"], groups=None
        )
        assert groups.order == ("early", "mid", "late")
        assert groups.palette == dict(zip(groups.order, stored_colors, strict=True))
        # A scale of labels alone keeps the stored order.
        renamed = splt.CategoricalScale(labels={"mid": "middle"})
        assert _plotted_orders(store, layout, "stage", renamed) == _in_every_plot(
            ("early", "middle", "late"), stored_colors
        )
    finally:
        plt.close("all")


def test_only_a_live_store_reads_a_column_stored_as_ordered():
    import zarr
    from zarr.storage import MemoryStore

    from cytearc.plotting._display import stored_categorical_scale

    root = zarr.open_group(store=MemoryStore(), mode="w")
    grade = root.create_group("cellData").create_array(
        "grade", data=np.asarray(["low", "high"])
    )
    grade.attrs.update({"levels": ["low", "high"], "ordered": True})

    assert stored_categorical_scale(SimpleNamespace(zw=root), "grade") == (
        splt.CategoricalScale(order=("low", "high"))
    )
    # A run's plot store reads its frozen fields, never the live column.
    frozen = SimpleNamespace(zw=root, _stored_display_metadata=lambda column: None)
    assert stored_categorical_scale(frozen, "grade") is None
    # A store without a root stores no scale.
    assert stored_categorical_scale(SimpleNamespace(), "grade") is None
