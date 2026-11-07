"""Dot and matrix plot layout, stacked composition, and raster gaps."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import cytearc.plotting as splt
from tests.test_plotting_foundation import _ArrayStore
from tests.test_plotting_raster import _GuardedMeta, _layout_store


class _GeneScoreStore(_ArrayStore):
    """The array store, its one assay of gene scores rather than RNA."""

    _defaultAssay = "GeneScores"

    def _get_assay(self, name):
        if name != "GeneScores":
            raise KeyError(name)
        return self.RNA


class _TwoAssayStore(_ArrayStore):
    """The array store, its values read as RNA and as gene scores."""

    def _get_assay(self, name):
        if name not in ("RNA", "GeneScores"):
            raise KeyError(name)
        return self.RNA


def _paired_profile_store(store_type=_ArrayStore):
    """Groups a and b share one expression profile, c and d another."""
    groups = np.repeat(np.asarray(["a", "b", "c", "d"], dtype=object), 3)
    profiles = {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0], "c": [0.0, 5.0, 4.0]}
    profiles["d"] = profiles["c"]
    values = np.asarray([profiles[group] for group in groups])
    return store_type(
        {"I": np.ones(len(groups), dtype=bool), "group": groups},
        values,
        names=("GeneA", "GeneB", "GeneC"),
    )


@pytest.mark.parametrize("plot", [splt.dotplot, splt.matrixplot])
@pytest.mark.parametrize(
    ("assay", "standardize", "label"),
    [
        ("RNA", "none", "Mean expression"),
        ("RNA", "feature", "Standardized mean expression"),
        ("GeneScores", "none", "Mean of GeneScores"),
        ("GeneScores", "feature", "Standardized mean of GeneScores"),
    ],
)
def test_the_colour_bar_names_the_assay_whose_means_it_draws(
    plot, assay, standardize, label
):
    store = _paired_profile_store(
        _GeneScoreStore if assay == "GeneScores" else _ArrayStore
    )
    result = plot(
        store,
        features=["GeneA", "GeneB"],
        group_by="group",
        standardize=standardize,
        show=False,
    )
    try:
        (colorbar,) = [spec for spec in result.legends if spec.kind == "colorbar"]
        assert colorbar.label == label
        drawn = {
            text.get_text()
            for ax in result.figure.axes
            for text in ax.findobj()
            if hasattr(text, "get_text")
        }
        assert label in drawn
    finally:
        result.close()


@pytest.mark.parametrize("plot", [splt.dotplot, splt.matrixplot])
def test_the_colour_bar_names_every_assay_of_features_from_several(plot):
    from cytearc.metadata.selection import FeatureRef

    store = _paired_profile_store(_TwoAssayStore)
    result = plot(
        store,
        features=[
            FeatureRef("GeneA", assay="RNA"),
            FeatureRef("GeneB", assay="GeneScores", label="GeneB score"),
        ],
        group_by="group",
        show=False,
    )
    try:
        (colorbar,) = [spec for spec in result.legends if spec.kind == "colorbar"]
        assert colorbar.label == "Mean of GeneScores and RNA"
    finally:
        result.close()


@pytest.mark.parametrize(
    ("store_type", "assays", "label"),
    [
        (_ArrayStore, ("RNA", "RNA"), "mean expression"),
        (_GeneScoreStore, ("GeneScores", "GeneScores"), "mean of GeneScores"),
        (_TwoAssayStore, ("RNA", "GeneScores"), "mean of GeneScores and RNA"),
    ],
)
def test_the_stacked_violins_colour_bar_names_the_assay_whose_means_it_draws(
    store_type, assays, label
):
    from cytearc.metadata.selection import FeatureRef

    store = _paired_profile_store(store_type)
    result = splt.distribution(
        store,
        [
            FeatureRef(name, assay=assay, label=f"{name} of {assay}")
            for name, assay in zip(("GeneA", "GeneB"), assays, strict=True)
        ],
        grouping=splt.CellField("group"),
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
        max_points=0,
        show=False,
    )
    try:
        (colorbar,) = [spec for spec in result.legends if spec.kind == "colorbar"]
        assert colorbar.label == label
        (drawn,) = [
            ax for ax in result.figure.axes if ax.get_label().startswith("<colorbar")
        ]
        assert drawn.get_ylabel() == label
    finally:
        result.close()


def test_dotplot_with_swapped_axes_reads_groups_from_the_top():
    store = _paired_profile_store()
    result = splt.dotplot(
        store, features=["GeneA"], group_by="group", swap_axes=True, show=False
    )
    try:
        ax = result.axes["dotplot"]
        assert ax.yaxis_inverted()
        assert [tick.get_text() for tick in ax.get_yticklabels()] == list("abcd")
        heights = [ax.transData.transform((0, row))[1] for row in range(4)]
        # The first group sits highest, as in the unswapped plot's reading order.
        assert heights == sorted(heights, reverse=True)
    finally:
        result.close()


def _dendrogram_leaf_pairs(inset, axis):
    """Leaf positions joined at height zero, from the drawn dendrogram."""
    (lines,) = inset.collections
    pairs = []
    for segment in lines.get_segments():
        positions, heights = (
            (segment[:, 0], segment[:, 1])
            if axis == "x"
            else (segment[:, 1], segment[:, 0])
        )
        # A merge is drawn as a cap from one leaf up and back down to another.
        if len(segment) == 4 and np.allclose(heights, 0.0):
            pairs.append(
                tuple(sorted(int(round(position)) for position in positions[[0, 3]]))
            )
    return sorted(pairs)


def test_clustered_matrix_axes_draw_dendrograms_aligned_with_their_leaves():
    from scipy.cluster.hierarchy import leaves_list

    from cytearc.plotting._heatmap_utils import _linkage

    store = _paired_profile_store()
    result = splt.matrixplot(
        store,
        features=["GeneA", "GeneB", "GeneC"],
        group_by="group",
        cluster_groups=True,
        cluster_features=True,
        show=False,
    )
    try:
        matrix = result.tables["matrix"]
        columns = matrix.columns.tolist()
        rows = matrix.index.tolist()
        top = result.axes["column_dendrogram"]
        right = result.axes["row_dendrogram"]
        assert top.get_gid() == "top-dendrogram"
        assert right.get_gid() == "right-dendrogram"
        # The two zero-height merges join the groups that share a profile,
        # at the matrix columns where those groups are drawn.
        joined = [
            {columns[first], columns[second]}
            for first, second in _dendrogram_leaf_pairs(top, "x")
        ]
        assert sorted(joined, key=sorted) == [{"a", "b"}, {"c", "d"}]
        assert top.get_xlim() == result.axes["matrixplot"].get_xlim()
        assert right.get_ylim() == result.axes["matrixplot"].get_ylim()
        # Every drawn leaf lies on a matrix row, in the linkage's leaf order.
        (row_lines,) = right.collections
        leaves = sorted(
            {
                round(float(segment[0, 1]))
                for segment in row_lines.get_segments()
                if segment[0, 0] == 0.0
            }
            | {
                round(float(segment[3, 1]))
                for segment in row_lines.get_segments()
                if segment[3, 0] == 0.0
            }
        )
        assert leaves == [0, 1, 2]
        means = (
            result.tables["aggregate"]
            .pivot(index="feature", columns="group", values="mean")
            .loc[["GeneA", "GeneB", "GeneC"]]
        )
        order = leaves_list(
            _linkage(means.to_numpy(), method="average", metric="euclidean")
        )
        assert rows == [means.index[index] for index in order]
        assert result.provenance.extras["cluster_groups"] is True
    finally:
        result.close()

    plain = splt.matrixplot(
        store, features=["GeneA", "GeneB"], group_by="group", show=False
    )
    try:
        assert set(plain.axes) == {"matrixplot"}
    finally:
        plain.close()


def _clustered_matrixplot(ax):
    return splt.matrixplot(
        _paired_profile_store(),
        features=["GeneA", "GeneB", "GeneC"],
        group_by="group",
        cluster_groups=True,
        cluster_features=True,
        target=ax,
        show=False,
    )


@pytest.mark.parametrize("shared_legends", [True, False])
def test_a_composite_keeps_the_dendrograms_of_a_clustered_matrixplot(shared_legends):
    figure, ax = plt.subplots(layout="constrained")
    try:
        child = _clustered_matrixplot(ax)
        composite = splt.compose_results(
            figure, {"matrix": child}, shared_legends=shared_legends
        )
        figure.canvas.draw()
        # The composite names the insets that the matrix still holds.
        assert set(composite.axes) == {
            ("matrix", "matrixplot"),
            ("matrix", "column_dendrogram"),
            ("matrix", "row_dendrogram"),
        }
        for key in ("column_dendrogram", "row_dendrogram"):
            assert composite.axes[("matrix", key)] is child.axes[key]
            assert child.axes[key] in ax.child_axes
    finally:
        plt.close(figure)


def test_two_composed_matrixplots_keep_their_own_dendrograms():
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    try:
        first = _clustered_matrixplot(left)
        second = _clustered_matrixplot(right)
        composite = splt.compose_results(figure, {"first": first, "second": second})
        for name, child, ax in (("first", first, left), ("second", second, right)):
            for key in ("column_dendrogram", "row_dendrogram"):
                assert composite.axes[(name, key)] is child.axes[key]
                assert child.axes[key] in ax.child_axes
        # Each matrix is one lettered panel, and its dendrograms take none.
        letters = sorted(
            text.get_text()
            for axis in composite.axes.values()
            for text in axis.texts
            if text.get_text() in {"A", "B", "C", "D", "E", "F"}
        )
        assert letters == ["A", "B"]
    finally:
        plt.close(figure)


def test_a_dendrogram_removed_before_composing_is_left_out():
    figure, ax = plt.subplots(layout="constrained")
    try:
        child = _clustered_matrixplot(ax)
        child.axes["row_dendrogram"].remove()
        # The shared colour bar spans the matrix alone, so composing lays out.
        composite = splt.compose_results(figure, {"matrix": child})
        figure.canvas.draw()
        assert set(composite.axes) == {
            ("matrix", "matrixplot"),
            ("matrix", "column_dendrogram"),
        }
    finally:
        plt.close(figure)


def _legend_texts(legend) -> list[str]:
    return [text.get_text() for text in legend.get_texts()]


def test_matrixplot_annotation_legends_list_missing_annotations():
    store = _paired_profile_store()
    annotations = {
        "row_annotations": {"panel": {"GeneA": "x", "GeneB": None, "GeneC": "y"}},
        "column_annotations": {"kind": {"a": "u", "b": "v", "c": "u", "d": "v"}},
    }
    result = splt.matrixplot(
        store,
        features=["GeneA", "GeneB", "GeneC"],
        group_by="group",
        annotation_scales={"panel": splt.CategoricalScale(missing_color="#123456")},
        show=False,
        **annotations,
    )
    try:
        # The grey strip of GeneB has its entry, last among the panel's.
        (legend,) = result.figure.legends
        assert _legend_texts(legend) == [
            "panel: x",
            "panel: y",
            "panel: NA",
            "kind: u",
            "kind: v",
        ]
        assert to_hex(legend.legend_handles[2].get_markerfacecolor()) == "#123456"
        assert [
            (spec.label, spec.extras.get("missing"))
            for spec in result.legends
            if spec.kind == "categorical"
        ] == [("panel", True), ("kind", False)]
    finally:
        result.close()

    figure, ax = plt.subplots(layout="constrained")
    try:
        child = splt.matrixplot(
            store,
            features=["GeneA", "GeneB", "GeneC"],
            group_by="group",
            target=ax,
            show=False,
            **annotations,
        )
        splt.compose_results(figure, [child], panel_labels=False)
        # The composed legends list the missing panel too.
        assert [
            (legend.get_title().get_text(), _legend_texts(legend))
            for legend in figure.legends
        ] == [("panel", ["x", "y", "NA"]), ("kind", ["u", "v"])]
    finally:
        plt.close(figure)


def test_grouped_matrix_features_keep_a_row_per_group_and_bracket_them():
    store = _paired_profile_store()
    features = {"first": ["GeneA", "GeneB"], "second": ["GeneB", "GeneC"]}
    matrix = splt.matrixplot(store, features=features, group_by="group", show=False)
    dots = splt.dotplot(store, features=features, group_by="group", show=False)
    try:
        table = matrix.tables["matrix"]
        assert table.index.names == ["feature_group", "feature"]
        assert table.index.tolist() == [
            ("first", "GeneA"),
            ("first", "GeneB"),
            ("second", "GeneB"),
            ("second", "GeneC"),
        ]
        aggregate = dots.tables["aggregate"].set_index(
            ["feature_group", "feature", "group"]
        )["mean"]
        for (group, feature), row in table.iterrows():
            for column, value in row.items():
                assert value == pytest.approx(aggregate[(group, feature, column)])
        ax = matrix.axes["matrixplot"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "GeneA",
            "GeneB",
            "GeneB",
            "GeneC",
        ]
        brackets = [
            line for line in ax.lines if line.get_gid() == "feature-group-bracket"
        ]
        assert [line.get_ydata().tolist() for line in brackets] == [
            pytest.approx([-0.35, 1.35]),
            pytest.approx([1.65, 3.35]),
        ]
        assert matrix.provenance.extras["feature_group_brackets"] == 2
        assert matrix.provenance.extras["feature_order"] == [
            "GeneA",
            "GeneB",
            "GeneB",
            "GeneC",
        ]
        reordered = splt.matrixplot(
            store,
            features=features,
            group_by="group",
            feature_order=["GeneC", "GeneB", "GeneA"],
            row_annotations={"panel": {"GeneA": "x", "GeneB": "y", "GeneC": "z"}},
            show=False,
        )
        assert reordered.tables["matrix"].index.tolist() == [
            ("second", "GeneC"),
            ("first", "GeneB"),
            ("second", "GeneB"),
            ("first", "GeneA"),
        ]
        assert reordered.tables["row_annotations"]["panel"].tolist() == [
            "z",
            "y",
            "y",
            "x",
        ]
        reordered.close()
        with pytest.raises(ValueError, match="cluster_features would mix the rows"):
            splt.matrixplot(
                store,
                features=features,
                group_by="group",
                cluster_features=True,
                show=False,
            )
    finally:
        matrix.close()
        dots.close()


def test_stacked_composition_puts_the_first_category_on_top():
    store = _ArrayStore(
        {
            "I": np.ones(8, dtype=bool),
            "category": np.asarray(list("aaababbc"), dtype=object),
            "sample": np.repeat(np.asarray(["s0", "s1"], dtype=object), 4),
        },
        np.zeros((8, 2)),
    )
    scale = splt.CategoricalScale(
        palette={"a": "#111111", "b": "#222222", "c": "#333333"}
    )
    single = splt.composition(
        store, category_by="category", categorical_scale=scale, show=False
    )
    per_sample = splt.composition(
        store,
        category_by="category",
        sample_by="sample",
        categorical_scale=scale,
        show=False,
    )
    try:
        bars = single.axes["composition"].patches
        # a, b and c hold 4, 3 and 1 of 8 cells; a is the top segment.
        assert [to_hex(bar.get_facecolor()) for bar in bars] == [
            "#111111",
            "#222222",
            "#333333",
        ]
        assert [(bar.get_y(), bar.get_height()) for bar in bars] == [
            pytest.approx((0.5, 0.5)),
            pytest.approx((0.125, 0.375)),
            pytest.approx((0.0, 0.125)),
        ]
        legend = single.figure.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == ["a", "b", "c"]
        # The stack spans the whole proportion axis.
        assert single.axes["composition"].get_ylim() == (0.0, 1.0)
        assert per_sample.axes["composition"].get_ylim() == (0.0, 1.0)

        bars = per_sample.axes["composition"].patches
        # Sample s0 holds a, a, a, b; s1 holds a, b, b, c.
        assert [(bar.get_y(), bar.get_height()) for bar in bars] == [
            pytest.approx((0.25, 0.75)),
            pytest.approx((0.75, 0.25)),
            pytest.approx((0.0, 0.25)),
            pytest.approx((0.25, 0.5)),
            pytest.approx((0.0, 0.0)),
            pytest.approx((0.0, 0.25)),
        ]
    finally:
        single.close()
        per_sample.close()


@pytest.mark.parametrize(
    ("kind", "sample_by"),
    [("stacked", None), ("stacked", "sample"), ("per_sample", "sample")],
)
def test_a_composed_composition_lists_its_missing_category(kind, sample_by):
    store = _ArrayStore(
        {
            "I": np.ones(12, dtype=bool),
            "category": np.asarray(["a", "b", "c", None] * 3, dtype=object),
            "sample": np.repeat(np.asarray(["s0", "s1", "s2"], dtype=object), 4),
        },
        np.zeros((12, 2)),
    )
    figure, ax = plt.subplots(layout="constrained")
    try:
        child = splt.composition(
            store,
            category_by="category",
            sample_by=sample_by,
            kind=kind,
            target=ax,
            show=False,
        )
        assert child.legends[0].extras["missing"] is True
        splt.compose_results(figure, [child], panel_labels=False)
        # The composed legend keeps the NA category, last and in the missing
        # grey, as the composition draws it.
        (legend,) = figure.legends
        assert legend.get_title().get_text() == "category"
        assert [text.get_text() for text in legend.get_texts()] == [
            "a",
            "b",
            "c",
            "NA",
        ]
        assert to_hex(legend.legend_handles[-1].get_markerfacecolor()) == "#bdbdbd"
    finally:
        plt.close(figure)


def test_a_composition_without_missing_categories_lists_no_na():
    store = _ArrayStore(
        {"I": np.ones(6, dtype=bool), "category": np.asarray(list("aabbcc"))},
        np.zeros((6, 2)),
    )
    figure, ax = plt.subplots(layout="constrained")
    try:
        child = splt.composition(store, category_by="category", target=ax, show=False)
        assert child.legends[0].extras["missing"] is False
        splt.compose_results(figure, [child], panel_labels=False)
        (legend,) = figure.legends
        assert [text.get_text() for text in legend.get_texts()] == ["a", "b", "c"]
    finally:
        plt.close(figure)


def test_raster_paints_pixels_whose_cells_lack_a_value():
    from cytearc.plotting._raster import raster_from_metadata

    rng = np.random.default_rng(1)
    n = 4_000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    values = rng.random(n)
    values[x > 1] = np.nan
    columns = {"I": np.ones(n, dtype=bool), "U1": x, "U2": y, "v": values}

    canvas = raster_from_metadata(
        _GuardedMeta(columns, chunk=512),
        x_key="U1",
        y_key="U2",
        color_key="v",
        pixels=32,
        block_rows=512,
        quantiles=None,
    )
    xmin, xmax, ymin, ymax = canvas.extent
    bins = {"bins": 32, "range": ((xmin, xmax), (ymin, ymax))}
    occupied = np.flipud(np.histogram2d(x, y, **bins)[0].T) > 0
    valued = np.isfinite(values)
    with_value = np.flipud(np.histogram2d(x[valued], y[valued], **bins)[0].T) > 0
    np.testing.assert_array_equal(canvas.missing, occupied & ~with_value)
    assert canvas.missing.any()
    assert canvas.n_cells == n
    np.testing.assert_array_equal(np.isnan(canvas.image), ~with_value)

    open_figures = plt.get_fignums()
    result = splt.embedding_raster(
        _layout_store(columns, chunk=512),
        layout_key="U",
        color_by="v",
        pixels=32,
        show=False,
    )
    try:
        image, missing = result.axes["v"].get_images()
        overlay = missing.get_array()
        painted = overlay[..., 3] > 0
        np.testing.assert_array_equal(painted, canvas.missing)
        assert {to_hex(color) for color in overlay[painted]} == {"#bdbdbd"}
        assert result.provenance.extras["missing_color"] == "#bdbdbd"
    finally:
        result.close()

    density = raster_from_metadata(
        _GuardedMeta(columns, chunk=512),
        x_key="U1",
        y_key="U2",
        pixels=32,
    )
    # A density raster counts every cell, so none lacks a value.
    assert density.missing is None
    # Closing the plot leaves no figure of this test open.
    assert plt.get_fignums() == open_figures
