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


def _paired_profile_store():
    """Groups a and b share one expression profile, c and d another."""
    groups = np.repeat(np.asarray(["a", "b", "c", "d"], dtype=object), 3)
    profiles = {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0], "c": [0.0, 5.0, 4.0]}
    profiles["d"] = profiles["c"]
    values = np.asarray([profiles[group] for group in groups])
    return _ArrayStore(
        {"I": np.ones(len(groups), dtype=bool), "group": groups},
        values,
        names=("GeneA", "GeneB", "GeneC"),
    )


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
