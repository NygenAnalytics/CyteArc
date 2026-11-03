"""The charts that clone figures draw, on hand-made tables whose drawing is worked out by hand."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgba

from cytearc.plotting._charts import draw_bars, draw_lines, draw_matrix, pair_matrix
from cytearc.plotting._contracts import CategoricalScale, ColorScale
from cytearc.plotting._figure import side_legend_title
from cytearc.plotting._groups import PlotGroups
from cytearc.plotting._heatmap_utils import draw_dendrogram
from cytearc.plotting._style import (
    LEGEND_SIDE_MAX_ENTRIES,
    MAX_NAMED_ITEMS,
    MAX_PRINTED_VALUES,
    RECEPTOR_TYPE_SCALE,
    figure_size,
    item_inches,
)
from cytearc.repertoire.chains import RECEPTOR_TYPES

pytestmark = pytest.mark.usefixtures("figures_closed")

RED, GREEN, BLUE = "#cc0000", "#00aa00", "#0000cc"
SEGMENTS = CategoricalScale(
    order=("s1", "s2", "s3"),
    palette={"s1": RED, "s2": GREEN, "s3": BLUE},
)


@pytest.fixture
def ax():
    figure, axis = plt.subplots(figsize=(6, 4))
    yield axis
    plt.close(figure)


def _stacked() -> pd.DataFrame:
    # Item a holds 1, 2 and 3 of the three segments, and item b 4 and 6 of
    # the first two; its third segment is missing, so 0.
    return pd.DataFrame(
        {
            "item": ["a", "a", "a", "b", "b"],
            "segment": ["s3", "s1", "s2", "s1", "s2"],
            "value": [3, 1, 2, 4, 6],
        }
    )


def _bars(
    ax, *, horizontal: bool = True
) -> dict[tuple[float, str], tuple[float, float]]:
    """Each bar by its item's position and its colour: its start and its length."""
    found = {}
    for patch in ax.patches:
        x, y = patch.get_xy()
        color = to_hex(patch.get_facecolor())
        if horizontal:
            found[(round(y + patch.get_height() / 2, 6), color)] = (
                x,
                patch.get_width(),
            )
        else:
            found[(round(x + patch.get_width() / 2, 6), color)] = (
                y,
                patch.get_height(),
            )
    return found


def _legend(ax) -> tuple[str, list[str], list[str]]:
    legend = ax.get_legend()
    return (
        legend.get_title().get_text(),
        [text.get_text() for text in legend.get_texts()],
        [to_hex(handle.get_facecolor()) for handle in legend.legend_handles],
    )


def test_horizontal_bars_stack_their_segments_from_the_left_in_the_legends_order(
    ax,
) -> None:
    drawn = draw_bars(
        ax,
        _stacked(),
        item="item",
        value="value",
        segment="segment",
        colors=SEGMENTS,
        value_label="Cells",
        legend_title="Segment",
    )
    bars = _bars(ax)
    # Item a, the first row, is on top at 1 and b below at 0. Each part
    # starts where the parts before it in the scale's order end.
    assert bars[(1.0, RED)] == (0.0, 1.0)
    assert bars[(1.0, GREEN)] == (1.0, 2.0)
    assert bars[(1.0, BLUE)] == (3.0, 3.0)
    assert bars[(0.0, RED)] == (0.0, 4.0)
    assert bars[(0.0, GREEN)] == (4.0, 6.0)
    assert bars[(0.0, BLUE)] == (10.0, 0.0)
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["a", "b"]
    assert ax.get_xlabel() == "Cells"
    # The legend lists the segments in their stacking order, each in the
    # colour of its bars, and the result reports it with its scale.
    assert _legend(ax) == ("Segment", ["s1", "s2", "s3"], [RED, GREEN, BLUE])
    assert drawn.legend.kind == "categorical" and drawn.legend.label == "Segment"
    assert drawn.scale.order == ("s1", "s2", "s3")
    assert drawn.scale.palette == SEGMENTS.palette


def test_vertical_bars_stack_their_first_segment_on_top(ax) -> None:
    draw_bars(
        ax,
        _stacked(),
        item="item",
        value="value",
        segment="segment",
        colors=SEGMENTS,
        horizontal=False,
        value_label="Cells",
        legend_title="Segment",
    )
    bars = _bars(ax, horizontal=False)
    # Item a is first, at the left; its last segment sits at the bottom and
    # its first on top, as the legend lists them from the top.
    assert bars[(0.0, BLUE)] == (0.0, 3.0)
    assert bars[(0.0, GREEN)] == (3.0, 2.0)
    assert bars[(0.0, RED)] == (5.0, 1.0)
    assert bars[(1.0, GREEN)] == (0.0, 6.0)
    assert bars[(1.0, RED)] == (6.0, 4.0)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["a", "b"]
    assert _legend(ax)[1] == ["s1", "s2", "s3"]


def test_normalized_bars_draw_each_items_fractions(ax) -> None:
    table = pd.concat(
        [_stacked(), pd.DataFrame({"item": ["c"], "segment": ["s1"], "value": [0]})]
    )
    draw_bars(
        ax,
        table,
        item="item",
        value="value",
        segment="segment",
        colors=SEGMENTS,
        normalize=True,
        value_label="Fraction of cells",
    )
    bars = _bars(ax)
    # a holds 1, 2 and 3 of 6, b 4 and 6 of 10, and c no cell, drawn as 0.
    assert [bars[(2.0, color)][1] for color in (RED, GREEN, BLUE)] == pytest.approx(
        [1 / 6, 2 / 6, 3 / 6]
    )
    assert [bars[(1.0, color)][1] for color in (RED, GREEN, BLUE)] == pytest.approx(
        [0.4, 0.6, 0.0]
    )
    assert [bars[(0.0, color)][1] for color in (RED, GREEN, BLUE)] == [0, 0, 0]
    # Fractions span the whole axis.
    assert ax.get_xlim() == (0.0, 1.0)
    # Without a legend title, no legend is drawn.
    assert ax.get_legend() is None


def test_a_value_that_is_not_a_number_draws_no_bar_and_shifts_none(ax) -> None:
    table = pd.DataFrame(
        {"item": ["a", "a"], "segment": ["s1", "s2"], "value": [np.nan, 2.0]}
    )
    draw_bars(
        ax,
        table,
        item="item",
        value="value",
        segment="segment",
        colors=SEGMENTS,
        value_label="Cells",
    )
    bars = _bars(ax)
    assert np.isnan(bars[(0.0, RED)][1])
    # The second part starts at 0, where the first would have ended.
    assert bars[(0.0, GREEN)] == (0.0, 2.0)


def test_plain_bars_take_one_colour_or_their_items_colours(ax) -> None:
    table = pd.DataFrame({"group": ["y", "x"], "value": [2.0, 5.0]})
    assert (
        draw_bars(
            ax, table, item="group", value="value", colors="#333333", value_label=""
        )
        is None
    )
    assert {to_hex(patch.get_facecolor()) for patch in ax.patches} == {"#333333"}
    ax.clear()
    groups = CategoricalScale(order=("x", "y"), palette={"x": RED, "y": BLUE})
    drawn = draw_bars(
        ax,
        table,
        item="group",
        value="value",
        colors=groups,
        value_label="",
        legend_title="group",
    )
    # Bars follow the table, y on top, each in its item's colour; the legend
    # lists the items in the scale's order.
    bars = _bars(ax)
    assert bars[(1.0, BLUE)][1] == 2.0 and bars[(0.0, RED)][1] == 5.0
    assert _legend(ax) == ("group", ["x", "y"], [RED, BLUE])
    assert drawn.scale.order == ("x", "y")


@pytest.mark.parametrize("n_items", [MAX_NAMED_ITEMS, MAX_NAMED_ITEMS + 1])
def test_bars_name_at_most_40_items(ax, n_items) -> None:
    table = pd.DataFrame(
        {"item": [f"i{at}" for at in range(n_items)], "value": np.ones(n_items)}
    )
    draw_bars(ax, table, item="item", value="value", colors="#333333", value_label="")
    names = [tick.get_text() for tick in ax.get_yticklabels()]
    assert names == ([f"i{at}" for at in range(n_items)] if n_items <= 40 else [])
    assert len(ax.patches) == n_items


@pytest.mark.parametrize(
    ("n_segments", "title"),
    [
        (LEGEND_SIDE_MAX_ENTRIES, "Segment"),
        (LEGEND_SIDE_MAX_ENTRIES + 1, "Segment (80 of 81)"),
        (120, "Segment (80 of 120)"),
    ],
)
def test_bar_legends_list_at_most_80_entries_and_say_so(ax, n_segments, title) -> None:
    names = tuple(f"s{at:03d}" for at in range(n_segments))
    scale = CategoricalScale(order=names, palette=dict.fromkeys(names, RED))
    table = pd.DataFrame(
        {"item": ["a"] * n_segments, "segment": list(names), "value": 1.0}
    )
    drawn = draw_bars(
        ax,
        table,
        item="item",
        value="value",
        segment="segment",
        colors=scale,
        value_label="",
        legend_title="Segment",
    )
    shown = min(n_segments, LEGEND_SIDE_MAX_ENTRIES)
    found_title, labels, _colors = _legend(ax)
    assert found_title == title == drawn.legend.label
    assert labels == list(names[:shown])
    assert drawn.scale.order == names[:shown]
    # Every segment is drawn, those the legend leaves out included.
    assert len(ax.patches) == n_segments


def test_a_hidden_legend_is_still_reported_with_its_scale(ax) -> None:
    drawn = draw_bars(
        ax,
        _stacked(),
        item="item",
        value="value",
        segment="segment",
        colors=SEGMENTS,
        value_label="",
        legend_title="Segment",
        show_legend=False,
    )
    assert ax.get_legend() is None
    assert drawn.legend.label == "Segment" and drawn.scale.order == SEGMENTS.order


def test_bars_refuse_what_they_cannot_draw(ax) -> None:
    with pytest.raises(ValueError, match="needs a row to draw"):
        draw_bars(
            ax,
            _stacked().iloc[:0],
            item="item",
            value="value",
            colors=RED,
            value_label="",
        )
    table = _stacked().assign(segment=["s3", "s1", "s2", "s1", "s9"])
    with pytest.raises(ValueError, match="give no colour to 's9'"):
        draw_bars(
            ax,
            table,
            item="item",
            value="value",
            segment="segment",
            colors=SEGMENTS,
            value_label="",
        )
    twice = _stacked().assign(segment=["s1", "s1", "s2", "s1", "s2"])
    with pytest.raises(ValueError, match="takes each item once in each"):
        draw_bars(
            ax,
            twice,
            item="item",
            value="value",
            segment="segment",
            colors=SEGMENTS,
            value_label="",
        )
    with pytest.raises(TypeError, match="Stacked bars take a CategoricalScale"):
        draw_bars(
            ax,
            _stacked(),
            item="item",
            value="value",
            segment="segment",
            colors=RED,
            value_label="",
        )
    unnamed = CategoricalScale(order=("x",), palette={"x": RED})
    with pytest.raises(ValueError, match="give no colour to 'a'"):
        draw_bars(
            ax, _stacked(), item="item", value="value", colors=unnamed, value_label=""
        )


def test_charts_refuse_scales_without_an_order_or_a_colour(ax) -> None:
    table = pd.DataFrame({"group": ["x"], "value": [1.0]})
    with pytest.raises(ValueError, match="need an order and a palette"):
        draw_bars(
            ax,
            table,
            item="group",
            value="value",
            colors=CategoricalScale(palette={"x": RED}),
            value_label="",
        )
    with pytest.raises(ValueError, match="give no colour to 'y'"):
        draw_bars(
            ax,
            table,
            item="group",
            value="value",
            colors=CategoricalScale(order=("x", "y"), palette={"x": RED}),
            value_label="",
        )


SERIES = CategoricalScale(order=("q", "p"), palette={"p": RED, "q": BLUE})


def _curves() -> pd.DataFrame:
    # Series p rises through x = 1, 2 and 3, the last two extrapolated, and
    # its observed point is at 1; q runs through 1 and 2.
    return pd.DataFrame(
        {
            "series": ["p", "p", "p", "q", "q"],
            "x": [1.0, 2.0, 3.0, 1.0, 2.0],
            "y": [10.0, 20.0, 30.0, 5.0, 6.0],
            "dashed": [False, True, True, False, False],
            "marked": [True, False, False, False, True],
        }
    )


def test_lines_follow_the_scale_with_dashed_parts_and_marked_points(ax) -> None:
    drawn = draw_lines(
        ax,
        _curves(),
        x="x",
        y="y",
        series="series",
        colors=SERIES,
        dashed="dashed",
        marked="marked",
        y_label="Hill number",
        legend_title="group",
    )
    lines = ax.get_lines()
    # Each series in the scale's order, q first: its solid line, its dashed
    # line from its last solid point, and its marked points.
    assert len(lines) == 6
    q_solid, q_dashed, q_marked, p_solid, p_dashed, p_marked = lines
    assert p_solid.get_xdata().tolist() == [1.0] and p_solid.get_label() == "p"
    assert p_dashed.get_xdata().tolist() == [1.0, 2.0, 3.0]
    assert p_dashed.get_ydata().tolist() == [10.0, 20.0, 30.0]
    assert p_dashed.get_linestyle() == "--"
    assert p_marked.get_xdata().tolist() == [1.0] and p_marked.get_marker() == "o"
    assert q_solid.get_xdata().tolist() == [1.0, 2.0]
    # With nothing dashed, the dashed line is the last solid point alone.
    assert q_dashed.get_xdata().tolist() == [2.0]
    assert q_marked.get_xdata().tolist() == [2.0]
    assert {to_hex(line.get_color()) for line in (q_solid, q_dashed, q_marked)} == {
        BLUE
    }
    assert {to_hex(line.get_color()) for line in (p_solid, p_dashed, p_marked)} == {RED}
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["q", "p"]
    assert [to_hex(handle.get_color()) for handle in legend.legend_handles] == [
        BLUE,
        RED,
    ]
    assert drawn.scale.order == ("q", "p")


def test_lines_along_named_positions_leave_gaps(ax) -> None:
    table = pd.DataFrame(
        {"clone": ["p", "p", "q"], "group": ["d4", "d2", "d7"], "y": [0.5, 0.25, 1.0]}
    )
    draw_lines(
        ax,
        table,
        x="group",
        y="y",
        series="clone",
        colors=SERIES,
        positions=["d2", "d4", "d7"],
        markers=True,
        y_label="Fraction",
    )
    q_line, p_line = ax.get_lines()
    assert p_line.get_xdata().tolist() == [0, 1, 2]
    assert p_line.get_ydata().tolist()[:2] == [0.25, 0.5]
    assert np.isnan(p_line.get_ydata()[2])
    assert np.isnan(q_line.get_ydata()[:2]).all() and q_line.get_ydata()[2] == 1.0
    assert p_line.get_marker() == "o"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["d2", "d4", "d7"]
    assert ax.get_legend() is None
    with pytest.raises(ValueError, match="do not name 'd9'"):
        draw_lines(
            ax,
            table.assign(group=["d4", "d2", "d9"]),
            x="group",
            y="y",
            series="clone",
            colors=SERIES,
            positions=["d2", "d4", "d7"],
            y_label="",
        )
    with pytest.raises(ValueError, match="takes a position twice"):
        draw_lines(
            ax,
            table.assign(group=["d4", "d4", "d7"]),
            x="group",
            y="y",
            series="clone",
            colors=SERIES,
            positions=["d2", "d4", "d7"],
            y_label="",
        )
    with pytest.raises(ValueError, match="take neither dashed nor marked"):
        draw_lines(
            ax,
            _curves(),
            x="x",
            y="y",
            series="series",
            colors=SERIES,
            positions=["1"],
            dashed="dashed",
            y_label="",
        )
    with pytest.raises(ValueError, match="give no colour to 'r'"):
        draw_lines(
            ax,
            table.assign(clone=["p", "r", "q"]),
            x="group",
            y="y",
            series="clone",
            colors=SERIES,
            positions=["d2", "d4", "d7"],
            y_label="",
        )


@pytest.mark.parametrize("n_positions", [MAX_NAMED_ITEMS, MAX_NAMED_ITEMS + 1])
def test_lines_name_at_most_40_positions(ax, n_positions) -> None:
    positions = [f"g{at}" for at in range(n_positions)]
    table = pd.DataFrame({"clone": "p", "group": positions, "y": 1.0})
    draw_lines(
        ax,
        table,
        x="group",
        y="y",
        series="clone",
        colors=SERIES,
        positions=positions,
        y_label="",
    )
    names = [tick.get_text() for tick in ax.get_xticklabels()]
    assert names == (positions if n_positions <= 40 else [])


def test_log_axes_of_counts_read_whole_numbers(ax) -> None:
    table = pd.DataFrame({"series": "p", "x": [1, 2, 3], "y": [5, 2, 1]})
    draw_lines(
        ax,
        table,
        x="x",
        y="y",
        series="series",
        colors=SERIES,
        log_x=True,
        log_y=True,
        y_label="Clones",
    )
    ax.figure.canvas.draw()
    labels = [
        tick.get_text()
        for tick in [*ax.get_xticklabels(), *ax.get_xticklabels(minor=True)]
        if tick.get_text()
    ]
    # Plain numbers, never 2x10^0, and nothing below 1.
    assert "1" in labels and "2" in labels and "3" in labels
    assert all(label.replace(",", "").isdigit() for label in labels)
    formatter = ax.xaxis.get_major_formatter()
    assert formatter(0.5) == "" and formatter(10_000) == "10,000"


@pytest.mark.parametrize(
    ("n_series", "title"), [(80, "Clone"), (81, "Clone (80 of 81)")]
)
def test_line_legends_list_at_most_80_series(ax, n_series, title) -> None:
    names = tuple(f"c{at:03d}" for at in range(n_series))
    scale = CategoricalScale(order=names, palette=dict.fromkeys(names, RED))
    table = pd.DataFrame({"series": list(names), "x": 0.0, "y": 1.0})
    drawn = draw_lines(
        ax,
        table,
        x="x",
        y="y",
        series="series",
        colors=scale,
        y_label="",
        legend_title="Clone",
    )
    legend = ax.get_legend()
    assert legend.get_title().get_text() == title == drawn.legend.label
    assert [text.get_text() for text in legend.get_texts()] == list(names[:80])
    assert len(ax.get_lines()) == n_series


def test_side_legend_titles_count_what_they_leave_out() -> None:
    assert side_legend_title("Clone", 80) == "Clone"
    assert side_legend_title("Clone", 81) == "Clone (80 of 81)"
    assert side_legend_title("Clone", 120) == "Clone (80 of 120)"
    assert side_legend_title("Clone", 1) == "Clone"


def test_pairs_fill_both_halves_of_their_matrix() -> None:
    pairs = pd.DataFrame(
        {"group_1": ["a", "a"], "group_2": ["b", "c"], "value": [0.5, 0.25]}
    )
    matrix = pair_matrix(pairs, ["a", "b", "c"], "value", diagonal=1.0)
    expected = [[1.0, 0.5, 0.25], [0.5, 1.0, np.nan], [0.25, np.nan, 1.0]]
    np.testing.assert_array_equal(matrix.to_numpy(), expected)
    assert matrix.index.tolist() == matrix.columns.tolist() == ["a", "b", "c"]
    sizes = pair_matrix(pairs, ["a", "b", "c"], "value", diagonal=np.array([3, 4, 5]))
    assert np.diag(sizes).tolist() == [3, 4, 5]
    with pytest.raises(ValueError, match="names a group the names lack"):
        pair_matrix(pairs, ["a", "b"], "value")


def _matrix() -> pd.DataFrame:
    # b and c were not compared, and their pair is NaN.
    return pd.DataFrame(
        [[9.0, 0.0, 1.0], [0.0, 9.0, np.nan], [1.0, np.nan, 9.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )


def test_heatmaps_colour_through_their_scale_and_blank_the_diagonal(ax) -> None:
    scale = ColorScale(cmap="viridis", missing_color="#ff00ff")
    drawn = draw_matrix(
        ax,
        _matrix(),
        color_scale=scale,
        value_label="Jaccard index",
        blank_diagonal=True,
    )
    image, missing = ax.images
    # The limits span the pairs, not the diagonal's 9.
    assert image.get_clim() == (0.0, 1.0) and image.cmap.name == "viridis"
    data = image.get_array()
    assert data.mask.tolist() == [
        [True, False, False],
        [False, True, True],
        [False, True, True],
    ]
    # The pair without a value takes the missing colour, and the diagonal
    # is blank in both layers.
    layer = missing.get_array()
    assert tuple(layer[1, 2]) == tuple(layer[2, 1]) == to_rgba("#ff00ff")
    assert layer[0, 0][3] == layer[1, 1][3] == 0.0 and layer[0, 1][3] == 0.0
    # Each coloured value is printed, dark on the light top of viridis and
    # light on its dark bottom.
    texts = {text.get_position(): text for text in ax.texts}
    assert sorted(texts) == [(0, 2), (1, 0), (2, 0)] or sorted(texts) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
    ]
    assert texts[(2, 0)].get_text() == "1" and texts[(2, 0)].get_color() == "black"
    assert texts[(1, 0)].get_text() == "0" and texts[(1, 0)].get_color() == "white"
    # The colour bar names the value and the result reports its limits.
    colorbar = ax.figure.axes[-1]
    assert colorbar.get_ylabel() == "Jaccard index"
    assert drawn.legend.kind == "colorbar"
    assert drawn.legend.extras == {"vmin": 0.0, "vmax": 1.0}
    assert drawn.scale is scale


def test_heatmaps_take_their_scales_limits_and_centre(ax) -> None:
    scale = ColorScale(cmap="RdBu_r", vmin=-2.0, vmax=2.0, vcenter=0.0)
    draw_matrix(
        ax, _matrix() - 0.5, color_scale=scale, value_label="", show_legend=False
    )
    image = ax.images[0]
    assert image.get_clim() == (-2.0, 2.0) and image.cmap.name == "RdBu_r"
    # 0 sits at the middle of the map.
    assert image.norm(0.0) == pytest.approx(0.5)
    # Without the legend, no colour bar is drawn.
    assert len(ax.figure.axes) == 1


@pytest.mark.parametrize("n_groups", [MAX_PRINTED_VALUES, MAX_PRINTED_VALUES + 1])
def test_heatmaps_print_the_values_of_at_most_12_rows(ax, n_groups) -> None:
    names = [f"g{at}" for at in range(n_groups)]
    matrix = pd.DataFrame(np.ones((n_groups, n_groups)), index=names, columns=names)
    draw_matrix(
        ax,
        matrix,
        color_scale=ColorScale(cmap="viridis"),
        value_label="",
        blank_diagonal=True,
        show_legend=False,
    )
    assert len(ax.texts) == (n_groups * (n_groups - 1) if n_groups <= 12 else 0)


@pytest.mark.parametrize("n_groups", [MAX_NAMED_ITEMS, MAX_NAMED_ITEMS + 1])
def test_heatmaps_name_at_most_40_rows(ax, n_groups) -> None:
    names = [f"g{at}" for at in range(n_groups)]
    matrix = pd.DataFrame(np.ones((n_groups, n_groups)), index=names, columns=names)
    draw_matrix(ax, matrix, color_scale=ColorScale(), value_label="", show_legend=False)
    named = [tick.get_text() for tick in ax.get_yticklabels()]
    assert named == (names if n_groups <= 40 else [])
    assert [tick.get_text() for tick in ax.get_xticklabels()] == named


def test_a_trees_leaves_stand_over_the_matrixs_columns(ax) -> None:
    from scipy.cluster.hierarchy import leaves_list, linkage

    rng = np.random.default_rng(3)
    points = rng.normal(size=(5, 2))
    tree = linkage(points, method="average")
    order = leaves_list(tree)
    names = [f"g{at}" for at in order]
    matrix = pd.DataFrame(np.corrcoef(points[order]), index=names, columns=names)
    draw_matrix(
        ax,
        matrix,
        color_scale=ColorScale(cmap="RdBu_r", vcenter=0.0),
        value_label="",
        show_legend=False,
    )
    inset = draw_dendrogram(ax, tree, side="top", color="#333333")
    segments = np.concatenate(inset.collections[0].get_segments())
    # The tree's leaves, at height 0, stand at the matrix's column centres.
    leaves = sorted(set(segments[segments[:, 1] == 0, 0].tolist()))
    assert leaves == list(range(5))
    assert inset.get_xlim() == ax.get_xlim()


def test_item_extents_stop_growing_where_names_stop() -> None:
    assert item_inches(2, per_item=0.5, base=1.0, minimum=3.0) == 3.0
    assert item_inches(10, per_item=0.5, base=1.0, minimum=3.0) == 6.0
    assert item_inches(MAX_NAMED_ITEMS, per_item=0.5, base=1.0, minimum=3.0) == 21.0
    assert item_inches(500, per_item=0.5, base=1.0, minimum=3.0) == 21.0


def test_figures_stop_at_the_page_width() -> None:
    # One column of panels narrows to the page and keeps its height.
    assert figure_size(2, panel=(12.0, 3.0)) == ((7.5, 6.0), 1)
    # A row of panels shrinks them to share the page, keeping their shape.
    size, columns = figure_size(4, panel=(4.2, 3.6), columns=3)
    assert columns == 3 and size == pytest.approx((7.5, 2 * 3.6 * 7.5 / 12.6))
    assert figure_size(2, panel=(3.0, 2.0), columns=3) == ((6.0, 2.0), 2)


def test_plot_groups_keep_each_groups_colour_in_every_subset() -> None:
    every = PlotGroups(
        names=("d2", "d4", "d7"),
        columns=("sample",),
        codes=np.empty(0, dtype=np.int64),
        colors=CategoricalScale(
            order=("d2", "d4", "d7"), palette={"d2": RED, "d4": GREEN, "d7": BLUE}
        ),
    )
    assert every.title == "sample"
    subset = every.scale(["d7", "d4"])
    # The colours' order, and each group's colour, whatever is drawn.
    assert subset.order == ("d4", "d7")
    assert subset.palette == {"d4": GREEN, "d7": BLUE}
    with pytest.raises(ValueError, match="no group named 'd9'"):
        every.scale(["d9"])
    combined = PlotGroups(
        names=("d2, x", "d4, x"),
        columns=("sample", "type"),
        codes=np.empty(0, dtype=np.int64),
        colors=None,
    )
    # Groups of several columns take colours over the groups drawn.
    assert combined.title == "sample, type"
    assert combined.scale(["d4, x"]).palette == {"d4, x": "#1f77b4"}


def test_receptor_types_keep_one_palette_in_their_vocabularys_order() -> None:
    assert RECEPTOR_TYPE_SCALE.order == RECEPTOR_TYPES
    assert set(RECEPTOR_TYPE_SCALE.palette) == set(RECEPTOR_TYPES)
