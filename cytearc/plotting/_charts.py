"""Charts that the figures of clones, receptors, lineages and variants draw from tidy tables.

Each chart draws one panel. It names at most ``MAX_NAMED_ITEMS`` items on an
axis, a heatmap as many as its laid-out axis holds, draws a titled legend
beside the panel when given a legend title, and returns what that legend
shows with the scale behind it, a :class:`Drawn`, which ``finish_plot``
reports so that ``compose_results`` can draw a legend that several figures
share once.
"""

import functools
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..utils.arguments import integer_argument, quoted_value
from ..utils.arrays import has_duplicates
from ._contracts import CategoricalScale, ColorScale
from ._deps import require_matplotlib
from ._figure import Drawn, LegendSpec, side_legend, side_legend_title
from ._style import (
    LEGEND_SIDE_MAX_ENTRIES,
    MAX_NAMED_ITEMS,
    MAX_PRINTED_VALUES,
    category_label,
    continuous_norm,
    resolve_color_limits,
    scalable_values,
)

__all__ = [
    "capped_counts",
    "draw_bars",
    "draw_histograms",
    "draw_lines",
    "draw_matrix",
    "draw_tree",
    "legend_of",
    "log_values",
    "pair_matrix",
    "tree_drawing_bytes",
]

# Bars fill this share of each item's slot, lines are this wide, and the
# points on them this large.
_BAR_EXTENT = 0.8
_LINE_WIDTH = 1.4
_POINT_SIZE = 3.5
_MARKED_POINT_SIZE = 4.0
# The room that a heatmap's row or column name takes along its axis, in
# multiples of the size of its text.
_NAME_PITCH = 1.4
# A tree's wedge fills this share of its slot, so that neighbours stay apart,
# its lines are this wide, and a wedge that ends where it starts, such as one
# leaf, is a point of this area.
_WEDGE_EXTENT = 0.9
_TREE_LINE_WIDTH = 0.8
_TREE_POINT_AREA = 9.0
# What drawing a tree holds: each legend entry; each text; and for each node
# drawn its branch, wedge and colour, and the figure's table and labels of it.
_TREE_BYTES_PER_ENTRY = 30 * 1024
_TREE_BYTES_PER_TEXT = 4 * 1024
_TREE_BYTES_PER_ROW = 448


def _ordered(scale: CategoricalScale, what: str) -> tuple[list[Any], dict[Any, str]]:
    """Return a scale's order and palette, which a chart needs both of."""
    if scale.order is None or scale.palette is None:
        raise ValueError(f"The colours of the {what} need an order and a palette")
    missing = [value for value in scale.order if value not in scale.palette]
    if missing:
        raise ValueError(
            f"The colours of the {what} give no colour to {quoted_value(missing[0])}"
        )
    return list(scale.order), dict(scale.palette)


def legend_of(scale: CategoricalScale, entries: Sequence[Any], title: str) -> Drawn:
    """Return the categorical legend of a scale's first entries, as :func:`side_legend` lists them.

    The legend is titled as :func:`side_legend_title` titles it, and its scale
    holds the entries that the side legend lists, so a figure that draws a
    side legend itself, such as one beside a scatter, reports what it shows.
    """
    shown = list(entries[:LEGEND_SIDE_MAX_ENTRIES])
    palette = scale.palette or {}
    return Drawn(
        LegendSpec(kind="categorical", label=side_legend_title(title, len(entries))),
        CategoricalScale(
            order=tuple(shown),
            palette={value: palette[value] for value in shown},
            labels=(
                None
                if scale.labels is None
                else {value: category_label(scale, value) for value in shown}
            ),
            missing_color=scale.missing_color,
            missing_label=scale.missing_label,
            palette_name=scale.palette_name,
        ),
    )


def _name_items(
    ax: Any, positions: np.ndarray, names: list[str], *, horizontal: bool
) -> None:
    """Name the items on their axis, or none of them past ``MAX_NAMED_ITEMS``."""
    if horizontal:
        if len(names) <= MAX_NAMED_ITEMS:
            ax.set_yticks(positions, names)
        else:
            ax.set_yticks([])
    elif len(names) <= MAX_NAMED_ITEMS:
        ax.set_xticks(positions, names, rotation=90)
    else:
        ax.set_xticks([])


def draw_bars(
    ax: Any,
    table: pd.DataFrame,
    *,
    item: str,
    value: str,
    colors: CategoricalScale | str,
    segment: str | None = None,
    horizontal: bool = True,
    normalize: bool = False,
    side_by_side: bool = False,
    numeric_items: bool = False,
    value_label: str,
    title: str | None = None,
    legend_title: str | None = None,
    show_legend: bool = True,
) -> Drawn | None:
    """Draw one bar per item, in the order of the table, the first at the top or left.

    ``table`` holds one row per item, or with ``segment`` one row per item
    and segment, and a segment that an item lacks is 0. With ``segment``,
    ``colors`` gives each segment's colour and its order the order of each
    bar's parts: the first at the left of a horizontal bar and at the top of
    a vertical one, as the legend lists them. ``side_by_side`` draws the
    parts beside each other in the item's slot instead, the first at the
    left of a vertical slot and at the top of a horizontal one, for parts
    that are no part of one whole. Without ``segment``, ``colors`` is one
    colour, or a scale of each item's colour whose legend lists the items.
    ``normalize`` draws each item's parts as fractions of its total, 0
    without one, and a value that is not a number draws no bar.
    ``numeric_items`` stands each bar at its item's value on a numeric axis,
    which keeps its own ticks, such as a junction length, rather than in a
    named slot.

    Returns:
        The legend of the segments, or of the items' colours, drawn beside
        the panel when ``show_legend`` is true, and its scale; or None
        without ``legend_title``, or for bars of one colour, which draw no
        legend.

    Raises:
        ValueError: If the table is empty, an item or segment has no colour,
            an item takes one segment twice, ``side_by_side`` comes without
            ``segment``, or ``numeric_items`` with an item that is not a
            number.
    """
    _, mpl = require_matplotlib()
    if table.empty:
        raise ValueError("A bar chart needs a row to draw")
    if side_by_side and segment is None:
        raise ValueError("Bars side by side take a segment column")
    items = list(pd.unique(table[item].to_numpy(dtype=object)))
    rows = pd.Index(items).get_indexer(table[item].to_numpy(dtype=object))
    scaled = isinstance(colors, CategoricalScale)
    order: list[Any] = []
    palette: dict[Any, str] = {}
    if isinstance(colors, CategoricalScale):
        order, palette = _ordered(colors, "bars")
    elif segment is not None:
        raise TypeError("Stacked bars take a CategoricalScale of their segments")
    if segment is None:
        segments: list[Any] = [None]
        parts = np.zeros(len(table), dtype=np.int64)
        unknown = [name for name in items if scaled and name not in palette]
        if unknown:
            raise ValueError(
                f"The colours of the bars give no colour to {quoted_value(unknown[0])}"
            )
        fills: list[Any] = [[palette[name] for name in items] if scaled else colors]
        drawn = set(items)
        entries = [name for name in order if name in drawn]
    else:
        segments = order
        parts = pd.Index(segments).get_indexer(table[segment].to_numpy(dtype=object))
        if (parts < 0).any():
            unlisted = table[segment].to_numpy(dtype=object)[parts < 0][0]
            raise ValueError(
                f"The colours of the bars give no colour to {quoted_value(unlisted)}"
            )
        fills = [palette[name] for name in segments]
        entries = segments
    cells = rows * len(segments) + parts
    if has_duplicates(cells):
        raise ValueError(f"A bar chart takes each {item} once in each of its segments")
    values = np.zeros((len(items), len(segments)), dtype=np.float64)
    values[rows, parts] = table[value].to_numpy(dtype=np.float64)
    if normalize:
        totals = np.nansum(values, axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    if numeric_items:
        try:
            positions = np.asarray(items, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(
                f"Bars at numeric positions take numbers as their {item}"
            ) from None
    else:
        positions = np.arange(len(items))[::-1] if horizontal else np.arange(len(items))
    extent = _BAR_EXTENT / len(segments) if side_by_side else _BAR_EXTENT
    start = np.zeros(len(items), dtype=np.float64)
    # A vertical stack starts from its last segment, so the first ends on top.
    stacking = horizontal or side_by_side
    for at in range(len(segments)) if stacking else reversed(range(len(segments))):
        lengths = values[:, at]
        where, base = positions, start
        if side_by_side:
            # The first part stands at the left of a vertical slot, or at the
            # top of a horizontal one.
            shift = (at - (len(segments) - 1) / 2) * extent
            where = positions - shift if horizontal else positions + shift
            base = np.zeros(len(items), dtype=np.float64)
        if horizontal:
            ax.barh(where, lengths, left=base, color=fills[at], height=extent)
        else:
            ax.bar(where, lengths, bottom=base, color=fills[at], width=extent)
        start = start + np.nan_to_num(lengths)
    if not numeric_items:
        _name_items(ax, positions, [str(name) for name in items], horizontal=horizontal)
    if normalize:
        # Fractions span the whole axis, whichever parts are empty.
        (ax.set_xlim if horizontal else ax.set_ylim)(0.0, 1.0)
    (ax.set_xlabel if horizontal else ax.set_ylabel)(value_label)
    if title is not None:
        ax.set_title(title)
    if legend_title is None or not isinstance(colors, CategoricalScale):
        return None
    if show_legend:
        side_legend(
            ax,
            [
                mpl.patches.Patch(facecolor=palette[name], edgecolor="none")
                for name in entries
            ],
            [category_label(colors, name) for name in entries],
            title=legend_title,
        )
    return legend_of(colors, entries, legend_title)


@functools.cache
def _item_names_type() -> type:
    """A locator of every nth row or column of a heatmap, as many as its axis holds."""
    from matplotlib.ticker import Locator

    class ItemNames(Locator):
        """Tick every nth of ``n_items``, as many as the axis holds ``pitch`` points apart.

        The axis's length is read whenever the figure is drawn, after its
        layout, so names keep apart however a page shares its width; an axis
        names every one of its items where it can, and never more than
        ``MAX_NAMED_ITEMS``.
        """

        def __init__(self, n_items: int, pitch: float) -> None:
            self.n_items = n_items
            self.pitch = pitch

        def __call__(self) -> Any:
            return self.tick_values(0.0, float(self.n_items - 1))

        def tick_values(self, vmin: float, vmax: float) -> Any:
            axes = self.axis.axes
            pixels = axes.bbox.width if self.axis.axis_name == "x" else axes.bbox.height
            room = int(pixels * 72.0 / axes.figure.dpi // self.pitch)
            step = -(-self.n_items // max(1, min(MAX_NAMED_ITEMS, room)))
            return np.arange(0, self.n_items, max(1, step), dtype=np.float64)

    return ItemNames


def _item_name(names: list[str]) -> Any:
    """Name the item at a tick, or nothing past the items."""

    def name(value: float, position: int | None = None) -> str:
        at = int(round(value))
        return names[at] if 0 <= at < len(names) else ""

    return name


@functools.cache
def _count_formatter_type() -> type:
    """A log formatter that labels counts as whole numbers, never below 1."""
    from matplotlib.ticker import LogFormatter

    class CountLogFormatter(LogFormatter):
        """Label the ticks that ``LogFormatter`` labels, as whole numbers."""

        def __call__(self, x: float, pos: int | None = None) -> str:
            label = super().__call__(x, pos)
            return f"{x:,.0f}" if label and x >= 1 else ""

    return CountLogFormatter


def _log_counts(axis: Any) -> None:
    """Put an axis of counts on a log scale labelled with whole numbers."""
    formatter = _count_formatter_type()
    axis.set_major_formatter(formatter())
    axis.set_minor_formatter(formatter())


@functools.cache
def _value_formatter_type() -> type:
    """A log formatter that labels values as plain numbers, fractions included."""
    from matplotlib.ticker import LogFormatter

    class ValueLogFormatter(LogFormatter):
        """Label the ticks that ``LogFormatter`` labels, as plain numbers."""

        def __call__(self, x: float, pos: int | None = None) -> str:
            if not super().__call__(x, pos):
                return ""
            # Ten significant digits written out, never in powers of ten,
            # with thousands apart.
            text = np.format_float_positional(
                x, precision=10, unique=False, fractional=False, trim="-"
            )
            whole, _, fraction = text.partition(".")
            return f"{int(whole):,}" + (f".{fraction}" if fraction else "")

    return ValueLogFormatter


def log_values(axis: Any) -> None:
    """Label a log axis of values that need not be whole, such as mean depths, as plain numbers.

    The ticks that Matplotlib labels read as numbers such as 0.000001, 0.5,
    60 and 1,000 rather than powers of ten.
    """
    formatter = _value_formatter_type()
    axis.set_major_formatter(formatter())
    axis.set_minor_formatter(formatter())


def draw_lines(
    ax: Any,
    table: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str,
    colors: CategoricalScale,
    positions: Sequence[str] | None = None,
    markers: bool = False,
    dashed: str | None = None,
    marked: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    x_label: str | None = None,
    y_label: str,
    title: str | None = None,
    legend_title: str | None = None,
    show_legend: bool = True,
) -> Drawn | None:
    """Draw one line per series in the order of ``colors``, in its colour.

    Each series' rows run in the order of the table. With ``positions``, the
    ``x`` values name positions on the axis, in that order, and a series
    has a gap at a position it lacks. ``markers`` draws a point at every
    value; ``dashed``, a column of flags, draws the flagged rows dashed,
    from the last row before them, such as the extrapolated part of a
    curve; and ``marked``, another, draws a point at each flagged row, such
    as a curve's observed size. A log axis of counts is labelled with whole
    numbers.

    Each series draws, in order, its solid line, then its dashed line with
    ``dashed`` and its points with ``marked``.

    Returns:
        The legend of the series, drawn beside the panel when
        ``show_legend`` is true, and their scale; or None without
        ``legend_title``, which draws no legend.

    Raises:
        ValueError: If a series has no colour or takes a position twice,
            or ``positions`` lacks an ``x`` value or comes with ``dashed``
            or ``marked``.
    """
    names, palette = _ordered(colors, "lines")
    values = table[series].to_numpy(dtype=object)
    unknown = [name for name in pd.unique(values) if name not in palette]
    if unknown:
        raise ValueError(
            f"The colours of the lines give no colour to {quoted_value(unknown[0])}"
        )
    if positions is not None and (dashed is not None or marked is not None):
        raise ValueError("Lines along named positions take neither dashed nor marked")
    by_series = {
        name: rows for name, rows in table.groupby(series, sort=False, observed=True)
    }
    empty = table.iloc[:0]
    lines = []
    for name in names:
        rows = by_series.get(name, empty)
        xs = rows[x].to_numpy()
        ys = rows[y].to_numpy(dtype=np.float64)
        if positions is not None:
            at = pd.Index(list(positions)).get_indexer(xs)
            if (at < 0).any():
                raise ValueError(
                    f"The positions do not name {quoted_value(xs[at < 0][0])}"
                )
            if has_duplicates(at):
                raise ValueError(
                    f"The line of {quoted_value(name)} takes a position twice"
                )
            aligned = np.full(len(positions), np.nan)
            aligned[at] = ys
            xs, ys = np.arange(len(positions)), aligned
        solid = (
            np.ones(len(ys), dtype=bool)
            if dashed is None
            else ~rows[dashed].to_numpy(dtype=bool)
        )
        (line,) = ax.plot(
            xs[solid],
            ys[solid],
            color=palette[name],
            linewidth=_LINE_WIDTH,
            marker="o" if markers else None,
            markersize=_POINT_SIZE,
            label=category_label(colors, name),
        )
        lines.append(line)
        if dashed is not None:
            # The dashed part starts at the last solid row before it.
            flagged = np.flatnonzero(~solid)
            joined = flagged
            if flagged.size and flagged[0] > 0 and solid[flagged[0] - 1]:
                joined = np.concatenate(([flagged[0] - 1], flagged))
            elif not flagged.size and solid.any():
                joined = np.flatnonzero(solid)[-1:]
            ax.plot(
                xs[joined],
                ys[joined],
                color=palette[name],
                linewidth=_LINE_WIDTH,
                linestyle="--",
            )
        if marked is not None:
            points = rows[marked].to_numpy(dtype=bool)
            ax.plot(
                xs[points],
                ys[points],
                "o",
                color=palette[name],
                markersize=_MARKED_POINT_SIZE,
            )
    if log_x:
        ax.set_xscale("log")
        _log_counts(ax.xaxis)
    if log_y:
        ax.set_yscale("log")
        _log_counts(ax.yaxis)
    if positions is not None:
        ticks = np.arange(len(positions))
        if len(positions) <= MAX_NAMED_ITEMS:
            ax.set_xticks(
                ticks, [str(name) for name in positions], rotation=45, ha="right"
            )
        else:
            ax.set_xticks([])
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if legend_title is None:
        return None
    if show_legend:
        side_legend(
            ax,
            lines,
            [category_label(colors, name) for name in names],
            title=legend_title,
        )
    return legend_of(colors, names, legend_title)


def _bin_edges(low: float, high: float, bins: int, *, log_x: bool) -> np.ndarray:
    """The edges of ``bins`` bins from ``low`` to ``high``, even on a linear or log axis."""
    return (
        np.geomspace(low, high, bins + 1) if log_x else np.linspace(low, high, bins + 1)
    )


def draw_histograms(
    ax: Any,
    table: pd.DataFrame,
    *,
    value: str,
    series: str,
    colors: CategoricalScale,
    bins: int,
    log_x: bool = False,
    x_label: str,
    y_label: str,
    title: str | None = None,
    legend_title: str | None = None,
    show_legend: bool = True,
) -> Drawn | None:
    """Draw one step histogram per series in the order of ``colors``, in its colour.

    The series share ``bins`` bins, which span the values of every series
    evenly, or with ``log_x`` evenly on a log axis labelled with plain
    numbers, which leaves out values at or below 0; values that are all
    equal, or too close to split into bins, take bins around them. A value
    that is not finite is left out, and a series without a value draws a
    flat line.

    Returns:
        The legend of the series, drawn beside the panel when
        ``show_legend`` is true, and their scale; or None without
        ``legend_title``, which draws no legend.

    Raises:
        TypeError: If ``bins`` is not an integer.
        ValueError: If ``bins`` is below 1, a series has no colour, or no
            value can be drawn.
    """
    _, mpl = require_matplotlib()
    bins = integer_argument(bins, "bins", minimum=1)
    names, palette = _ordered(colors, "histograms")
    labels = table[series].to_numpy(dtype=object)
    unknown = [name for name in pd.unique(labels) if name not in palette]
    if unknown:
        raise ValueError(
            f"The colours of the histograms give no colour to {quoted_value(unknown[0])}"
        )
    values = table[value].to_numpy(dtype=np.float64)
    placed = np.isfinite(values) & ((values > 0) if log_x else True)
    if not placed.any():
        raise ValueError(
            "A histogram on a log axis needs a value above 0"
            if log_x
            else "A histogram needs a finite value"
        )
    low, high = float(values[placed].min()), float(values[placed].max())
    edges = _bin_edges(low, high, bins, log_x=log_x)
    if not np.all(np.diff(edges) > 0):
        # Values too close to tell apart take bins around them.
        low, high = (low / 2, low * 2) if log_x else (low - 0.5, low + 0.5)
        edges = _bin_edges(low, high, bins, log_x=log_x)
    handles = []
    for name in names:
        ax.hist(
            values[placed & (labels == name)],
            bins=edges,
            histtype="step",
            color=palette[name],
            linewidth=_LINE_WIDTH,
        )
        handles.append(
            mpl.lines.Line2D([], [], color=palette[name], linewidth=_LINE_WIDTH)
        )
    if log_x:
        ax.set_xscale("log")
        log_values(ax.xaxis)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if legend_title is None:
        return None
    if show_legend:
        side_legend(
            ax,
            handles,
            [category_label(colors, name) for name in names],
            title=legend_title,
        )
    return legend_of(colors, names, legend_title)


def pair_matrix(
    pairs: pd.DataFrame,
    names: Sequence[str],
    value: str,
    *,
    diagonal: float | np.ndarray = np.nan,
) -> pd.DataFrame:
    """Return the square table of a value of pairs of groups, the same either way round.

    ``pairs`` holds one row per pair, its groups in ``group_1`` and
    ``group_2``; a pair it lacks is NaN, and ``diagonal`` gives each group's
    own value.

    Raises:
        ValueError: If a pair names a group that ``names`` does not list.
    """
    index = pd.Index(list(names))
    first = index.get_indexer(pairs["group_1"].to_numpy(dtype=object))
    second = index.get_indexer(pairs["group_2"].to_numpy(dtype=object))
    if (first < 0).any() or (second < 0).any():
        unlisted = pairs[["group_1", "group_2"]].to_numpy(dtype=object)[
            (first < 0) | (second < 0)
        ][0]
        raise ValueError(
            f"A pair names a group the names lack: {quoted_value(list(unlisted))}"
        )
    values = np.full((len(index), len(index)), np.nan, dtype=np.float64)
    measured = pairs[value].to_numpy(dtype=np.float64)
    values[first, second] = measured
    values[second, first] = measured
    values[np.diag_indices(len(index))] = diagonal
    return pd.DataFrame(values, index=index, columns=index)


def draw_matrix(
    ax: Any,
    matrix: pd.DataFrame,
    *,
    color_scale: ColorScale,
    value_label: str,
    blank_diagonal: bool = False,
    value_format: str = ".3g",
    show_legend: bool = True,
    colorbar_beside: Sequence[Any] | None = None,
) -> Drawn:
    """Draw a table as a heatmap through a colour scale, with a colour bar.

    A cell without a value the scale can colour, such as NaN, an infinite
    value or one at or below 0 on a log scale, takes the scale's
    ``missing_color``, and with ``blank_diagonal`` the diagonal is left
    blank and out of the colour limits, so the colours span the other cells.
    A table of at most ``MAX_PRINTED_VALUES`` rows and columns prints each
    coloured value, dark on light colours and light on dark ones. Each axis
    names as many of its rows or columns as its length holds once the
    figure is laid out, at most ``MAX_NAMED_ITEMS``: every one, or else
    every second, third or nth.
    ``colorbar_beside`` names the panels whose heatmaps share this one's
    scale and limits, beside all of which the colour bar stands; by
    default it stands beside this panel alone.

    Returns:
        The colour bar's legend, with the limits drawn, and the scale; the
        colour bar is drawn beside the panel when ``show_legend`` is true.
    """
    _, mpl = require_matplotlib()
    values = matrix.to_numpy(dtype=np.float64)
    n_rows, n_columns = values.shape
    blank = np.zeros(values.shape, dtype=bool)
    if blank_diagonal:
        blank[np.diag_indices(min(n_rows, n_columns))] = True
    colored = scalable_values(values, color_scale) & ~blank
    vmin, vmax = resolve_color_limits(values[colored], color_scale)
    norm = continuous_norm(
        mpl, vmin=vmin, vmax=vmax, vcenter=color_scale.vcenter, scale=color_scale.scale
    )
    colormap = mpl.colormaps[color_scale.cmap or "viridis"].with_extremes(
        bad=(0.0, 0.0, 0.0, 0.0)
    )
    image = ax.imshow(
        np.ma.masked_array(values, mask=~colored), cmap=colormap, norm=norm
    )
    missing = ~colored & ~blank
    if missing.any():
        # Cells without a value take the missing colour, in a layer over the
        # blank cells of the image.
        layer = np.zeros((n_rows, n_columns, 4), dtype=np.float64)
        layer[missing] = mpl.colors.to_rgba(color_scale.missing_color)
        ax.imshow(layer)
    # Each axis names as many of its items as its length holds once the
    # figure is laid out, every one or else every second, third or nth.
    for axis, names in ((ax.xaxis, matrix.columns), (ax.yaxis, matrix.index)):
        size = mpl.font_manager.FontProperties(
            size=mpl.rcParams[f"{axis.axis_name}tick.labelsize"]
        ).get_size_in_points()
        axis.set_major_locator(_item_names_type()(len(names), _NAME_PITCH * size))
        axis.set_major_formatter(
            mpl.ticker.FuncFormatter(_item_name([str(name) for name in names]))
        )
    ax.tick_params(axis="x", labelrotation=90)
    if n_rows <= MAX_PRINTED_VALUES and n_columns <= MAX_PRINTED_VALUES:
        for row, column in zip(*np.nonzero(colored), strict=True):
            value = values[row, column]
            red, green, blue, _alpha = colormap(norm(value))
            light = 0.299 * red + 0.587 * green + 0.114 * blue > 0.5
            ax.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                color="black" if light else "white",
                fontsize="x-small",
            )
    if show_legend:
        colorbar = ax.figure.colorbar(
            image,
            ax=ax if colorbar_beside is None else list(colorbar_beside),
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(value_label)
    return Drawn(
        LegendSpec(
            kind="colorbar",
            label=value_label,
            extras={"vmin": float(vmin), "vmax": float(vmax)},
        ),
        color_scale,
    )


def capped_counts(values: Any, cap: int) -> pd.Series:
    """Count whole numbers by value, those of ``cap`` or more in one last bin.

    ``values`` are counts of at least 0, such as each cell's chains, and the
    bins are each value below ``cap`` and the values of ``cap`` or more.

    Returns:
        The count of each bin, indexed by its name: ``"0"`` to the text of
        ``cap - 1``, then ``f"{cap}+"``.

    Raises:
        TypeError: If ``cap`` is not an integer, or ``values`` are not whole
            numbers.
        ValueError: If ``cap`` is below 1, or a value is below 0.
    """
    cap = integer_argument(cap, "cap", minimum=1)
    numbers = np.asarray(values).ravel()
    if numbers.size and not np.issubdtype(numbers.dtype, np.integer):
        raise TypeError("Capped counts take whole numbers")
    numbers = numbers.astype(np.int64)
    if numbers.size and int(numbers.min()) < 0:
        raise ValueError("Capped counts take counts of at least 0")
    counts = np.bincount(np.minimum(numbers, cap), minlength=cap + 1)
    return pd.Series(
        counts.astype(np.int64),
        index=pd.Index([*(str(count) for count in range(cap)), f"{cap}+"]),
    )


def tree_drawing_bytes(n_rows: int, *, n_entries: int, n_texts: int) -> int:
    """What :func:`draw_tree` holds to draw ``n_rows`` nodes, with the figure it draws in.

    ``n_entries`` counts the categories of the wedges, of which the legend
    lists at most ``LEGEND_SIDE_MAX_ENTRIES``, and ``n_texts`` the wedges that
    say something.
    """
    return (
        _TREE_BYTES_PER_ENTRY * min(int(n_entries), LEGEND_SIDE_MAX_ENTRIES)
        + _TREE_BYTES_PER_TEXT * int(n_texts)
        + _TREE_BYTES_PER_ROW * int(n_rows)
    )


def draw_tree(
    ax: Any,
    nodes: pd.DataFrame,
    *,
    categories: np.ndarray,
    colors: CategoricalScale,
    line_color: str,
    x_label: str,
    texts: np.ndarray | None = None,
    cut: float | None = None,
    title: str | None = None,
    legend_title: str | None = None,
    show_legend: bool = True,
) -> Drawn | None:
    """Draw a collapsed tree: a branch to each node, a wedge for each collapsed clade and the cut as a line.

    ``nodes`` holds one row per node drawn, each after the node it hangs
    from, as :func:`~cytearc.clones.phylogeny.collapse_phylogeny` places
    them: ``parent`` (the row it hangs from, -1 for none), ``wedge``
    (whether it is drawn collapsed), ``x`` (where it stands on the x axis),
    ``start`` (where its branch starts), ``end`` (its latest leaf), and
    ``y``, ``low`` and ``high``, its middle and extent down the y axis, which
    runs from 0 at the top. A node opened joins its children with a line
    at its ``x``. A wedge runs from its node to a base at ``end`` as tall as
    nine tenths of its extent, in the colour that the palette of ``colors``
    gives its category in ``categories``; a wedge that ends where it starts,
    such as one leaf, is a point. ``texts`` holds what each wedge says at its
    base, empty for nothing, and ``cut`` draws a dashed line at that ``x``.
    The categories and texts are arrays aligned with the rows, so a figure
    of many rows adds no columns to its table to draw them.

    The legend lists the wedges' categories that the scale's order names, in
    that order and at most ``LEGEND_SIDE_MAX_ENTRIES``; a figure leaves
    categories out of the order to keep them out of the legend, such as
    clones too thin to see, and the legend's title says how many of the
    categories drawn it lists when it leaves some out.

    Returns:
        The legend of the categories listed, drawn beside the panel when
        ``show_legend`` is true, and its scale; or None without
        ``legend_title``.

    Raises:
        ValueError: If the table is empty, the categories or texts are not
            one for each row, a wedge's category has no colour, or the
            scale's order, with a legend title, lists no category drawn.
    """
    _, mpl = require_matplotlib()
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if nodes.empty:
        raise ValueError("A tree needs a node to draw")
    order, palette = _ordered(colors, "wedges")
    parent = nodes["parent"].to_numpy(dtype=np.int64)
    x, start, end, y, low, high = (
        nodes[column].to_numpy(dtype=np.float64)
        for column in ("x", "start", "end", "y", "low", "high")
    )
    wedge = nodes["wedge"].to_numpy(dtype=bool)
    kinds = np.asarray(categories)
    if kinds.shape != (len(nodes),) or (
        texts is not None and np.shape(texts) != (len(nodes),)
    ):
        raise ValueError("A tree takes one category and one text for each row")
    unknown = [
        value for value in pd.unique(kinds[wedge]).tolist() if value not in palette
    ]
    if unknown:
        raise ValueError(
            f"The colours of the wedges give no colour to {quoted_value(unknown[0])}"
        )
    # Each node's branch from where it starts, and each opened node's line
    # from its first child to its last, as one path of segments.
    has_parent = parent >= 0
    upper = np.full(len(nodes), np.inf)
    lower = np.full(len(nodes), -np.inf)
    np.minimum.at(upper, parent[has_parent], y[has_parent])
    np.maximum.at(lower, parent[has_parent], y[has_parent])
    branched = start != x
    joined = ~wedge & np.isfinite(upper)
    segments = np.concatenate(
        [
            np.stack([start, y, x, y], axis=1)[branched],
            np.stack([x, upper, x, lower], axis=1)[joined],
        ]
    ).reshape(-1, 2)
    # The paths are added as artists, since adding a patch walks each of its
    # segments to find the panel's limits, which the nodes give.
    if len(segments):
        steps = np.tile(
            np.asarray([Path.MOVETO, Path.LINETO], dtype=Path.code_type),
            len(segments) // 2,
        )
        ax.add_artist(
            PathPatch(
                Path(segments, steps),
                fill=False,
                edgecolor=line_color,
                linewidth=_TREE_LINE_WIDTH,
                capstyle="butt",
            )
        )
    # Each wedge a triangle from its node to its base, one path of a colour's
    # wedges for each colour.
    middle = (low + high) / 2
    spread = (high - low) * _WEDGE_EXTENT / 2
    wide = wedge & (end != x)
    outline = np.asarray(
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY], dtype=Path.code_type
    )
    # Each wedge's colour as a position among the colours, through its
    # category's position in the palette, which also colours the categories
    # that the legend leaves out.
    coloured = list(palette)
    shades = list(dict.fromkeys(palette[value] for value in coloured))
    shade_of = {shade: at for at, shade in enumerate(shades)}
    shading = np.asarray(
        [shade_of[palette[value]] for value in coloured], dtype=np.int64
    )[pd.Index(coloured).get_indexer(kinds)]
    shading[~wide] = -1
    for at, shade in enumerate(shades):
        drawn = shading == at
        if not drawn.any():
            continue
        corners = np.stack(
            [
                np.stack([x[drawn], middle[drawn]], axis=1),
                np.stack([end[drawn], middle[drawn] - spread[drawn]], axis=1),
                np.stack([end[drawn], middle[drawn] + spread[drawn]], axis=1),
                np.stack([x[drawn], middle[drawn]], axis=1),
            ],
            axis=1,
        ).reshape(-1, 2)
        ax.add_artist(
            PathPatch(
                Path(corners, np.tile(outline, int(drawn.sum()))),
                facecolor=shade,
                edgecolor="none",
            )
        )
    flat = wedge & ~wide
    if flat.any():
        ax.scatter(
            x[flat],
            middle[flat],
            c=[palette[value] for value in kinds[flat].tolist()],
            s=_TREE_POINT_AREA,
            linewidths=0,
            zorder=3,
        )
    if texts is not None:
        # A text stands inside its wedge's base, or beside a point.
        said_at = np.asarray(texts, dtype=object)
        for row in np.flatnonzero(wedge & (said_at != "")).tolist():
            said = str(said_at[row])
            if said:
                ax.annotate(
                    said,
                    (end[row], middle[row]),
                    xytext=(-2, 0) if wide[row] else (4, 0),
                    textcoords="offset points",
                    ha="right" if wide[row] else "left",
                    va="center",
                    fontsize="x-small",
                    color=line_color,
                )
    if cut is not None:
        ax.axvline(cut, color=line_color, linestyle="--", linewidth=1.0)
    ax.update_datalim(
        [
            (float(min(start.min(), x.min())), 0.0),
            (float(max(end.max(), x.max())), float(high.max())),
        ]
    )
    ax.autoscale_view()
    ax.set_ylim(float(high.max()) or 1.0, 0.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title)
    if legend_title is None:
        return None
    drawn = set(pd.unique(kinds[wedge]).tolist())
    entries = [value for value in order if value in drawn]
    if not entries:
        raise ValueError(
            "The colours of the wedges list no category that a wedge holds"
        )
    # Only the entries that the legend lists get handles, under a title that
    # counts every category drawn.
    shown = entries[:LEGEND_SIDE_MAX_ENTRIES]
    label = (
        legend_title
        if len(shown) == len(drawn)
        else f"{legend_title} ({len(shown)} of {len(drawn)})"
    )
    if show_legend:
        side_legend(
            ax,
            [
                mpl.patches.Patch(facecolor=palette[value], edgecolor="none")
                for value in shown
            ],
            [category_label(colors, value) for value in shown],
            title=label,
        )
    reported = legend_of(colors, entries, legend_title)
    return Drawn(replace(reported.legend, label=label), reported.scale)
