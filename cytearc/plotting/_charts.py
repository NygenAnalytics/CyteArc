"""Charts that the figures of clones, receptors, lineages and variants draw from tidy tables.

Each chart draws one panel. It names at most ``MAX_NAMED_ITEMS`` items on an
axis, draws a titled legend beside the panel when given a legend title, and
returns what that legend shows with the scale behind it, a :class:`Drawn`,
which ``finish_plot`` reports so that ``compose_results`` can draw a legend
that several figures share once.
"""

import functools
from collections.abc import Sequence
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
    "legend_of",
    "log_values",
    "pair_matrix",
]

# Bars fill this share of each item's slot, lines are this wide, and the
# points on them this large.
_BAR_EXTENT = 0.8
_LINE_WIDTH = 1.4
_POINT_SIZE = 3.5
_MARKED_POINT_SIZE = 4.0


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
            label = super().__call__(x, pos)
            return f"{x:,.10g}" if label else ""

    return ValueLogFormatter


def log_values(axis: Any) -> None:
    """Label a log axis of values that need not be whole, such as mean depths, as plain numbers.

    The ticks that Matplotlib labels read as numbers such as 0.5, 60 and
    1,000 rather than powers of ten.
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
    equal take bins around them. A value that is not finite is left out, and
    a series without a value draws a flat line.

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
    if high == low:
        low, high = (low / 2, low * 2) if log_x else (low - 0.5, low + 0.5)
    edges = (
        np.geomspace(low, high, bins + 1) if log_x else np.linspace(low, high, bins + 1)
    )
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
    coloured value, dark on light colours and light on dark ones, and one
    of at most ``MAX_NAMED_ITEMS`` names its rows and columns.
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
    if n_rows <= MAX_NAMED_ITEMS and n_columns <= MAX_NAMED_ITEMS:
        ax.set_xticks(
            np.arange(n_columns), [str(name) for name in matrix.columns], rotation=90
        )
        ax.set_yticks(np.arange(n_rows), [str(name) for name in matrix.index])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
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
