"""Themes and categorical palettes for cytearc.plotting."""

import contextlib
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd

from ..utils.arguments import float_argument
from ..utils.palettes import palette_for_n
from ._display import labels_in_stored_order, merge_categorical_scales
from ._contracts import (
    CategoricalScale,
    ColorScale,
    DensityOverlay,
    FrameStyle,
    LegendLoc,
    SizeScale,
)

# Shared CyteArc figure defaults used by embedding-like plots.
DEFAULT_RASTERIZE_THRESHOLD = 50_000
DEFAULT_PANEL_INCHES = 3.2
MAX_FIGURE_WIDTH_INCHES = 7.5
LEGEND_SIDE_MAX_CATEGORIES = 12
LEGEND_ON_DATA_MAX_CATEGORIES = 40
LEGEND_SIDE_ENTRIES_PER_COLUMN = 20
LEGEND_SIDE_MAX_COLUMNS = 4
LEGEND_SIDE_MAX_ENTRIES = LEGEND_SIDE_ENTRIES_PER_COLUMN * LEGEND_SIDE_MAX_COLUMNS
# The charts of clone figures name at most this many items on an axis, and
# print a heatmap's values when it has at most this many rows and columns;
# their tables hold every item.
MAX_NAMED_ITEMS = 40
MAX_PRINTED_VALUES = 12
# The colors of immune receptor types, in the order of
# cytearc.repertoire.chains.RECEPTOR_TYPES, for every figure whose bars or
# groups are receptor types.
RECEPTOR_TYPE_SCALE = CategoricalScale(
    order=("TCR", "BCR", "ambiguous", "multichain", "no IR"),
    palette={
        "TCR": "#2b6cb0",
        "BCR": "#dd6b20",
        "ambiguous": "#805ad5",
        "multichain": "#c53030",
        "no IR": "#bdbdbd",
    },
)
# A vocabulary's value for no value, such as a cell without a receptor chain
# or a recorder site without a read, takes the missing color of every scale,
# the receptor types' gray, and a value that carries nothing, such as an
# unedited site, one muted color.
NO_VALUE_COLOR = "#bdbdbd"
MUTED_COLOR = "#e2e2e2"
# A categorical scale draws cells without a category in this color and
# names them with this label unless it, or the column's stored display, sets
# others.
MISSING_CATEGORY_COLOR = NO_VALUE_COLOR
MISSING_CATEGORY_LABEL = "NA"
# The colors of the chain pairings of cells, in the order of
# cytearc.repertoire.chains.CHAIN_PAIRINGS, sharing the receptor types'
# colors of the pairings they name.
CHAIN_PAIRING_SCALE = CategoricalScale(
    order=(
        "single pair",
        "orphan VJ",
        "orphan VDJ",
        "extra VJ",
        "extra VDJ",
        "two full chains",
        "ambiguous",
        "multichain",
        "no IR",
    ),
    palette={
        "single pair": "#2f855a",
        "orphan VJ": "#9ae6b4",
        "orphan VDJ": "#68d391",
        "extra VJ": "#4299e1",
        "extra VDJ": "#2b6cb0",
        "two full chains": "#1a365d",
        "ambiguous": "#805ad5",
        "multichain": "#c53030",
        "no IR": NO_VALUE_COLOR,
    },
)
# The colors of the two arms of a receptor, VJ chains (TRA, TRG, IGK, IGL)
# and VDJ chains (TRB, TRD, IGH), which no receptor type or chain pairing
# takes, so that a color of receptor_qc means one thing.
CHAIN_ARM_SCALE = CategoricalScale(
    order=("VJ", "VDJ"), palette={"VJ": "#cc79a7", "VDJ": "#b7791f"}
)
# The colors of the loci of receptor chains, in the order of
# cytearc.repertoire.chains.LOCI: T cell loci in blues and teals, B cell
# loci in oranges.
LOCUS_SCALE = CategoricalScale(
    order=("TRA", "TRG", "IGK", "IGL", "TRB", "TRD", "IGH"),
    palette={
        "TRA": "#4299e1",
        "TRG": "#38b2ac",
        "IGK": "#d69e2e",
        "IGL": "#975a16",
        "TRB": "#2b6cb0",
        "TRD": "#2c7a7b",
        "IGH": "#dd6b20",
    },
)
# The colors and words of the states of a recorder site that figures draw,
# of cytearc.clones.recorders.STATES, in the order of their stacks: a site
# that does not apply to a cell is drawn in no state.
RECORDER_STATE_SCALE = CategoricalScale(
    order=("mark", "unknown_mark", "unedited", "missing"),
    palette={
        "mark": "#0072b2",
        "unknown_mark": "#e69f00",
        "unedited": MUTED_COLOR,
        "missing": NO_VALUE_COLOR,
    },
    labels={
        "mark": "Mark",
        "unknown_mark": "Unknown mark",
        "unedited": "Unedited",
        "missing": "Missing",
    },
)
# The colors and words of the genotype calls of cytearc.variants.CALLS, in
# its order: a cell without evidence at a site takes no value's gray, and
# evidence that a rule did not call the muted gray.
GENOTYPE_CALL_SCALE = CategoricalScale(
    order=(
        "wild_type",
        "heterozygous",
        "homozygous_alternative",
        "alternative_present",
        "missing",
        "filtered",
    ),
    palette={
        "wild_type": "#4c78a8",
        "heterozygous": "#f58518",
        "homozygous_alternative": "#e45756",
        "alternative_present": "#b279a2",
        "missing": NO_VALUE_COLOR,
        "filtered": MUTED_COLOR,
    },
    labels={
        "wild_type": "Wild type",
        "heterozygous": "Heterozygous",
        "homozygous_alternative": "Homozygous alternative",
        "alternative_present": "Alternative present",
        "missing": "Missing",
        "filtered": "Filtered",
    },
)


@dataclass(frozen=True, slots=True)
class _PointSizeSpec:
    """How to resize one scatter artist once its panel size is final."""

    n_points: int
    size_min: float
    size_max: float
    multiplier: float
    edgecolor: str | None
    edgewidth: float | None


_LAYOUT_POINT_SIZE_SPECS: WeakKeyDictionary[Any, _PointSizeSpec] = WeakKeyDictionary()


THEMES: dict[str, dict[str, Any]] = {
    "paper": {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    },
    "notebook": {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 150,
        "figure.dpi": 100,
    },
    "minimal": {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    },
    "dark": {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": "#e8e8e8",
        "axes.labelcolor": "#e8e8e8",
        "axes.edgecolor": "#e8e8e8",
        "xtick.color": "#e8e8e8",
        "ytick.color": "#e8e8e8",
        "axes.titlecolor": "#e8e8e8",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 150,
        "figure.dpi": 100,
    },
}


def default_point_size(
    n_cells: int,
    *,
    panel_area: float = DEFAULT_PANEL_INCHES**2,
    size_min: float = 1.0,
    size_max: float = 28.0,
) -> float:
    """Marker area derived from selected cells and physical panel area."""
    if panel_area <= 0:
        raise ValueError("panel_area must be positive")
    if size_min <= 0 or size_max < size_min:
        raise ValueError("point-size bounds must satisfy 0 < size_min <= size_max")
    n = max(1, int(n_cells))
    reference_area = DEFAULT_PANEL_INCHES**2
    area_factor = (float(panel_area) / reference_area) ** 0.72
    population_factor = (1_000.0 / n) ** 0.5
    return float(min(size_max, max(size_min, 16.0 * area_factor * population_factor)))


def default_point_edgewidth(
    n_cells: int,
    *,
    point_size: float | None = None,
) -> float:
    """Tune point outlines to marker area and cloud density."""
    n = max(1, int(n_cells))
    area = point_size if point_size is not None else default_point_size(n)
    if n >= 20_000 or area < 2.5:
        return 0.0
    if area < 7.0 or n >= 10_000:
        return 0.05
    return 0.15


def scatter_edges(edgecolor: str, edgewidth: float) -> tuple[str, float]:
    """Return scatter edge color and width, hiding edges of zero width."""
    if edgewidth <= 0:
        return "none", 0.0
    return edgecolor, float(edgewidth)


def panel_area_inches(ax: Any) -> float:
    """Physical axes area in square inches, floored for collapsed panels."""
    bounds = ax.get_position()
    width = max(float(bounds.width * ax.figure.get_figwidth()), 0.1)
    height = max(float(bounds.height * ax.figure.get_figheight()), 0.1)
    return width * height


def register_layout_point_size(
    collection: Any,
    *,
    n_points: int,
    size_min: float,
    size_max: float,
    multiplier: float = 1.0,
    edgecolor: str | None = None,
    edgewidth: float | None = None,
) -> None:
    """Mark a scatter artist whose marker area follows its final panel size.

    ``multiplier`` scales the panel point size, for example for highlighted
    cells. With ``edgecolor``, marker edges are refreshed with the size, using
    ``edgewidth`` or a width derived from the point size and population.
    """
    _LAYOUT_POINT_SIZE_SPECS[collection] = _PointSizeSpec(
        n_points=int(n_points),
        size_min=float(size_min),
        size_max=float(size_max),
        multiplier=float(multiplier),
        edgecolor=edgecolor,
        edgewidth=None if edgewidth is None else float(edgewidth),
    )


def refresh_layout_point_sizes(figure: Any) -> None:
    """Refresh marked scatter artists after figure layout is resolved.

    The layout is resolved without rendering, so the cost does not grow with
    the number of points.
    """
    marked = [
        (ax, collection, specification)
        for ax in figure.axes
        for collection in ax.collections
        if (specification := _LAYOUT_POINT_SIZE_SPECS.get(collection)) is not None
    ]
    if not marked:
        return
    figure.draw_without_rendering()
    for ax, collection, specification in marked:
        point_size = default_point_size(
            specification.n_points,
            panel_area=panel_area_inches(ax),
            size_min=specification.size_min,
            size_max=specification.size_max,
        )
        collection.set_sizes(
            np.full(
                len(collection.get_offsets()),
                point_size * specification.multiplier,
                dtype=np.float64,
            )
        )
        if specification.edgecolor is not None:
            edges, linewidth = scatter_edges(
                specification.edgecolor,
                (
                    specification.edgewidth
                    if specification.edgewidth is not None
                    else default_point_edgewidth(
                        specification.n_points,
                        point_size=point_size,
                    )
                ),
            )
            collection.set_edgecolors(edges)
            collection.set_linewidths(linewidth)


def resolve_legend_loc(n_categories: int, legend_loc: LegendLoc = "auto") -> LegendLoc:
    """Choose a legend placement that survives many clusters."""
    if legend_loc != "auto":
        if legend_loc not in ("right", "on_data", "none"):
            raise ValueError(
                "legend_loc must be one of 'auto', 'right', 'on_data', 'none'"
            )
        return legend_loc
    if n_categories <= LEGEND_SIDE_MAX_CATEGORIES:
        return "right"
    if n_categories <= LEGEND_ON_DATA_MAX_CATEGORIES:
        return "on_data"
    return "right"


def legend_side_columns(
    n_entries: int,
    *,
    rows: int = LEGEND_SIDE_ENTRIES_PER_COLUMN,
) -> int:
    """Columns for a side legend, bounded so wide category sets stay readable.

    ``rows`` is the most entries one column holds.
    """
    columns = int(np.ceil(max(int(n_entries), 1) / max(int(rows), 1)))
    return max(1, min(columns, LEGEND_SIDE_MAX_COLUMNS))


# Side-legend height in multiples of the legend font size: each entry row,
# the title, and the legend's own padding. Matplotlib's default spacing gives
# rows of about 1.6 font sizes with the 5-point legend markers that CyteArc
# draws, so these bounds slightly overestimate the height.
_LEGEND_ROW_FONT_SIZES = 1.65
_LEGEND_TITLE_FONT_SIZES = 1.7
_LEGEND_PADDING_FONT_SIZES = 0.5
# Constrained layout keeps this margin, in inches, at the figure's edges.
_LAYOUT_EDGE_INCHES = 2 * 3.0 / 72.0


def _legend_font_points() -> float:
    from matplotlib.font_manager import FontProperties

    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    return float(
        FontProperties(size=mpl.rcParams["legend.fontsize"]).get_size_in_points()
    )


def legend_side_height(rows: int, *, title: bool) -> float:
    """Height in inches of a side legend with ``rows`` entry rows."""
    fontsize = _legend_font_points()
    font_sizes = (
        _LEGEND_PADDING_FONT_SIZES
        + (_LEGEND_TITLE_FONT_SIZES if title else 0.0)
        + max(int(rows), 1) * _LEGEND_ROW_FONT_SIZES
    )
    return fontsize * font_sizes / 72.0 + _LAYOUT_EDGE_INCHES


def legend_side_rows(height: float, *, title: bool) -> int:
    """Entry rows of a side legend that fit in ``height`` inches.

    The result is at least one row and at most
    ``LEGEND_SIDE_ENTRIES_PER_COLUMN``, in the active theme's legend font.
    """
    fontsize = _legend_font_points()
    fixed = _LEGEND_PADDING_FONT_SIZES + (_LEGEND_TITLE_FONT_SIZES if title else 0.0)
    usable = (float(height) - _LAYOUT_EDGE_INCHES) * 72.0 / fontsize - fixed
    # The tolerance keeps a height from legend_side_height at its row count.
    rows = int(np.floor(usable / _LEGEND_ROW_FONT_SIZES + 1e-9))
    return max(1, min(rows, LEGEND_SIDE_ENTRIES_PER_COLUMN))


# Matplotlib lowers a legend handle box by _HANDLE_DESCENT * (handleheight -
# _HANDLE_BASE) legend font sizes below its label's baseline and shortens it
# by as much. In a box of the base height, the default, a marker sits half the
# base height above the baseline, about the middle of the label.
_HANDLE_DESCENT = 0.35
_HANDLE_BASE = 0.7
_LABEL_MIDDLE = _HANDLE_BASE / 2


def _center_legend_marker(
    legend_handle: Any, orig_handle: Any, *, center: float
) -> None:
    """Style a legend line like its handle and center its marker on the label.

    ``center`` is the label's middle in points above its baseline.
    """
    legend_handle.update_from(orig_handle)
    legend_handle.set_ydata(np.full(len(legend_handle.get_xdata()), center))


@dataclass(frozen=True, slots=True)
class LegendMarkerBoxes:
    """Legend handle boxes that hold line markers up to ``diameter`` points."""

    diameter: float

    def merged(self, other: "LegendMarkerBoxes") -> "LegendMarkerBoxes":
        """Boxes that hold the markers of both legends."""
        return LegendMarkerBoxes(max(self.diameter, other.diameter))

    def legend_kwargs(self) -> dict[str, Any]:
        """Legend keywords that hold the markers in the current theme."""
        from ._deps import require_matplotlib

        _, mpl = require_matplotlib()
        from matplotlib.font_manager import FontProperties
        from matplotlib.legend_handler import HandlerLine2D

        rc = mpl.rcParams
        fontsize = FontProperties(size=rc["legend.fontsize"]).get_size_in_points()
        radius = 0.5 * self.diameter / fontsize
        # The lowered box spans -d to h - 2 * d about the baseline, with
        # d = _HANDLE_DESCENT * (h - _HANDLE_BASE). Solve for the least h
        # whose box holds the marker's top, then its bottom.
        handleheight = max(
            float(rc["legend.handleheight"]),
            (_LABEL_MIDDLE + radius - 2 * _HANDLE_DESCENT * _HANDLE_BASE)
            / (1 - 2 * _HANDLE_DESCENT),
            _HANDLE_BASE + (radius - _LABEL_MIDDLE) / _HANDLE_DESCENT,
        )
        center = functools.partial(
            _center_legend_marker,
            center=_LABEL_MIDDLE * fontsize,
        )
        return {
            "handlelength": max(float(rc["legend.handlelength"]), 2 * radius),
            "handleheight": handleheight,
            "markerscale": 1.0,
            "handler_map": {mpl.lines.Line2D: HandlerLine2D(update_func=center)},
        }


def size_legend_layout(
    scale: SizeScale,
    values: Sequence[float] | np.ndarray,
    *,
    max_area: float | None = None,
) -> tuple[np.ndarray, LegendMarkerBoxes]:
    """Return the marker areas of a size legend and the boxes that hold them."""
    areas = scale.areas(np.asarray(values, dtype=np.float64))
    largest = max(float(np.max(areas, initial=0.0)), float(max_area or 0.0))
    return areas, LegendMarkerBoxes(float(np.sqrt(largest)))


def item_inches(n_items: int, *, per_item: float, base: float, minimum: float) -> float:
    """Return a panel's extent along an axis of items, such as clones or groups.

    The extent is ``base`` and ``per_item`` for each item up to the
    ``MAX_NAMED_ITEMS`` that an axis names, and at least ``minimum``, so a
    panel of many items stops growing where its names stop.
    """
    return max(float(minimum), base + per_item * min(int(n_items), MAX_NAMED_ITEMS))


def figure_size(
    n_panels: int, *, panel: tuple[float, float], columns: int = 1
) -> tuple[tuple[float, float], int]:
    """Return a figure's default size and the columns that its panels fill.

    Panels of ``panel`` inches fill rows of at most ``columns``, and the
    figure is at most a page wide, ``MAX_FIGURE_WIDTH_INCHES``. A row of
    several panels that would be wider shrinks them to share the page,
    keeping their shape, and one column of panels narrows to the page,
    keeping its height.
    """
    n_columns = max(1, min(int(columns), int(n_panels)))
    n_rows = -(-int(n_panels) // n_columns)
    width = panel[0] * n_columns
    shrink = min(1.0, MAX_FIGURE_WIDTH_INCHES / width)
    height = panel[1] * (shrink if n_columns > 1 else 1.0)
    return (width * shrink, height * n_rows), n_columns


def capped_figsize(
    width: float,
    height: float,
    *,
    max_width: float | None = MAX_FIGURE_WIDTH_INCHES,
) -> tuple[float, float]:
    """Clamp figure width so atlas-scale category counts stay page-sized."""
    resolved_width = float(width)
    if max_width is not None:
        if max_width <= 0:
            raise ValueError("max_width must be positive or None")
        resolved_width = min(resolved_width, float(max_width))
    return (resolved_width, float(height))


def categorical_color_map(
    categories: list[Any],
    *,
    palette: Mapping[Any, str] | None = None,
    palette_name: str = "default",
) -> dict[Any, str]:
    cats = list(categories)
    if palette is not None:
        out = dict(palette)
        for cat in cats:
            if cat not in out:
                raise KeyError(f"Category {cat!r} missing from palette")
        return out
    colors = palette_for_n(len(cats), palette_name=palette_name)
    return dict(zip(cats, colors))


def _held_colormap(mpl: Any, name: str) -> Any:
    """Return the colormap that ``matplotlib.colormaps`` holds under ``name``, or None.

    seaborn adds its colormaps, such as ``"rocket"`` and ``"vlag"``, to the
    registry when it is imported, so a name that the registry does not hold
    is looked up again once seaborn is imported, as a plot that draws with
    seaborn would find it.
    """
    if name not in mpl.colormaps:
        with contextlib.suppress(ImportError):
            import seaborn  # noqa: F401
    return mpl.colormaps.get(name)


def colormap_argument(
    value: object, name: str, *, optional: bool = False
) -> str | None:
    """Check an argument that takes a Matplotlib colormap, and return its name.

    The argument takes the name of a colormap that ``matplotlib.colormaps``
    holds, such as ``"viridis"`` or seaborn's ``"rocket"``, or a ``Colormap``
    that it holds under that ``Colormap``'s own name, and None where
    ``optional`` allows it. Plots draw, record and compose a colormap by its
    name, so a ``Colormap`` of one's own, or one changed from another, such
    as ``viridis.with_extremes(bad="gray")``, is used once
    ``matplotlib.colormaps.register`` holds it under a name of its own. Plots
    check each colormap argument before they read the store.

    Raises:
        TypeError: If ``value`` is of another type.
        ValueError: If ``value`` names no colormap that Matplotlib holds, or
            is a ``Colormap`` that it does not hold under the ``Colormap``'s
            name.
    """
    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    if value is None and optional:
        return None
    if isinstance(value, mpl.colors.Colormap):
        if _held_colormap(mpl, value.name) != value:
            raise ValueError(
                f"{name} is a Colormap that matplotlib.colormaps does not hold "
                f"under its name {value.name!r}; register it with "
                "matplotlib.colormaps.register under a name of its own, and "
                "pass it or that name"
            )
        return str(value.name)
    if not isinstance(value, str):
        allowed = "None, a" if optional else "a"
        raise TypeError(
            f"{name} must be {allowed} Matplotlib colormap name or a registered "
            f"Colormap, not {type(value).__name__}"
        )
    if _held_colormap(mpl, value) is None:
        raise ValueError(
            f"{name} must name a Matplotlib colormap, such as 'viridis'; "
            f"{value!r} is not one"
        )
    return value


def scale_argument[Scale: (ColorScale, DensityOverlay)](
    value: object, name: str, kind: type[Scale]
) -> Scale | None:
    """Check an argument that takes a ``ColorScale`` or ``DensityOverlay``, and return it.

    The scale's colormap follows :func:`colormap_argument`, None included,
    and the returned scale names it, as results record it and
    ``compose_results`` and provenance files read it. Plots check each scale
    argument before they read the store.

    Raises:
        TypeError: If ``value`` is neither None nor a ``kind``, or its
            colormap is of another type.
        ValueError: If its colormap is not one that :func:`colormap_argument`
            takes.
    """
    if value is None:
        return None
    if not isinstance(value, kind):
        raise TypeError(
            f"{name} must be a {kind.__name__} or None, not {type(value).__name__}"
        )
    cmap = colormap_argument(value.cmap, f"{name}.cmap", optional=True)
    return value if cmap == value.cmap else replace(value, cmap=cmap)


def colormap_palette(categories: Sequence[Any], cmap: str | None) -> dict[Any, str]:
    """Color categories in their order, from a colormap or the colorblind palette.

    A colormap is sampled at evenly spaced positions from 0 to 1, so a single
    category takes its first color. Without one, the categories take the
    colorblind palette in order, as a ``CategoricalScale`` of
    ``palette_name="colorblind"`` does, and more categories than its 12
    colors take evenly spaced hues, with a warning that they may not be
    colorblind safe.
    """
    if cmap is None:
        colors = palette_for_n(len(categories), palette_name="colorblind")
        return dict(zip(categories, colors, strict=True))
    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    colormap = mpl.colormaps.get_cmap(cmap)
    last = max(len(categories) - 1, 1)
    return {
        category: mpl.colors.to_hex(colormap(index / last))
        for index, category in enumerate(categories)
    }


def _distinct_categories(values: Sequence[Any] | np.ndarray) -> list[Any]:
    """Distinct non-missing values in order of first appearance.

    Typed arrays are compared by value directly; anything else, such as a
    list of tuples, compares as Python objects.
    """
    array = (
        values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    )
    if array.dtype.kind not in "biufcUS":
        array = np.asarray(array, dtype=object)
    return [
        value
        for value in pd.unique(array.ravel()).tolist()
        if not _is_missing_category(value)
    ]


def resolve_category_scale(
    observed: Sequence[Any] | np.ndarray,
    scale: CategoricalScale | None,
    *,
    context: str = "categorical_scale",
    universe: Sequence[Any]
    | np.ndarray
    | Callable[[], Sequence[Any] | np.ndarray]
    | None = None,
    order: Sequence[Any] | None = None,
    stored: CategoricalScale | None = None,
) -> CategoricalScale:
    """Resolve the displayed order and colors of the observed categories.

    ``scale`` is the caller's and ``stored`` the scale that the plotted
    column or label artifact stores, which ``scale`` merges over field by
    field (:func:`~cytearc.plotting._display.merge_categorical_scales`). A
    caller's ``scale.order`` must list every observed category and sets
    their display order; categories it lists that no plotted cell carries are
    left out of the display. A stored order lists its categories first, in
    its order, and the categories it leaves out follow in natural order, as
    the group axes of every plot order them. Without either, the categories
    take their natural order. Generated colors are assigned over the full
    order, every category of the explicit or stored order and of
    ``universe``, every value of the plotted column or label artifact, so a
    plot of some of its cells keeps the colors of a plot of all of them.
    ``universe`` may be a function that reads those values, which is called
    only when generated colors need them. ``order``, such as requested
    groups, sets the display order instead and must list every observed
    category; it never changes colors. A caller's ``palette`` gives the
    colors of the categories it names and must name every category shown. A
    stored palette colors the categories it names, and each other category
    takes the generated color of its place in the full order. Missing values
    are never categories. The resolved scale fills the missing color, missing
    label, and palette name that the scales leave unset with their defaults.
    """
    given = CategoricalScale() if scale is None else scale
    merged = merge_categorical_scales(stored, given) or given
    present = _distinct_categories(observed)
    listed = (
        ()
        if given.order is not None or stored is None or stored.order is None
        else tuple(stored.order)
    )
    if given.order is not None:
        full_order = list(given.order)
        if len(set(full_order)) != len(full_order):
            raise ValueError(f"{context}.order cannot contain duplicates")
        unlisted = [value for value in present if value not in full_order]
        if unlisted:
            raise ValueError(
                f"{context}.order is missing observed values: "
                + ", ".join(map(str, unlisted[:10]))
            )
    else:
        known = set(listed)
        # Every value of the column places the categories that the stored
        # order leaves out, unless a caller's palette colors them all.
        if (
            callable(universe)
            and given.palette is None
            and any(value not in known for value in present)
        ):
            universe = universe()
        values = [] if universe is None or callable(universe) else universe
        full_order = labels_in_stored_order(
            dict.fromkeys([*listed, *present, *_distinct_categories(values)]),
            None if not listed else CategoricalScale(order=listed),
        )
    shown = set(present)
    if order is not None:
        unrequested = [value for value in present if value not in set(order)]
        if unrequested:
            raise ValueError(
                "Requested groups are missing observed values: "
                + ", ".join(map(str, unrequested[:10]))
            )
        display = list(dict.fromkeys(value for value in order if value in shown))
    else:
        display = [value for value in full_order if value in shown]
    palette_name = merged.palette_name or "default"
    if given.palette is not None:
        # A caller's palette needs the colors of the categories shown,
        # whichever order lists them.
        colors = categorical_color_map(
            display, palette=given.palette, palette_name=palette_name
        )
    else:
        stored_colors = merged.palette or {}
        colors = (
            dict(stored_colors)
            if all(value in stored_colors for value in display)
            else {
                **categorical_color_map(full_order, palette_name=palette_name),
                **stored_colors,
            }
        )
    labels = merged.labels
    missing_color, missing_label = missing_category_entry(merged)
    return CategoricalScale(
        order=tuple(display),
        palette={value: colors[value] for value in display},
        labels=(
            None
            if labels is None
            else {value: str(labels.get(value, value)) for value in display}
        ),
        missing_color=missing_color,
        missing_label=missing_label,
        palette_name=palette_name,
    )


def missing_category_entry(scale: CategoricalScale) -> tuple[str, str]:
    """The color that draws cells without a category, and the label that names them.

    A scale that leaves either unset takes the default.
    """
    return (
        MISSING_CATEGORY_COLOR if scale.missing_color is None else scale.missing_color,
        MISSING_CATEGORY_LABEL if scale.missing_label is None else scale.missing_label,
    )


def category_label(scale: CategoricalScale, value: Any) -> str:
    """Display text for one category of a resolved scale."""
    if scale.labels is None:
        return str(value)
    return str(scale.labels.get(value, value))


def _is_missing_category(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def resolve_color_limits(values: Any, scale: ColorScale) -> tuple[float, float]:
    """Resolve continuous color limits under the policy every plot shares.

    Limits come from the finite values, or from their ``quantiles``, and an
    explicit ``vmin`` or ``vmax`` replaces the matching side. A ``vcenter``
    pivot widens derived limits so a diverging map works on one-sided values.
    Tied or inverted limits keep the explicit or lower side and extend the
    other side by one unit, so a constant value takes the low end of the map.
    A log scale takes its limits from the positive values only, as
    Matplotlib's ``LogNorm`` masks the others, and extends tied limits by a
    factor of ten; explicit limits must still be positive.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    log = scale.scale == "log"
    finite = data[scalable_values(data, scale)]
    if finite.size == 0:
        low, high = (1.0, 10.0) if log else (0.0, 1.0)
    elif scale.quantiles is not None:
        low, high = (float(value) for value in np.quantile(finite, scale.quantiles))
    else:
        low, high = float(finite.min()), float(finite.max())
    if scale.vmin is not None:
        low = float(scale.vmin)
    if scale.vmax is not None:
        high = float(scale.vmax)
    if scale.vcenter is not None:
        center = float(scale.vcenter)
        if (scale.vmin is not None and center <= low) or (
            scale.vmax is not None and center >= high
        ):
            raise ValueError("vcenter must be strictly between the color limits")
        if not low < center < high:
            margin = max((max(high, center) - min(low, center)) * 1e-6, 1e-9)
            low = min(low, center - margin)
            high = max(high, center + margin)
    elif high <= low:
        if scale.vmax is not None and scale.vmin is None:
            low = high / 10.0 if log else high - 1.0
        else:
            high = low * 10.0 if log else low + 1.0
    if log and low <= 0:
        raise ValueError("Log color scale requires positive values")
    return low, high


def scalable_values(values: Any, scale: ColorScale) -> np.ndarray:
    """Mark the values a continuous scale can color.

    Finite values qualify; a log scale also needs them positive. Other values
    take the scale's missing color.
    """
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    if scale.scale == "log":
        finite &= data > 0
    return finite


def continuous_norm(
    mpl: Any,
    *,
    vmin: float,
    vmax: float,
    vcenter: float | None,
    scale: str = "linear",
) -> Any:
    """Build the Matplotlib norm for resolved limits and a ColorScale scale."""
    if vmax <= vmin:
        vmax = vmin + 1.0
    if scale == "log":
        if vmin <= 0:
            raise ValueError("Log color scale requires positive values")
        return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    if scale == "symlog":
        return mpl.colors.SymLogNorm(
            linthresh=max(abs(vmax - vmin) * 0.001, 1e-12),
            vmin=vmin,
            vmax=vmax,
        )
    if vcenter is None:
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if not vmin < vcenter < vmax:
        raise ValueError("vcenter must be strictly between the color limits")
    return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def padded_square_limits(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pad finite coordinates by 5% on each axis and square the window."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.shape != yy.shape:
        raise ValueError("Coordinate columns must have matching shapes")
    finite = np.isfinite(xx) & np.isfinite(yy)
    if not finite.any():
        raise ValueError("No finite coordinates are available to plot")
    xx = xx[finite]
    yy = yy[finite]
    x_pad = 0.05 * (float(xx.max() - xx.min()) or 1.0)
    y_pad = 0.05 * (float(yy.max() - yy.min()) or 1.0)
    return square_axis_limits(
        (float(xx.min() - x_pad), float(xx.max() + x_pad)),
        (float(yy.min() - y_pad), float(yy.max() + y_pad)),
    )


def square_axis_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pad axis limits so the data window is square.

    Keeps equal-aspect embeddings visually square after legends and colorbars
    shrink the available axes width.
    """
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    span = max(x1 - x0, y1 - y0, 1e-12)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * span
    return (cx - half, cx + half), (cy - half, cy + half)


def scatter_edgecolor(theme: str = "notebook") -> str:
    """Marker edge color that stays readable on light and dark themes."""
    if theme == "dark":
        # Mid gray keeps dark fills legible without turning markers into rings.
        return "#8f8f8f"
    return "#333333"


def foreground_color(theme: str = "notebook") -> str:
    """High-contrast foreground for annotations and segment borders."""
    return "#e8e8e8" if theme == "dark" else "#333333"


def finish_embedding_axes(
    ax: Any,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    frame: FrameStyle = "minimal",
) -> None:
    """Apply shared CyteArc chrome to a 2D embedding axes."""
    if frame not in ("axes", "minimal", "none"):
        raise ValueError("frame must be one of 'axes', 'minimal', 'none'")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    if frame == "axes":
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    if frame == "none" and hasattr(ax, "spines"):
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        ax.set_title(title)


def apply_figure_chrome(figure: Any, theme: str = "notebook") -> None:
    """Apply CyteArc figure background and spine defaults after axes creation."""
    opaque = theme != "dark"
    if opaque:
        figure.patch.set_facecolor("white")
        figure.patch.set_alpha(1.0)
    else:
        figure.patch.set_alpha(0)
    for ax in figure.axes:
        if opaque:
            ax.patch.set_facecolor("white")
            ax.patch.set_alpha(1.0)
        else:
            ax.patch.set_alpha(0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def positive_argument(value: object, name: str) -> float:
    """Check a size or width that must be positive and finite, and return it as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not positive and finite.
    """
    try:
        resolved = float_argument(value, name)
    except ValueError:
        resolved = np.nan
    if not resolved > 0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


def theme_argument(name: object) -> str:
    """Check the name of a theme of ``THEMES``, and return it.

    ``close_figures_on_error`` checks with it the ``theme`` of each plot that it wraps before the plot runs, run plots check theirs before they read the run, the mean-variance plot of ``features.hvgs`` before the store computes anything, and ``theme_context`` when a plot enters the theme.

    Raises:
        KeyError: If ``name`` names no theme, a value that is not a string included.
    """
    if isinstance(name, str) and name in THEMES:
        return name
    raise KeyError(f"Unknown theme {name!r}. Choose from: {sorted(THEMES)}")


@contextmanager
def theme_context(name: str = "notebook") -> Iterator[None]:
    """Apply a CyteArc theme's rcParams inside the context and restore them on exit.

    Matplotlib's interactive mode is the exception: it stays as the body of
    the context leaves it. Jupyter's inline backend turns interactive mode on
    when it loads, which happens when the session draws its first figure, so
    a session whose first figure is a CyteArc plot stays interactive. A call
    of ``plt.ion()`` or ``plt.ioff()`` in the body also lasts beyond the
    context.
    """
    name = theme_argument(name)
    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    interactive = mpl.is_interactive()
    try:
        with mpl.rc_context(THEMES[name]):
            try:
                yield
            finally:
                interactive = mpl.is_interactive()
    finally:
        mpl.interactive(interactive)
