"""Figures of pooled CRISPR screens."""

from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import betaincinv

from ..perturbation.edistance import EDISTANCE_STATUSES, EDISTANCE_TESTED
from ..perturbation.efficiency import EFFICIENCY_STATUSES, EFFICIENCY_TESTED
from ..perturbation.mixscape import MIXSCAPE_STATUSES
from ..perturbation.mixtures import (
    FIT_STATUSES,
    gaussian_posterior,
    poisson_gaussian_posterior,
)
from ..storage.refs import ArtifactRef
from ..utils.arguments import float_argument, integer_argument
from ._contracts import CategoricalScale, ColorScale, PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._labels import place_layout_labels, register_layout_labels
from ._style import (
    COLORBLIND_PALETTE,
    DEFAULT_RASTERIZE_THRESHOLD,
    apply_figure_chrome,
    default_point_size,
    finish_embedding_axes,
    foreground_color,
    padded_square_limits,
    theme_context,
)

# Colours by role, which every screen figure shares. The highlight marks what
# a figure is about: called cells, fitted fits, significant rows and pairs,
# knocked-out cells, discovery pairs and positive controls; the muted grey
# marks the rest. Cells without a call take a lighter grey, cells with several
# called guides red, and the reference colour marks what a figure is read
# against: thresholds, the Benjamini-Hochberg cut, the fitted perturbed
# component and the negative controls.
_HIGHLIGHT_COLOR = "#2b6cb0"
_MUTED_COLOR = "#9e9e9e"
_UNCALLED_COLOR = "#bdbdbd"
_SEVERAL_GUIDES_COLOR = "#d1495b"
_REFERENCE_COLOR = "#d55e00"

_PANELS = ("guides_per_cell", "cells_per_guide", "top_fraction")
# A library of at most this many guides gets one labelled bar per guide; a
# larger one is drawn as guides ranked by their cells.
_LABELLED_GUIDES = 40
_CALL_GROUPS = (
    ("No guide called", _UNCALLED_COLOR),
    ("One guide called", _HIGHLIGHT_COLOR),
    ("Several guides called", _SEVERAL_GUIDES_COLOR),
)

_OVERVIEW_PANELS = ("thresholds", "statuses")
# The statuses whose fits keep their parameters, so a panel can draw the model.
_MODEL_STATUSES = ("fitted", "no_positive")
# The posterior of each fitted method and the columns that hold its parameters.
_POSTERIORS: dict[str, tuple[Callable[..., np.ndarray], tuple[str, ...]]] = {
    "poisson_gaussian": (
        poisson_gaussian_posterior,
        ("poisson_rate", "gaussian_mean", "gaussian_std", "gaussian_weight"),
    ),
    "gaussian": (gaussian_posterior, ("low_mean", "high_mean", "std", "high_weight")),
}
# A curve over the integers from 1 to n, such as the posterior at each UMI
# count of a guide or the QQ band of n p-values, is drawn at every integer
# when n is at most this, and at this many spread on the log scale otherwise.
_LOG_SPREAD_POINTS = 1000
_CALLED_LABEL = "Cells called"
_NOT_CALLED_LABEL = "Cells not called"
_POSTERIOR_LABEL = "Posterior of the guide population"
_THRESHOLD_LABEL = "Threshold of the model"
_RAISED_LABEL = "Threshold raised to min_umi"
_LEGEND_ORDER = (
    _CALLED_LABEL,
    _NOT_CALLED_LABEL,
    _POSTERIOR_LABEL,
    _THRESHOLD_LABEL,
    _RAISED_LABEL,
)

_EFFICIENCY_PANEL = "efficiency"
# A point's area, in points squared, grows linearly with the row's perturbed
# cells from the first area to the second, which the row with the most takes.
_POINT_AREAS = (16.0, 144.0)
# A fold change of minus infinity is drawn at the left edge with this marker.
_MINUS_INFINITY_MARKER = "<"
_NO_UMI_LABEL = "No UMI remains in the perturbed cells"

_EDISTANCE_PANEL = "edistance"

_CLASSES_PANEL = "classes"
_SCORES_PANEL = "scores"
_NP_LABEL = "NP (not perturbed)"
# A bar's share of the space between two neighbouring targets.
_CLASS_BAR_HEIGHT = 0.7
_CONTROL_CURVE_LABEL = "Fitted control component"
_PERTURBED_CURVE_LABEL = "Fitted perturbed component"
# A scores panel sorts its cells' scores into this many bins of equal width.
_SCORE_BINS = 30
# A fitted normal is drawn at this many points, from the first bin or 3
# standard deviations below the mean of either component, whichever is lower,
# to the last bin or as far above, whichever is higher.
_CURVE_POINTS = 200
_CURVE_REACH = 3.0
# A scores figure puts at most this many panels in a row.
_SCORE_COLUMNS = 4

_VOLCANO_PANEL = "volcano"
_CALIBRATION_PANELS = ("qq", "fold_changes", "controls")
# A plot of association tests rasterizes the points of a panel that draws more
# than this many.
_ASSOCIATION_RASTERIZE_THRESHOLD = 100_000
# A p-value below this, such as 0 when it underflows, is drawn at the top edge
# of a -log10 axis.
_SMALLEST_P_VALUE = 1e-300
_TOP_EDGE_MARKER = "^"
_TOP_EDGE_LABEL = "p-value below 1e-300"
_BEYOND_AXIS_LABEL = "Beyond the axis"
_NO_COUNTS_LABEL = "No counts in the perturbed cells"
_CUT_LABEL = "Benjamini-Hochberg cut"
# The label of a p-value axis names the side of a one-sided test.
_SIDE_WORDS = {"left": " for a decrease", "right": " for an increase"}
# The label of a point, such as a volcano's pair or a named perturbation,
# sits this many points beside it. A label that would overlap another steps
# up or down by its height and this gap, in points, at most this many times
# each way.
_LABEL_OFFSET = 5.0
_LABEL_GAP = 2.0
_LABEL_STEPS = 8
# A QQ band holds this share of the null distribution of each ranked p-value.
_BAND_PROBABILITY = 0.95
# The points of a strip of control p-values spread at random, with a fixed
# seed, over this distance on each side of the strip's position.
_STRIP_HALF_WIDTH = 0.25
_STRIP_SEED = 0
_FOLD_CHANGE_BINS = 30

_HEATMAP_PANEL = "heatmap"
_EMBEDDING_PANEL = "embedding"
# HDBSCAN's cluster of a perturbation that it leaves unassigned.
_UNASSIGNED = -1
# Clusters take CyteArc's colourblind-safe palette without its black, its
# yellow and the two colours that read as grey, in the order in which they
# first appear in the leaf order, and the ninth takes the first colour again.
_CLUSTER_COLORS = tuple(
    color
    for color in COLORBLIND_PALETTE
    if color not in {"#F0E442", "#000000", "#6F4E7C", "#8C564B"}
)
# The grey of unassigned perturbations recedes on each theme and stays apart
# from every cluster colour, even with colour vision deficiency.
_UNASSIGNED_COLOR = "#d4d4d4"
_DARK_UNASSIGNED_COLOR = "#4d4d4d"
_UNASSIGNED_LABEL = "Unassigned"
# A heatmap names its perturbations when it draws at most this many, giving
# each this many inches, and is otherwise this many inches square.
_NAMED_PERTURBATIONS = 50
_NAMED_ROW_INCHES = 0.16
_HEATMAP_INCHES = 6.0
_HEATMAP_MIN_INCHES = 3.0
# A heatmap's own figure changes its width at most this many times to make
# the heatmap square.
_SQUARE_PASSES = 3
# The cluster bar's width and its gap to the heatmap, in shares of the
# heatmap's width.
_BAR_WIDTH = 0.03
_BAR_GAP = 0.01
_CORRELATION_CMAP = "RdBu_r"
_DISTANCE_CMAP = "Blues_r"
# The ring around a named perturbation has this many times the area of its
# point.
_RING_AREA = 4.0


def _load(store: Any, method: str, ref: ArtifactRef, **keywords: Any) -> Any:
    """Read a saved record with the store's narrow loader ``perturbation.<method>``."""
    loader = getattr(getattr(store, "perturbation", None), method, None)
    if not callable(loader):
        raise TypeError(f"store does not provide perturbation.{method}")
    return loader(ref, **keywords)


def _call_group(n_guides: np.ndarray) -> np.ndarray:
    groups = np.full(len(n_guides), _CALL_GROUPS[2][0], dtype=object)
    groups[n_guides == 0] = _CALL_GROUPS[0][0]
    groups[n_guides == 1] = _CALL_GROUPS[1][0]
    return groups


@close_figures_on_error
def guide_calls(
    store: Any,
    calls: ArtifactRef,
    *,
    max_guides_per_cell: int = 5,
    bins: int = 20,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how many guides each cell carries and how many cells carry each guide.

    ``calls`` is a ``guide_calls`` artifact, such as one from
    ``ds.perturbation.import_guide_calls``. The first panel counts cells by
    their number of called guides; its last bar gathers the cells with
    ``max_guides_per_cell`` or more. The second counts, for each guide, the
    cells in which it is the one called guide and the cells in which it is one
    of several: one labelled bar per guide for a library of at most 40 guides,
    and guides ranked by their cells for a larger one. The third shows the top
    guide's share of each cell's guide UMIs, for cells with guide UMIs, by
    call. A cell with several called guides whose top guide holds nearly all
    of its guide UMIs may carry one guide and ambient molecules of the others.
    The plot raises ValueError when the calls hold no cell.
    """
    max_guides_per_cell = integer_argument(
        max_guides_per_cell, "max_guides_per_cell", minimum=2
    )
    bins = integer_argument(bins, "bins", minimum=1)
    loaded = _load(store, "load_guide_calls", calls)
    cells = loaded.cells
    if cells.empty:
        raise ValueError("The guide calls hold no cell, so there is nothing to plot")
    n_guides = cells["n_guides"].to_numpy(dtype=np.int64)
    bucket_counts = np.bincount(
        np.minimum(n_guides, max_guides_per_cell), minlength=max_guides_per_cell + 1
    )
    guides_per_cell = pd.DataFrame(
        {
            "guides": [str(count) for count in range(max_guides_per_cell)]
            + [f"{max_guides_per_cell}+"],
            "cells": bucket_counts,
        }
    )
    single_calls = cells.loc[n_guides == 1, "call"].value_counts()
    all_calls = loaded.calls["guide"].value_counts()
    guide_ids = pd.Index(loaded.guides, name="guide")
    cells_per_guide = pd.DataFrame(
        {
            "one_guide_cells": single_calls.reindex(guide_ids, fill_value=0),
            "several_guide_cells": (
                all_calls.reindex(guide_ids, fill_value=0)
                - single_calls.reindex(guide_ids, fill_value=0)
            ),
        },
        index=guide_ids,
    )
    cells_per_guide["cells"] = (
        cells_per_guide["one_guide_cells"] + cells_per_guide["several_guide_cells"]
    )
    cells_per_guide = cells_per_guide.sort_values(
        ["cells", "one_guide_cells"], ascending=False, kind="stable"
    ).reset_index()
    groups = _call_group(n_guides)
    fractions = cells["top_fraction"].to_numpy(dtype=np.float64)
    counted = np.isfinite(fractions)
    edges = np.linspace(0.0, 1.0, bins + 1)
    top_fraction = pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_stop": edges[1:],
            **{
                group: np.histogram(fractions[counted & (groups == group)], edges)[0]
                for group, _color in _CALL_GROUPS
            },
        }
    )
    labelled = len(guide_ids) <= _LABELLED_GUIDES
    if figsize is None and target is None:
        figsize = (13.0, max(3.6, 0.22 * len(guide_ids) + 1.2) if labelled else 3.6)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(_PANELS), figsize=figsize, n_columns=3
        )
        ax = axes["guides_per_cell"]
        bar_colors = [_UNCALLED_COLOR, _HIGHLIGHT_COLOR] + [_SEVERAL_GUIDES_COLOR] * (
            max_guides_per_cell - 1
        )
        ax.bar(guides_per_cell["guides"], guides_per_cell["cells"], color=bar_colors)
        ax.set_xlabel("Called guides")
        ax.set_ylabel("Cells")
        ax.set_title("Guides per cell")

        ax = axes["cells_per_guide"]
        if labelled:
            positions = np.arange(len(cells_per_guide))[::-1]
            ax.barh(
                positions,
                cells_per_guide["one_guide_cells"],
                color=_HIGHLIGHT_COLOR,
                label="Only guide called",
            )
            ax.barh(
                positions,
                cells_per_guide["several_guide_cells"],
                left=cells_per_guide["one_guide_cells"],
                color=_SEVERAL_GUIDES_COLOR,
                label="One of several called",
            )
            ax.set_yticks(positions, cells_per_guide["guide"].tolist())
            ax.set_xlabel("Cells")
        else:
            ranks = np.arange(1, len(cells_per_guide) + 1)
            ax.plot(
                ranks,
                cells_per_guide["cells"],
                color=foreground_color(theme),
                label="All cells with the guide",
            )
            ax.plot(
                ranks,
                cells_per_guide["one_guide_cells"],
                color=_HIGHLIGHT_COLOR,
                label="Only guide called",
            )
            ax.set_xlabel("Guides, ranked by cells")
            ax.set_ylabel("Cells")
        ax.set_title("Cells per guide")
        if show_legend:
            ax.legend(frameon=False)

        ax = axes["top_fraction"]
        for group, color in _CALL_GROUPS:
            values = fractions[counted & (groups == group)]
            if len(values):
                ax.hist(
                    values,
                    bins=edges,
                    histtype="step",
                    linewidth=1.4,
                    color=color,
                    label=group,
                )
        ax.set_xlabel("Top guide's share of guide UMIs")
        ax.set_ylabel("Cells")
        ax.set_title("Top guide share")
        if show_legend:
            ax.legend(frameon=False)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "guides_per_cell": guides_per_cell,
            "cells_per_guide": cells_per_guide,
            "top_fraction": top_fraction,
        },
        legends=(LegendSpec(kind="categorical", label="Call"),),
        scales=(),
        provenance=PlotProvenance(
            assay=calls.assay,
            cell_key=None,
            n_cells=len(cells),
            renderer="matplotlib",
            notes=("guide_calls",),
            extras={
                "guide_calls": calls.to_dict(),
                "source": loaded.source,
                "max_guides_per_cell": max_guides_per_cell,
                "bins": bins,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _distinct_values(
    values: Sequence[Any] | None, name: str, noun: str, *, numbers: bool = False
) -> list[Any] | None:
    """Return the distinct labels of ``values`` in their order, or None.

    With ``numbers``, the values are whole numbers instead, which may be NumPy
    integers but not booleans and come back as Python integers.
    """
    if values is None:
        return None
    if (
        isinstance(values, str | bytes)
        or not isinstance(values, Sequence)
        or not all(
            isinstance(value, int | np.integer) and not isinstance(value, bool)
            if numbers
            else isinstance(value, str)
            for value in values
        )
    ):
        raise TypeError(f"{name} must be a sequence of {noun}s or None")
    if not values:
        raise ValueError(f"{name} must name at least one {noun}")
    return list(dict.fromkeys(int(value) if numbers else value for value in values))


def _quoted(values: Sequence[Any]) -> str:
    """Return the first five values quoted, and how many more there are."""
    shown = ", ".join(repr(value) for value in values[:5])
    return shown if len(values) <= 5 else f"{shown} and {len(values) - 5:,} more"


def _require_known(
    requested: Sequence[Any] | None,
    known: Collection[Any],
    *,
    holder: str,
    noun: str,
) -> None:
    """Refuse the names of ``requested`` that ``known`` lacks, if any.

    Raises:
        ValueError: Naming the first five of them, such as "The clusters have
            no perturbation named 'A'".
    """
    unknown = [name for name in requested or () if name not in known]
    if unknown:
        raise ValueError(f"{holder} have no {noun} named {_quoted(unknown)}")


def _status_counts(
    counts: pd.Series, statuses: Sequence[str], *, leave_out: str | None = None
) -> list[str]:
    """Count the rows of each status of ``statuses`` but ``leave_out``, as "3 no_cells".

    ``counts`` counts rows by status; a status without rows is left out.
    """
    return [
        f"{int(counts[status]):,} {status}"
        for status in statuses
        if status != leave_out and status in counts.index
    ]


def _untested_message(by: str, value: str, others: list[str]) -> str:
    """Say that no row was tested, so the plot has no ``value`` to draw.

    ``others`` counts the rows of each other status, if any.
    """
    held = f" ({', '.join(others)})" if others else ""
    return f"No {by} was tested{held}, so there is no {value} to plot"


def _alpha_argument(alpha: object) -> float:
    """Check the false discovery rate of a figure's significant rows."""
    resolved = float_argument(alpha, "alpha")
    if not 0 < resolved < 1:
        raise ValueError("alpha must be above 0 and below 1")
    return resolved


def _log_plus_one(values: np.ndarray) -> np.ndarray:
    logged: np.ndarray = np.log10(values + 1.0)
    return logged


def _exp_minus_one(values: np.ndarray) -> np.ndarray:
    counts: np.ndarray = 10.0**values - 1.0
    return counts


def _log_count_scale(ax: Any, axis: str, largest: float) -> None:
    """Put counts on a log scale of count + 1, which keeps a place for 0.

    Labelled ticks fall at 0 and the powers of ten, and also at 2 and 5 times
    each power when the counts, up to ``largest``, stay below 100.
    """
    from matplotlib.ticker import FixedLocator, StrMethodFormatter

    getattr(ax, f"set_{axis}scale")(
        "function", functions=(_log_plus_one, _exp_minus_one)
    )
    decades = 10.0 ** np.arange(11)
    steps = [1.0, 2.0, 5.0] if largest < 100 else [1.0]
    major = [0.0, *np.outer(decades, steps).ravel().tolist()]
    minor = np.outer(decades, np.arange(2, 10)).ravel().tolist()
    ticks = getattr(ax, f"{axis}axis")
    ticks.set_major_locator(FixedLocator(major))
    ticks.set_minor_locator(FixedLocator(minor))
    ticks.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def _log_spread_integers(n: int) -> np.ndarray:
    """Return the integers from 1 to ``n`` at which a curve over them is drawn.

    Every integer when ``n`` is at most 1,000, and otherwise 1,000 integers
    spread evenly on the log scale and rounded, without repeats.
    """
    if n <= _LOG_SPREAD_POINTS:
        return np.arange(1, n + 1, dtype=np.int64)
    spread = np.rint(np.geomspace(1.0, float(n), _LOG_SPREAD_POINTS))
    integers: np.ndarray = np.unique(spread.astype(np.int64))
    return integers


def _legend_entries(
    axes: Iterable[Any], order: Sequence[str]
) -> tuple[list[Any], list[str]]:
    """Return the first handle of each label that ``axes`` draw, in ``order``."""
    entries: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    labels = [label for label in order if label in entries]
    return [entries[label] for label in labels], labels


def _place_legend(
    figure: Any,
    ax: Any,
    handles: list[Any],
    labels: list[str],
    *,
    owns: bool,
    show_legend: bool,
    loc: str = "best",
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    """Centre the legend right of the plot's own figure, clear of a long title.

    In a caller's figure the legend goes in ``ax`` at ``loc``, so that a
    composed figure keeps it with its panel.
    """
    if not (show_legend and handles):
        return
    if owns:
        figure.legend(handles, labels, loc="outside right center", frameon=False)
    else:
        ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            frameon=False,
            fontsize="small",
        )


def _threshold_panels(
    loaded: Any,
    selected: pd.DataFrame,
    histograms: pd.DataFrame,
    *,
    n_columns: int,
    target: Any | None,
    figsize: tuple[float, float] | None,
    theme: str,
    show_legend: bool,
) -> tuple[Any, dict[Hashable, Any], bool, dict[str, pd.DataFrame]]:
    """Draw one panel per row of ``selected``, a guide in a sample.

    Returns the figure, the axes, whether the plot owns the figure, and the
    histogram and posterior tables of the panels.
    """
    rows = selected.to_dict("records")
    by_sample = loaded.sample_column is not None
    keys: list[Hashable] = [
        (row["sample"], row["guide"]) if by_sample else row["guide"] for row in rows
    ]
    if figsize is None and target is None:
        n_rows = -(-len(rows) // n_columns)
        figsize = (3.9 * n_columns + (2.3 if show_legend else 0.0), 3.0 * n_rows)
    foreground = foreground_color(theme)
    panel_histograms: list[pd.DataFrame] = []
    posteriors: list[tuple[Any, str, np.ndarray, np.ndarray]] = []
    twins: dict[Hashable, Any] = {}
    figure, axes, owns = normalize_axes_target(
        target, panel_keys=keys, figsize=figsize, n_columns=n_columns
    )
    for key, row in zip(keys, rows, strict=True):
        counts = histograms[
            histograms["sample"].isin([row["sample"]])
            & (histograms["guide"] == row["guide"])
        ]
        panel_histograms.append(counts)
        drawn = _threshold_panel(
            axes[key], row, counts, method=loaded.method, foreground=foreground
        )
        if drawn is not None:
            twins[key], posterior_counts, posterior = drawn
            posteriors.append(
                (row["sample"], row["guide"], posterior_counts, posterior)
            )
    handles, labels = _legend_entries([*axes.values(), *twins.values()], _LEGEND_ORDER)
    # The posterior axes lies over its panel, so a legend in the panel goes on
    # top of it.
    first = twins.get(keys[0], axes[keys[0]])
    _place_legend(figure, first, handles, labels, owns=owns, show_legend=show_legend)
    apply_figure_chrome(figure, theme)
    for twin in twins.values():
        twin.spines["right"].set_visible(True)
    axes.update({(key, "posterior"): twin for key, twin in twins.items()})
    tables = {
        "histograms": pd.concat(panel_histograms, ignore_index=True),
        "posterior": _posterior_table(posteriors),
    }
    return figure, axes, owns, tables


def _threshold_panel(
    ax: Any,
    row: dict[str, Any],
    counts: pd.DataFrame,
    *,
    method: str,
    foreground: str,
) -> tuple[Any, np.ndarray, np.ndarray] | None:
    """Draw the histogram, thresholds and model of one guide in one sample.

    Returns the posterior axes with the counts and posterior drawn on them, or
    None when the fit has no model to draw.
    """
    status = str(row["status"])
    umis = counts["umis"].to_numpy(dtype=np.int64)
    cells = counts["cells"].to_numpy(dtype=np.int64)
    threshold = int(row["threshold"])
    call_threshold = row["call_threshold"]
    limit = threshold if pd.isna(call_threshold) else int(call_threshold)
    called = (limit > 0) & (umis >= limit)
    _log_count_scale(ax, "x", max(int(umis.max(initial=0)), limit))
    ax.set_yscale("log")
    for chosen, color, label in (
        (called, _HIGHLIGHT_COLOR, _CALLED_LABEL),
        (~called, _UNCALLED_COLOR, _NOT_CALLED_LABEL),
    ):
        if chosen.any():
            # The edge keeps the narrow bars of large counts visible.
            ax.bar(
                umis[chosen],
                cells[chosen],
                width=1.0,
                color=color,
                edgecolor=color,
                linewidth=0.5,
                label=label,
            )
    # A bar of one cell keeps a visible height on the log scale.
    ax.set_ylim(bottom=0.5)
    if threshold > 0:
        ax.axvline(
            threshold,
            color=_REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=_THRESHOLD_LABEL,
        )
    if limit != threshold:
        ax.axvline(
            limit,
            color=_REFERENCE_COLOR,
            linestyle=":",
            linewidth=1.6,
            label=_RAISED_LABEL,
        )
    # The axis reaches 10 UMIs at least, so a lone bar at 0 stays narrow.
    ax.set_xlim(right=max(ax.get_xlim()[1], 10.0))
    if status != "fitted":
        ax.text(
            0.97,
            0.96,
            f"{status}, no threshold",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    ax.set_xlabel("UMIs of the guide in a cell")
    ax.set_ylabel("Cells")
    sample, guide = row["sample"], row["guide"]
    ax.set_title(guide if sample is None else f"{guide} in {sample}")
    if status not in _MODEL_STATUSES:
        return None
    function, names = _POSTERIORS[method]
    posterior_counts = _log_spread_integers(int(row["max_umi"]))
    posterior = function(posterior_counts, **{name: float(row[name]) for name in names})
    twin = ax.twinx()
    twin.plot(
        posterior_counts,
        posterior,
        color=foreground,
        linewidth=1.4,
        # A posterior at one count is a point, which a line cannot show.
        marker="o" if len(posterior_counts) == 1 else None,
        markersize=4,
        label=_POSTERIOR_LABEL,
    )
    twin.set_ylim(0.0, 1.0)
    twin.set_yticks([0.0, 0.5, 1.0])
    twin.set_ylabel(_POSTERIOR_LABEL)
    return twin, posterior_counts, posterior


def _posterior_table(
    posteriors: list[tuple[Any, str, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Return one row per sample, guide and drawn count, with the posterior there."""
    lengths = [len(counts) for _, _, counts, _ in posteriors]
    samples = np.asarray([sample for sample, _, _, _ in posteriors], dtype=object)
    guides = np.asarray([guide for _, guide, _, _ in posteriors], dtype=object)
    return pd.DataFrame(
        {
            "sample": np.repeat(samples, lengths),
            "guide": np.repeat(guides, lengths),
            "umis": np.concatenate(
                [
                    np.zeros(0, dtype=np.int64),
                    *(counts for _, _, counts, _ in posteriors),
                ]
            ),
            "posterior": np.concatenate(
                [np.zeros(0), *(values for _, _, _, values in posteriors)]
            ),
        }
    )


def _threshold_overview(
    selected: pd.DataFrame,
    *,
    rasterized: bool,
    target: Any | None,
    figsize: tuple[float, float] | None,
    theme: str,
    show_legend: bool,
) -> tuple[Any, dict[Hashable, Any], bool]:
    """Draw every fit's threshold against its cells, and the fits of each status."""
    if figsize is None and target is None:
        figsize = (9.0, 3.8)
    figure, axes, owns = normalize_axes_target(
        target, panel_keys=list(_OVERVIEW_PANELS), figsize=figsize, n_columns=2
    )
    fitted = selected["status"].to_numpy(dtype=object) == "fitted"
    n_umi_cells = selected["n_umi_cells"].to_numpy(dtype=np.int64)
    threshold = selected["threshold"].to_numpy(dtype=np.int64)
    ax = axes["thresholds"]
    _log_count_scale(ax, "x", n_umi_cells.max(initial=0))
    _log_count_scale(ax, "y", threshold.max(initial=0))
    for chosen, color, label, order in (
        (fitted, _HIGHLIGHT_COLOR, "fitted", 3.0),
        (~fitted, _MUTED_COLOR, "other statuses", 2.0),
    ):
        if chosen.any():
            ax.scatter(
                n_umi_cells[chosen],
                threshold[chosen],
                s=14,
                color=color,
                alpha=0.75,
                linewidths=0,
                label=label,
                zorder=order,
                rasterized=rasterized,
            )
    ax.set_xlabel("Cells with a UMI of the guide")
    ax.set_ylabel("Threshold of the model, UMIs")
    ax.set_title("Thresholds")
    if show_legend and len(selected):
        ax.legend(frameon=False, loc="best")
    ax = axes["statuses"]
    fits = selected["status"].value_counts().reindex(list(FIT_STATUSES), fill_value=0)
    positions = np.arange(len(FIT_STATUSES))[::-1]
    bars = ax.barh(
        positions,
        fits.to_numpy(dtype=np.int64),
        color=[
            _HIGHLIGHT_COLOR if status == "fitted" else _MUTED_COLOR
            for status in FIT_STATUSES
        ],
    )
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set_yticks(positions, list(FIT_STATUSES))
    ax.set_xlabel("Fits")
    ax.set_title("Fits by status")
    apply_figure_chrome(figure, theme)
    return figure, axes, owns


@close_figures_on_error
def guide_thresholds(
    store: Any,
    thresholds: ArtifactRef,
    *,
    guides: Sequence[str] | None = None,
    samples: Sequence[str] | None = None,
    max_panels: int = 12,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the UMI thresholds that guide calling fitted to each guide in each sample.

    ``thresholds`` is guide calls of ``ds.perturbation.call_guides`` with
    ``method="poisson_gaussian"`` or ``"gaussian"``, or their
    ``guide_thresholds`` artifact, as ``ds.perturbation.load_guide_thresholds``
    reads them.

    With ``guides``, each selected guide gets one panel per sample. Bars count
    the cells at each UMI count of the guide, blue where the count is called
    and grey where it is not, on log scales; the UMI axis is a log scale of
    count + 1, so the cells without a UMI of the guide appear at 0. A line on
    the right axis shows the posterior of the guide population, the model's
    probability that a cell with that count carries the guide, at counts from
    1 to the guide's largest. A dashed line marks the model's threshold, and a
    dotted line marks the higher threshold that ``min_umi`` set for the calls.
    A panel whose status is not ``"fitted"`` names its status and calls no
    cell; a ``"no_positive"`` fit still draws its model.

    Without ``guides``, the plot gives an overview of every fit: the model's
    threshold against the number of cells with a UMI of the guide, one point
    per guide and sample on log scales of count + 1, and the number of fits of
    each status. The points are rasterized when there are more than 50,000.

    ``samples`` keeps the panels or points of those sample labels. When
    ``guides`` and ``samples`` select more than ``max_panels`` panels, the plot
    raises ValueError rather than draw a crowded figure, as it does when the
    fits hold no guide. A
    panel's axes take the guide as key, or ``(sample, guide)`` when the fits
    have a sample column, and its posterior axes take ``(key, "posterior")``.
    The tables ``"thresholds"`` and ``"histograms"`` hold the selected rows of
    the record and, with ``guides``, ``"posterior"`` holds the posterior at
    each drawn count: every count when the guide's largest is at most 1,000,
    and otherwise 1,000 counts spread evenly on the log scale.
    """
    max_panels = integer_argument(max_panels, "max_panels", minimum=1)
    requested = _distinct_values(guides, "guides", "guide ID")
    kept = _distinct_values(samples, "samples", "sample label")
    loaded = _load(store, "load_guide_thresholds", thresholds, guides=requested)
    table = loaded.table
    histograms = loaded.histograms
    stored = list(dict.fromkeys(table["sample"].tolist()))
    sample_order = stored if kept is None else kept
    if kept is not None:
        _require_known(kept, stored, holder="The fits", noun="sample")
        table = table[table["sample"].isin(kept)]
        histograms = histograms[histograms["sample"].isin(kept)]
    if table.empty:
        raise ValueError("The fits hold no guide, so there is nothing to plot")
    if requested is None:
        selected = table.reset_index(drop=True)
    else:
        # One panel per guide and sample, the samples of a guide side by side.
        sample_rank = {sample: rank for rank, sample in enumerate(sample_order)}
        guide_rank = {
            guide: rank
            for rank, guide in enumerate(dict.fromkeys(table["guide"].tolist()))
        }
        order = np.lexsort(
            (
                [sample_rank[sample] for sample in table["sample"].tolist()],
                [guide_rank[guide] for guide in table["guide"].tolist()],
            )
        )
        selected = table.iloc[order].reset_index(drop=True)
        if len(selected) > max_panels:
            raise ValueError(
                f"guides and samples select {len(selected)} panels, more than "
                f"max_panels={max_panels}"
            )
    # The overview's points are rasterized beyond 50,000; the panels draw bars.
    rasterized = requested is None and len(selected) > DEFAULT_RASTERIZE_THRESHOLD
    require_matplotlib()
    legends: tuple[LegendSpec, ...]
    with theme_context(theme):
        if requested is None:
            figure, axes, owns = _threshold_overview(
                selected,
                rasterized=rasterized,
                target=target,
                figsize=figsize,
                theme=theme,
                show_legend=show_legend,
            )
            tables = {
                "thresholds": selected,
                "histograms": histograms.reset_index(drop=True),
            }
            legends = (LegendSpec(kind="categorical", label="Status"),)
        else:
            n_samples = len(sample_order)
            figure, axes, owns, panel_tables = _threshold_panels(
                loaded,
                selected,
                histograms,
                n_columns=min(
                    n_samples if 1 < n_samples <= 4 else 3, max(len(selected), 1)
                ),
                target=target,
                figsize=figsize,
                theme=theme,
                show_legend=show_legend,
            )
            tables = {"thresholds": selected, **panel_tables}
            legends = (
                LegendSpec(kind="categorical", label="Call"),
                LegendSpec(kind="line", label="Threshold"),
            )
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables=tables,
        legends=legends,
        scales=(),
        provenance=PlotProvenance(
            assay=thresholds.assay,
            cell_key=None,
            n_cells=int(selected.drop_duplicates("sample")["n_cells"].sum()),
            n_samples=None if loaded.sample_column is None else len(sample_order),
            renderer="matplotlib",
            notes=("guide_thresholds",),
            extras={
                "thresholds": thresholds.to_dict(),
                "method": loaded.method,
                "sample_column": loaded.sample_column,
                "min_umi": loaded.min_umi,
                "guides": requested,
                "samples": None if loaded.sample_column is None else sample_order,
                "max_panels": max_panels,
                "rasterized": rasterized,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _point_areas(n_cells: np.ndarray, most: int) -> np.ndarray:
    """Return point areas that grow linearly with ``n_cells``, largest at ``most``."""
    smallest, largest = _POINT_AREAS
    areas: np.ndarray = smallest + (largest - smallest) * n_cells / max(most, 1)
    return areas


def _axis_limits(values: np.ndarray) -> tuple[float, float, float]:
    """Return the limits of an axis of ``values`` and the position of minus infinity.

    The axis holds 0 and every finite value, with a margin of 5% of their
    range on each side; the range runs from -1 to 1 when no finite value
    differs from 0. Minus infinity, such as the fold change of a gene without
    counts, sits at the left edge, three margins left of the smallest value,
    and the axis reaches one margin past it when a value is minus infinity.
    """
    finite = values[np.isfinite(values)]
    low = float(finite.min(initial=0.0))
    high = float(finite.max(initial=0.0))
    if low == high:
        low, high = -1.0, 1.0
    margin = 0.05 * (high - low)
    edge = low - 3.0 * margin
    left = edge - margin if np.isneginf(values).any() else low - margin
    return left, high + margin, edge


def _row_figsize(
    n_rows: int, n_title: int, *, legend_inches: float
) -> tuple[float, float]:
    """Size a figure of ``n_rows`` labelled rows under ``n_title`` lines of title."""
    return (6.2 + legend_inches, max(3.0, 0.22 * n_rows + 1.0 + 0.25 * n_title))


def _draw_rows(
    ax: Any,
    values: np.ndarray,
    n_cells: np.ndarray,
    significant: np.ndarray,
    labels: list[str],
    *,
    foreground: str,
) -> None:
    """Draw one labelled row per perturbation, the first at the top.

    Each row's point is highlighted when it is significant, grows with its
    ``n_cells``, and marks minus infinity at the left edge.
    """
    positions = np.arange(len(values), dtype=np.float64)[::-1]
    colors = np.where(significant, _HIGHLIGHT_COLOR, _MUTED_COLOR)
    areas = _point_areas(n_cells, int(n_cells.max(initial=0)))
    left, right, edge = _axis_limits(values)
    minus_infinity = np.isneginf(values)
    x = np.where(minus_infinity, edge, values)
    ax.axvline(0.0, color=foreground, linewidth=0.8, zorder=1)
    for chosen, shape in (
        (~minus_infinity, "o"),
        (minus_infinity, _MINUS_INFINITY_MARKER),
    ):
        if chosen.any():
            ax.scatter(
                x[chosen],
                positions[chosen],
                s=areas[chosen],
                c=colors[chosen].tolist(),
                marker=shape,
                linewidths=0,
                zorder=3,
            )
    ax.set_xlim(left, right)
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.7, len(values) - 0.3)
    # Faint lines lead from each label to its point.
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=foreground, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)


def _row_legend(
    n_cells: np.ndarray,
    *,
    significant: np.ndarray | None,
    no_umi: np.ndarray | None,
    alpha: float,
    foreground: str,
) -> tuple[list[Any], list[str]]:
    """Return the legend handles and labels of the drawn rows.

    Filled points give the colours of the rows, unless ``significant`` is None
    because no row has a p-value. Hollow markers give the marker of the rows
    that ``no_umi`` marks, if any, and the point areas of the smallest, middle
    and largest perturbed cell counts.
    """
    handles: list[Any] = []
    labels: list[str] = []
    if significant is not None:
        for chosen, color, label in (
            (significant, _HIGHLIGHT_COLOR, f"Significant at FDR {alpha:g}"),
            (~significant, _MUTED_COLOR, "Not significant"),
        ):
            if chosen.any():
                handles.append(_legend_marker("o", color, foreground))
                labels.append(label)
    if no_umi is not None and no_umi.any():
        handles.append(_legend_marker(_MINUS_INFINITY_MARKER, None, foreground))
        labels.append(_NO_UMI_LABEL)
    ordered = np.sort(n_cells)
    counts = np.unique(ordered[[0, len(ordered) // 2, -1]])
    for count, area in zip(
        counts.tolist(), _point_areas(counts, int(ordered[-1])), strict=True
    ):
        handles.append(_legend_marker("o", None, foreground, size=float(np.sqrt(area))))
        labels.append(f"{count:,} perturbed cell" + ("" if count == 1 else "s"))
    return handles, labels


@close_figures_on_error
def efficiency(
    store: Any,
    efficiency: ArtifactRef,
    *,
    max_rows: int = 40,
    alpha: float = 0.05,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how strongly each perturbation changes the expression of its target gene.

    ``efficiency`` is a ``perturbation_efficiency`` artifact of
    ``ds.perturbation.efficiency``, as ``ds.perturbation.load_efficiency``
    reads it. Each tested target, or each tested guide when the efficiency was
    measured by guide, gets one row labelled with its name. A point marks the
    log2 fold change of the target gene's mean expression in the cells that
    carry the perturbation against control cells, so a knockdown falls left of
    the line at 0 and an activation right of it. A point is blue when the row
    is significant at the false discovery rate ``alpha``, its adjusted p-value
    at most ``alpha``, and grey otherwise, and its area grows with the cells
    that carry the perturbation. When no UMI of the gene remains
    in those cells, the fold change is minus infinity: a triangle at the left
    edge marks it, and the row's label says that no UMI remains.

    When more than ``max_rows`` rows were tested, the plot keeps the
    ``max_rows`` rows with the smallest adjusted p-values, ties broken by name.
    The rows run from the strongest knockdown at the top to the strongest
    activation at the bottom. Rows whose status is not ``"tested"`` are left
    out, and the title counts them by status; the plot raises ValueError when
    no row was tested. The table ``"efficiency"`` holds
    the drawn rows of the record, with every column, in drawing order.
    """
    max_rows = integer_argument(max_rows, "max_rows", minimum=1)
    alpha = _alpha_argument(alpha)
    loaded = _load(store, "load_efficiency", efficiency)
    table = loaded.table
    by = str(loaded.by)
    tested_status = EFFICIENCY_STATUSES[EFFICIENCY_TESTED]
    tested = table[table["status"] == tested_status]
    n_tested = len(tested)
    if n_tested > max_rows:
        ranked = tested.sort_values(["adjusted_p_value", by], na_position="last")
        tested = ranked.head(max_rows)
    drawn = tested.sort_values(
        ["log2_fold_change", by], na_position="last"
    ).reset_index(drop=True)
    statuses = table["status"].value_counts()
    title = [f"Perturbation efficiency by {by}"]
    if n_tested > max_rows:
        title.append(
            f"The {max_rows:,} of {n_tested:,} tested {by}s with the smallest "
            "adjusted p-values"
        )
    others = _status_counts(statuses, EFFICIENCY_STATUSES, leave_out=tested_status)
    if drawn.empty:
        raise ValueError(_untested_message(by, "fold change", others))
    if others:
        title.append("Not plotted: " + ", ".join(others))
    fold_changes = drawn["log2_fold_change"].to_numpy(dtype=np.float64)
    n_cells = drawn["n_cells"].to_numpy(dtype=np.int64)
    # A row is significant at the false discovery rate alpha, as a pair of
    # the association tests is.
    significant = drawn["adjusted_p_value"].to_numpy(dtype=np.float64) <= alpha
    no_umi = np.isneginf(fold_changes)
    labels = [
        f"{name} (no UMI remains)" if missing else str(name)
        for name, missing in zip(drawn[by].tolist(), no_umi.tolist(), strict=True)
    ]
    if figsize is None and target is None:
        figsize = _row_figsize(
            len(drawn), len(title), legend_inches=2.6 if show_legend else 0.0
        )
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=[_EFFICIENCY_PANEL], figsize=figsize
        )
        ax = axes[_EFFICIENCY_PANEL]
        foreground = foreground_color(theme)
        _draw_rows(
            ax, fold_changes, n_cells, significant, labels, foreground=foreground
        )
        ax.set_xlabel("Log2 fold change against control cells")
        ax.set_title("\n".join(title))
        handles, texts = _row_legend(
            n_cells,
            significant=significant,
            no_umi=no_umi,
            alpha=alpha,
            foreground=foreground,
        )
        _place_legend(figure, ax, handles, texts, owns=owns, show_legend=show_legend)
        apply_figure_chrome(figure, theme)
    controls = table["n_control_cells"].to_numpy(dtype=np.int64)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"efficiency": drawn},
        legends=(
            LegendSpec(kind="categorical", label="Adjusted p-value"),
            LegendSpec(kind="size", label="Perturbed cells"),
        ),
        scales=(),
        provenance=PlotProvenance(
            assay=efficiency.assay,
            cell_key=None,
            # The perturbed cells of the drawn rows and the control cells.
            n_cells=int(n_cells.sum()) + int(controls.max(initial=0)),
            renderer="matplotlib",
            notes=("perturbation_efficiency",),
            extras={
                "efficiency": efficiency.to_dict(),
                "by": by,
                "from_assay": loaded.from_assay,
                "max_rows": max_rows,
                "alpha": alpha,
                "n_tested": n_tested,
                "statuses": {
                    status: int(statuses.get(status, 0))
                    for status in EFFICIENCY_STATUSES
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


@close_figures_on_error
def edistance(
    store: Any,
    edistance: ArtifactRef,
    *,
    max_rows: int = 40,
    alpha: float = 0.05,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how far each perturbation moves its cells from the control cells.

    ``edistance`` is a ``perturbation_edistance`` artifact of
    ``ds.perturbation.edistance``, as ``ds.perturbation.load_edistance`` reads
    it. Each tested target, or each tested guide when the E-distances were
    measured by guide, gets one row labelled with its name. A point marks the
    E-distance between the cells that carry the perturbation and the control
    cells in the coordinates it was measured in, such as a PCA; it grows as the
    perturbation moves its cells away from the control cells. The E-distance
    is an unbiased estimate, so a perturbation whose cells look like the
    control cells can fall a little left of the line at 0. A point is blue when
    the row is significant at the false discovery rate ``alpha``, its adjusted
    p-value at most ``alpha``, and grey otherwise, and its area grows with the
    cells that carry the perturbation. When the E-distances
    were measured without permutation tests, no row has a p-value: every point
    is blue, and the title says so.

    When more than ``max_rows`` rows were tested, the plot keeps the
    ``max_rows`` rows with the largest E-distances, ties broken by name. The
    rows run from the largest E-distance at the top to the smallest at the
    bottom. Rows whose status is not ``"tested"`` are left out, and the title
    counts them by status; the plot raises ValueError when no row was tested. The table ``"edistance"`` holds the drawn rows of the
    record, with every column, in drawing order.
    """
    max_rows = integer_argument(max_rows, "max_rows", minimum=1)
    alpha = _alpha_argument(alpha)
    loaded = _load(store, "load_edistance", edistance)
    table = loaded.table
    by = str(loaded.by)
    tested_status = EDISTANCE_STATUSES[EDISTANCE_TESTED]
    tested = table[table["status"] == tested_status]
    n_tested = len(tested)
    # The rows run from the largest E-distance to the smallest, ties broken by
    # name, so the first max_rows of them have the largest E-distances.
    drawn = (
        tested.sort_values(
            ["edistance", by], ascending=[False, True], na_position="last"
        )
        .head(max_rows)
        .reset_index(drop=True)
    )
    distances = drawn["edistance"].to_numpy(dtype=np.float64)
    n_cells = drawn["n_cells"].to_numpy(dtype=np.int64)
    adjusted = drawn["adjusted_p_value"].to_numpy(dtype=np.float64)
    # Without permutation tests no row has a p-value, and every point is blue.
    permuted = bool(np.isfinite(adjusted).any())
    significant = (adjusted <= alpha) if permuted else np.ones(len(drawn), dtype=bool)
    statuses = table["status"].value_counts()
    title = [f"E-distance by {by}"]
    if n_tested > max_rows:
        title.append(
            f"The {max_rows:,} of {n_tested:,} tested {by}s with the largest "
            "E-distances"
        )
    if len(drawn) and not permuted:
        title.append("Without permutation tests")
    others = _status_counts(statuses, EDISTANCE_STATUSES, leave_out=tested_status)
    if drawn.empty:
        raise ValueError(_untested_message(by, "E-distance", others))
    if others:
        title.append("Not plotted: " + ", ".join(others))
    labels = [str(name) for name in drawn[by].tolist()]
    if figsize is None and target is None:
        figsize = _row_figsize(
            len(drawn), len(title), legend_inches=2.6 if show_legend else 0.0
        )
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=[_EDISTANCE_PANEL], figsize=figsize
        )
        ax = axes[_EDISTANCE_PANEL]
        foreground = foreground_color(theme)
        _draw_rows(ax, distances, n_cells, significant, labels, foreground=foreground)
        ax.set_xlabel("E-distance to control cells")
        ax.set_title("\n".join(title))
        # An E-distance is never minus infinity, and without permutation tests
        # no row has a p-value.
        handles, texts = _row_legend(
            n_cells,
            significant=significant if permuted else None,
            no_umi=None,
            alpha=alpha,
            foreground=foreground,
        )
        _place_legend(figure, ax, handles, texts, owns=owns, show_legend=show_legend)
        apply_figure_chrome(figure, theme)
    controls = table["n_control_cells"].to_numpy(dtype=np.int64)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"edistance": drawn},
        legends=(
            LegendSpec(kind="categorical", label="Adjusted p-value"),
            LegendSpec(kind="size", label="Perturbed cells"),
        ),
        scales=(),
        provenance=PlotProvenance(
            assay=edistance.assay,
            cell_key=None,
            # The perturbed cells of the drawn rows and the control cells.
            n_cells=int(n_cells.sum()) + int(controls.max(initial=0)),
            renderer="matplotlib",
            notes=("perturbation_edistance",),
            extras={
                "edistance": edistance.to_dict(),
                "by": by,
                "n_permutations": int(loaded.n_permutations),
                "random_seed": int(loaded.random_seed),
                "max_rows": max_rows,
                "alpha": alpha,
                "n_tested": n_tested,
                "statuses": {
                    status: int(statuses.get(status, 0))
                    for status in EDISTANCE_STATUSES
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _ko_label(perturbation_type: str) -> str:
    """Return the legend label of the knocked-out cells."""
    return f"{perturbation_type} (perturbed)"


def _pooled_targets(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one row per target, in record order, that pools its splits.

    The columns are ``target``, ``n_cells``, ``n_ko``, ``n_np``, ``ko_share``
    and ``fitted``, whether any split of the target was fitted.
    """
    pooled = (
        rows.assign(fitted=rows["status"] == "fitted")
        .groupby("target", sort=False)
        .agg(
            n_cells=("n_cells", "sum"),
            n_ko=("n_ko", "sum"),
            fitted=("fitted", "any"),
        )
        .reset_index()
    )
    pooled.insert(3, "n_np", pooled["n_cells"] - pooled["n_ko"])
    pooled.insert(4, "ko_share", pooled["n_ko"] / pooled["n_cells"])
    return pooled


def _most_cells(pooled: pd.DataFrame, max_targets: int) -> pd.DataFrame:
    """Keep the ``max_targets`` targets with the most cells, ties broken by name.

    They run from the largest KO share to the smallest, ties broken by name.
    """
    kept = pooled.sort_values(["n_cells", "target"], ascending=[False, True])
    return kept.head(max_targets).sort_values(
        ["ko_share", "target"], ascending=[False, True]
    )


def _unfitted_statuses(rows: pd.DataFrame, names: Sequence[str]) -> list[str]:
    """Count the targets of ``names`` that were fitted in no split, by status.

    A target whose splits have different statuses counts under each.
    """
    chosen = rows[rows["target"].isin(list(names))]
    fitted = chosen.loc[chosen["status"] == "fitted", "target"]
    unfitted = chosen[~chosen["target"].isin(fitted)]
    counts = unfitted.drop_duplicates(["target", "status"])["status"].value_counts()
    return _status_counts(counts, MIXSCAPE_STATUSES)


def _class_bars(
    rows: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    requested: list[str] | None,
    max_targets: int,
    perturbation_type: str,
    target: Any | None,
    figsize: tuple[float, float] | None,
    theme: str,
    show_legend: bool,
) -> tuple[Any, dict[Hashable, Any], bool, pd.DataFrame]:
    """Draw one bar per target with the shares of its cells called KO and NP.

    Returns the figure, the axes, whether the plot owns the figure, and the
    drawn targets.
    """
    from matplotlib.ticker import PercentFormatter

    if requested is None:
        chosen = _most_cells(pooled, max_targets)
    else:
        chosen = pooled.set_index("target").loc[requested].reset_index()
    drawn = chosen[["target", "n_cells", "n_ko", "n_np", "ko_share"]].reset_index(
        drop=True
    )
    title = ["Mixscape classes by target"]
    if requested is None and len(pooled) > max_targets:
        title.append(
            f"The {max_targets:,} of {len(pooled):,} targets with the most cells"
        )
    unfitted = _unfitted_statuses(rows, drawn["target"].tolist())
    if unfitted:
        title.append("Not fitted, so all NP: " + ", ".join(unfitted))
    n_cells = drawn["n_cells"].to_numpy(dtype=np.int64)
    ko_share = drawn["ko_share"].to_numpy(dtype=np.float64)
    np_share = 1.0 - ko_share
    positions = np.arange(len(drawn), dtype=np.float64)[::-1]
    labels = [
        f"{name} ({count:,} cell{'' if count == 1 else 's'})"
        for name, count in zip(drawn["target"].tolist(), n_cells.tolist(), strict=True)
    ]
    if figsize is None and target is None:
        figsize = _row_figsize(
            len(drawn), len(title), legend_inches=2.2 if show_legend else 0.0
        )
    figure, axes, owns = normalize_axes_target(
        target, panel_keys=[_CLASSES_PANEL], figsize=figsize
    )
    ax = axes[_CLASSES_PANEL]
    bars = (
        (
            ax.barh(
                positions,
                ko_share,
                height=_CLASS_BAR_HEIGHT,
                color=_HIGHLIGHT_COLOR,
                label=_ko_label(perturbation_type),
            ),
            ko_share,
        ),
        (
            ax.barh(
                positions,
                np_share,
                height=_CLASS_BAR_HEIGHT,
                left=ko_share,
                color=_MUTED_COLOR,
                label=_NP_LABEL,
            ),
            np_share,
        ),
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.6, max(len(drawn), 1) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of the target's cells")
    ax.set_title("\n".join(title))
    handles = [container for container, shares in bars if (shares > 0).any()]
    # Every bar spans the axes, so a legend in a caller's figure goes beside
    # them.
    _place_legend(
        figure,
        ax,
        handles,
        [container.get_label() for container in handles],
        owns=owns,
        show_legend=show_legend,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    apply_figure_chrome(figure, theme)
    return figure, axes, owns, drawn


def _normal_density(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Return the density of a normal distribution at ``values``."""
    gap = (values - mean) / sd
    density: np.ndarray = np.exp(-0.5 * gap * gap) / (sd * np.sqrt(2.0 * np.pi))
    return density


def _score_panel(
    ax: Any,
    name: str,
    cells: pd.DataFrame,
    fit: pd.Series,
    *,
    perturbation_type: str,
    split_column: str | None,
    foreground: str,
) -> None:
    """Draw the histogram of one target's scores and the normals of its fit."""
    from matplotlib.ticker import MaxNLocator

    scores = cells["score"].to_numpy(dtype=np.float64)
    ko = (cells["global_class"] == perturbation_type).to_numpy(dtype=bool)
    edges = np.histogram_bin_edges(scores, bins=_SCORE_BINS)
    widths = np.diff(edges)
    ko_counts = np.histogram(scores[ko], edges)[0]
    np_counts = np.histogram(scores[~ko], edges)[0]
    if ko_counts.any():
        ax.bar(
            edges[:-1],
            ko_counts,
            width=widths,
            align="edge",
            color=_HIGHLIGHT_COLOR,
            label=_ko_label(perturbation_type),
        )
    if np_counts.any():
        ax.bar(
            edges[:-1],
            np_counts,
            width=widths,
            bottom=ko_counts,
            align="edge",
            color=_MUTED_COLOR,
            label=_NP_LABEL,
        )
    # Each normal is scaled to the cells of its class, so that it covers as
    # many cells as their bars.
    components = (
        (
            float(fit["control_mean"]),
            float(fit["control_sd"]),
            int(np.count_nonzero(~ko)),
            foreground,
            _CONTROL_CURVE_LABEL,
        ),
        (
            float(fit["perturbed_mean"]),
            float(fit["perturbed_sd"]),
            int(np.count_nonzero(ko)),
            _REFERENCE_COLOR,
            _PERTURBED_CURVE_LABEL,
        ),
    )
    low = min(
        float(edges[0]), *(mean - _CURVE_REACH * sd for mean, sd, *_ in components)
    )
    high = max(
        float(edges[-1]), *(mean + _CURVE_REACH * sd for mean, sd, *_ in components)
    )
    values = np.linspace(low, high, _CURVE_POINTS)
    for mean, sd, n_class, color, label in components:
        ax.plot(
            values,
            n_class * widths[0] * _normal_density(values, mean, sd),
            color=color,
            linewidth=1.6,
            label=label,
        )
    # Bars count cells, so the ticks fall on whole numbers.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Perturbation score")
    ax.set_ylabel("Cells")
    ax.set_title(name if split_column is None else f"{name}, fit in {fit['split']}")


def _score_panels(
    loaded: Any,
    pooled: pd.DataFrame,
    *,
    requested: list[str] | None,
    max_targets: int,
    target: Any | None,
    figsize: tuple[float, float] | None,
    theme: str,
    show_legend: bool,
) -> tuple[Any, dict[Hashable, Any], bool, pd.DataFrame]:
    """Draw one panel per target with a fitted split.

    Returns the figure, the axes, whether the plot owns the figure, and the
    drawn cells.
    """
    rows = loaded.targets
    perturbation_type = str(loaded.perturbation_type)
    fitted = pooled[pooled["fitted"].to_numpy(dtype=bool)]
    if requested is None:
        candidates = pooled["target"].tolist()
        names = _most_cells(fitted, max_targets)["target"].tolist()
    else:
        candidates = requested
        fitted_names = set(fitted["target"].tolist())
        names = [name for name in requested if name in fitted_names]
    title = ["Mixscape scores by target"]
    if requested is None and len(fitted) > max_targets:
        title.append(
            f"The {max_targets:,} of {len(fitted):,} fitted targets with the most cells"
        )
    unfitted = _unfitted_statuses(rows, candidates)
    if unfitted:
        title.append("Not fitted, so not drawn: " + ", ".join(unfitted))
    if not names:
        raise ValueError(
            "None of the targets has a fitted split, so there are no scores to plot"
        )
    cells = loaded.cells
    owners = {f"{name} {perturbation_type}": name for name in names}
    owners.update({f"{name} NP": name for name in names})
    owner = cells["mixscape_class"].map(owners)
    chosen = owner.notna().to_numpy(dtype=bool) & np.isfinite(
        cells["score"].to_numpy(dtype=np.float64)
    )
    table = cells.loc[chosen, ["split", "score", "global_class", "posterior"]]
    table.insert(0, "target", owner.to_numpy()[chosen])
    rank = {name: position for position, name in enumerate(names)}
    table = table.iloc[np.argsort(table["target"].map(rank).to_numpy(), kind="stable")]
    # The first fitted split of each target, in record order.
    fits = rows[rows["status"] == "fitted"].drop_duplicates("target")
    fits = fits.set_index("target")
    n_columns = min(_SCORE_COLUMNS, len(names))
    if figsize is None and target is None:
        n_rows = -(-len(names) // n_columns)
        figsize = (
            3.3 * n_columns + (2.4 if show_legend else 0.0),
            2.7 * n_rows + 0.3 * len(title),
        )
    figure, axes, owns = normalize_axes_target(
        target, panel_keys=list(names), figsize=figsize, n_columns=n_columns
    )
    foreground = foreground_color(theme)
    by_target = table.groupby("target", sort=False)
    for name in names:
        _score_panel(
            axes[name],
            name,
            by_target.get_group(name),
            fits.loc[name],
            perturbation_type=perturbation_type,
            split_column=loaded.split_column,
            foreground=foreground,
        )
    handles, labels = _legend_entries(
        axes.values(),
        (
            _ko_label(perturbation_type),
            _NP_LABEL,
            _CONTROL_CURVE_LABEL,
            _PERTURBED_CURVE_LABEL,
        ),
    )
    first = axes[names[0]]
    _place_legend(figure, first, handles, labels, owns=owns, show_legend=show_legend)
    if owns:
        figure.suptitle("\n".join(title))
    else:
        # A caller's figure keeps its own title, so the plot's title goes
        # above the first panel's.
        first.set_title("\n".join([*title, first.get_title()]))
    apply_figure_chrome(figure, theme)
    return figure, axes, owns, table


@close_figures_on_error
def mixscape(
    store: Any,
    classes: ArtifactRef,
    *,
    panel: Literal["classes", "scores"] = "classes",
    targets: Sequence[str] | None = None,
    max_targets: int = 40,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the share of each target's cells that Mixscape calls knocked out, or their scores.

    ``classes`` is a ``mixscape_classes`` artifact of
    ``ds.perturbation.mixscape``, as ``ds.perturbation.load_mixscape`` reads
    it. Among the cells that carry one target, Mixscape calls each cell
    knocked out (KO, or the ``perturbation_type`` of the classes) or an
    escaper (NP, not perturbed).

    With ``panel="classes"``, each target gets one bar, labelled with its name
    and its cells in every split. The blue part of the bar is the share of its
    cells called KO and the grey part the share called NP. The bars run from
    the largest KO share at the top to the smallest, ties broken by name. A
    target fitted in no split, because it had too few DE genes or KO cells or
    its fit collapsed, has only NP cells, and the title counts such targets by
    status. The table ``"classes"`` holds one row per drawn target, with
    ``target``, ``n_cells``, ``n_ko``, ``n_np`` and ``ko_share``, in drawing
    order. Either panel raises ValueError when the classes hold no target.

    With ``panel="scores"``, each target with a fitted split gets a panel: a
    histogram of its cells' scores, their projections onto the target's
    perturbation direction, with the KO cells in blue below the NP cells in
    grey. Two curves show the normal distributions of the fit in the target's
    first fitted split: the control component, which keeps the mean and
    standard deviation of the control cells' scores, scaled to the NP cells of
    the histogram, and the perturbed component, scaled to its KO cells. When
    the classes were split, the histogram pools the target's fitted splits,
    and the panel's title names the split of the curves. A target fitted in no
    split has no scores and gets no panel; the title counts such targets by
    status, and the plot raises ValueError when no target is left to draw.
    The title stands above the panels, or above the first panel's own title
    when the plot draws into a caller's axes, which also holds the legend. A
    panel's axes take the target as key. The table ``"scores"``
    holds one row per drawn cell, indexed by cell ID, with ``target``,
    ``split``, ``score``, ``global_class`` and ``posterior``, in panel order.

    ``targets`` keeps those targets in that order. Otherwise the plot keeps
    the ``max_targets`` targets with the most cells, fitted targets for
    ``panel="scores"``, ties broken by name, in the order of the bars, and the
    title says when it leaves any out. When ``targets`` names more than
    ``max_targets`` targets, the plot raises ValueError rather than draw a
    crowded figure. A title counts a target whose splits have different
    statuses under each status.
    """
    if not isinstance(panel, str) or panel not in (_CLASSES_PANEL, _SCORES_PANEL):
        raise ValueError(f"panel must be 'classes' or 'scores', not {panel!r}")
    max_targets = integer_argument(max_targets, "max_targets", minimum=1)
    requested = _distinct_values(targets, "targets", "target")
    if requested is not None and len(requested) > max_targets:
        raise ValueError(
            f"targets names {len(requested)} targets, more than "
            f"max_targets={max_targets}"
        )
    loaded = _load(store, "load_mixscape", classes)
    rows = loaded.targets
    perturbation_type = str(loaded.perturbation_type)
    pooled = _pooled_targets(rows)
    if pooled.empty:
        raise ValueError(
            "The Mixscape classes hold no target, so there is nothing to plot"
        )
    _require_known(
        requested,
        set(pooled["target"].tolist()),
        holder="The Mixscape classes",
        noun="target",
    )
    statuses = rows["status"].value_counts()
    require_matplotlib()
    tables: dict[str, pd.DataFrame]
    legends: tuple[LegendSpec, ...]
    with theme_context(theme):
        if panel == _CLASSES_PANEL:
            figure, axes, owns, drawn = _class_bars(
                rows,
                pooled,
                requested=requested,
                max_targets=max_targets,
                perturbation_type=perturbation_type,
                target=target,
                figsize=figsize,
                theme=theme,
                show_legend=show_legend,
            )
            tables = {"classes": drawn}
            n_cells = int(drawn["n_cells"].sum())
            legends = (LegendSpec(kind="categorical", label="Class"),)
        else:
            figure, axes, owns, drawn = _score_panels(
                loaded,
                pooled,
                requested=requested,
                max_targets=max_targets,
                target=target,
                figsize=figsize,
                theme=theme,
                show_legend=show_legend,
            )
            tables = {"scores": drawn}
            n_cells = len(drawn)
            legends = (
                LegendSpec(kind="categorical", label="Class"),
                LegendSpec(kind="line", label="Fitted component"),
            )
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables=tables,
        legends=legends,
        scales=(),
        provenance=PlotProvenance(
            assay=classes.assay,
            cell_key=None,
            # The cells of the drawn targets, or the drawn cells.
            n_cells=n_cells,
            renderer="matplotlib",
            notes=("mixscape_classes",),
            extras={
                "classes": classes.to_dict(),
                "panel": panel,
                "perturbation_type": perturbation_type,
                "split_column": loaded.split_column,
                "targets": requested,
                "max_targets": max_targets,
                "statuses": {
                    status: int(statuses.get(status, 0)) for status in MIXSCAPE_STATUSES
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _tested_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return the pairs that passed QC and have a p-value."""
    tested = pairs["pass_qc"].to_numpy(dtype=bool) & ~np.isnan(
        pairs["p_value"].to_numpy(dtype=np.float64)
    )
    return pairs[tested]


def _p_value_heights(p_values: np.ndarray) -> np.ndarray:
    """Return -log10 of each p-value, and infinity for one below 1e-300."""
    heights = np.full(len(p_values), np.inf)
    drawn = p_values >= _SMALLEST_P_VALUE
    heights[drawn] = -np.log10(p_values[drawn])
    return heights


def _height_limits(high: float, edged: bool) -> tuple[float, float, float]:
    """Return the limits of a -log10 p-value axis and the height of its top edge.

    The axis holds 0 and ``high``, or 1 when ``high`` is not above 0, with a
    margin of 5% of that on each side. A point beyond the axis sits at its top
    edge, three margins above ``high``, and the axis then reaches one margin
    past the edge.
    """
    if not high > 0:
        high = 1.0
    margin = 0.05 * high
    edge = high + 3.0 * margin
    return -margin, (edge + margin) if edged else (high + margin), edge


def _legend_marker(
    shape: str, color: str | None, foreground: str, size: float = 7.0
) -> Any:
    """Return the legend handle of a marker ``size`` points wide, filled or hollow."""
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        marker=shape,
        linestyle="",
        markerfacecolor="none" if color is None else color,
        markeredgecolor=foreground if color is None else "none",
        markersize=size,
    )


def _legend_line(color: str, linestyle: Literal["-", "--"], linewidth: float) -> Any:
    """Return the legend handle of a line."""
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, linestyle=linestyle, linewidth=linewidth)


def _point_sizes(n_points: int) -> tuple[float, float]:
    """Return the areas of the dots and of the edge markers of a panel.

    The dots stay large enough to show the few pairs away from the crowd.
    """
    size = default_point_size(n_points, size_min=6.0, size_max=30.0)
    return size, max(1.5 * size, 20.0)


def _place_labels(
    figure: Any, ax: Any, labels: list[Any], *, avoid: Sequence[Any] = ()
) -> None:
    """Move each label, in order, to the first free slot beside its point.

    The slots lie beside the point, on the side the label starts on and then
    on the other, at the point's height, then one label height above and
    below it, then two, and so on up to eight. A label takes the first slot
    inside the axes that no earlier label overlaps and that keeps the label
    gap from every text of ``avoid``, and keeps its first slot when none is
    free. A thin line leads to the point from a label above or below it.
    """
    from matplotlib.text import Text

    # The layout settles, and the labels get their sizes, without rendering.
    figure.draw_without_rendering()
    scale = figure.dpi / 72.0
    frame = ax.get_window_extent()
    steps = [0] + [
        sign * step for step in range(1, _LABEL_STEPS + 1) for sign in (1, -1)
    ]
    gap = _LABEL_GAP * scale
    placed: list[tuple[float, float, float, float]] = [
        (box.x0 - gap, box.y0 - gap, box.x1 + gap, box.y1 + gap)
        for box in (Text.get_window_extent(text) for text in avoid)
    ]
    for label in labels:
        extent = Text.get_window_extent(label)
        width, height = extent.width, extent.height
        anchor_x, anchor_y = ax.transData.transform(label.xy)
        start = 1.0 if label.get_horizontalalignment() == "left" else -1.0
        slots = [
            (side * _LABEL_OFFSET, step * (height / scale + _LABEL_GAP))
            for step in steps
            for side in (start, -start)
        ]
        # In a slot, the label's left or right edge sits at the slot's offset,
        # which centres the label vertically.
        boxes = []
        for dx, dy in slots:
            left = anchor_x + dx * scale - (0.0 if dx > 0 else width)
            bottom = anchor_y + dy * scale - height / 2.0
            boxes.append((left, bottom, left + width, bottom + height))
        chosen = next(
            (
                index
                for index, (left, bottom, right, top) in enumerate(boxes)
                if frame.x0 <= left
                and right <= frame.x1
                and frame.y0 <= bottom
                and top <= frame.y1
                and not any(
                    left < other[2]
                    and other[0] < right
                    and bottom < other[3]
                    and other[1] < top
                    for other in placed
                )
            ),
            0,
        )
        placed.append(boxes[chosen])
        dx, dy = slots[chosen]
        label.xyann = (dx, dy)
        label.set_horizontalalignment("left" if dx > 0 else "right")
        label.arrow_patch.set_visible(dy != 0)


def _callouts(
    ax: Any,
    texts: list[str],
    x: np.ndarray,
    y: np.ndarray,
    *,
    middle: float,
    color: str,
    zorder: float,
) -> list[Any]:
    """Label each point beside it, on the side towards ``middle``, where there is room.

    A label's line to its point stays hidden until :func:`_place_labels` moves
    the label above or below the point.
    """
    labels = []
    for text, point_x, point_y in zip(texts, x.tolist(), y.tolist(), strict=True):
        rightward = point_x < middle
        label = ax.annotate(
            text,
            (point_x, point_y),
            xytext=(_LABEL_OFFSET if rightward else -_LABEL_OFFSET, 0.0),
            textcoords="offset points",
            ha="left" if rightward else "right",
            va="center",
            fontsize="small",
            color=color,
            zorder=zorder,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.6,
                "shrinkA": 0.0,
                "shrinkB": 2.0,
            },
        )
        label.arrow_patch.set_visible(False)
        labels.append(label)
    return labels


def _volcano_legend(
    significant: np.ndarray,
    at_top: np.ndarray,
    no_counts: np.ndarray,
    *,
    cut: bool,
    alpha: float,
    foreground: str,
) -> tuple[list[Any], list[str]]:
    """Return the legend handles and labels of the drawn pairs."""
    entries: list[tuple[Any, str]] = []
    if significant.any():
        entries.append(
            (
                _legend_marker("o", _HIGHLIGHT_COLOR, foreground),
                f"Significant at FDR {alpha:g}",
            )
        )
    if not significant.all():
        entries.append(
            (_legend_marker("o", _MUTED_COLOR, foreground), "Not significant")
        )
    if at_top.any():
        entries.append(
            (_legend_marker(_TOP_EDGE_MARKER, None, foreground), _TOP_EDGE_LABEL)
        )
    if no_counts.any():
        entries.append(
            (_legend_marker(_MINUS_INFINITY_MARKER, None, foreground), _NO_COUNTS_LABEL)
        )
    if cut:
        entries.append((_legend_line(_REFERENCE_COLOR, "--", 1.2), _CUT_LABEL))
    return [handle for handle, _ in entries], [label for _, label in entries]


@close_figures_on_error
def association_volcano(
    store: Any,
    tests: ArtifactRef,
    *,
    targets: Sequence[str] | None = None,
    label_top: int = 10,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how much and how significantly each perturbation changes each gene.

    ``tests`` is a ``perturbation_associations`` artifact of
    ``ds.perturbation.associations``, as ``ds.perturbation.load_associations``
    reads it. Each discovery pair of a target and a gene that passed QC gets
    a point at its log2 fold change, the gene's counts in the target's cells
    against the counts that its null fit predicts for them, fitted on the
    target's and the control cells, or on all cells in tests at high MOI,
    and at -log10 of its p-value. A point is blue when the pair is
    significant at the false discovery rate of the tests and grey otherwise.
    A dashed line marks the largest p-value of a significant pair, the cut of
    the Benjamini-Hochberg procedure, which runs over the pairs of every
    target. When the tests are one-sided, the label of the p-value axis names
    the side tested.

    A p-value below 1e-300, such as 0 when it underflows, is drawn as an
    upward triangle at the top edge. A fold change of minus infinity, when
    the target's cells hold no counts of the gene, is drawn as a triangle at
    the left edge. The ``label_top`` pairs with the smallest p-values, ties
    broken by the larger size of ``z`` and then by target and gene, are
    labelled ``target:feature_name`` beside their points, with the gene's
    name from the record's ``feature_name`` column, and ``label_top=0``
    labels none. A label that would overlap a label of a smaller p-value
    moves to the other side of its point, or above or below it with a thin
    line to it. The points are rasterized when there are more than 100,000.

    ``targets`` keeps the pairs of those targets, and the plot raises
    ValueError naming any target without discovery pairs. The plot also
    raises ValueError when none of the kept pairs passed QC. The title counts
    the significant pairs among the drawn pairs, and the pairs left out
    because they were not tested. The table ``"volcano"`` holds the drawn
    pairs with every column of the record, from the smallest p-value to the
    largest, ties broken as for the labels, so its first ``label_top`` rows
    are the labelled pairs.
    """
    label_top = integer_argument(label_top, "label_top", minimum=0)
    requested = _distinct_values(targets, "targets", "target")
    loaded = _load(store, "load_associations", tests)
    side = str(loaded.side)
    alpha = float(loaded.alpha)
    discovery = loaded.discovery
    # The procedure ran over the pairs of every target, so the cut comes from
    # every significant pair, drawn or not.
    every = _tested_pairs(discovery)
    called = every["p_value"].to_numpy(dtype=np.float64)[
        every["significant"].to_numpy(dtype=bool)
    ]
    cut = float(called.max()) if len(called) else None
    if requested is not None:
        _require_known(
            requested,
            set(discovery["target"].tolist()),
            holder="The discovery pairs",
            noun="target",
        )
        discovery = discovery[discovery["target"].isin(requested)]
    tested = _tested_pairs(discovery)
    tested = tested[~np.isnan(tested["log2_fold_change"].to_numpy(dtype=np.float64))]
    if tested.empty:
        scope = "" if requested is None else " of these targets"
        raise ValueError(
            f"None of the discovery pairs{scope} passed QC, so there are no "
            "p-values to plot"
        )
    n_untested = len(discovery) - len(tested)
    drawn = tested.sort_values(
        ["p_value", "z", "target", "feature"],
        key=lambda column: -column.abs() if column.name == "z" else column,
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    n_drawn = len(drawn)
    fold_changes = drawn["log2_fold_change"].to_numpy(dtype=np.float64)
    significant = drawn["significant"].to_numpy(dtype=bool)
    no_counts = np.isneginf(fold_changes)
    left, right, left_edge = _axis_limits(fold_changes)
    x = np.where(no_counts, left_edge, fold_changes)
    heights = _p_value_heights(drawn["p_value"].to_numpy(dtype=np.float64))
    at_top = ~np.isfinite(heights)
    # The axis also holds the cut, which a drawn pair need not reach.
    cut_height = None if cut is None else float(_p_value_heights(np.asarray([cut]))[0])
    reached = [float(heights[~at_top].max(initial=0.0))]
    if cut_height is not None and np.isfinite(cut_height):
        reached.append(cut_height)
    bottom, top, top_edge = _height_limits(
        max(reached),
        bool(at_top.any()) or (cut_height is not None and np.isinf(cut_height)),
    )
    y = np.where(at_top, top_edge, heights)
    shapes = np.where(
        no_counts, _MINUS_INFINITY_MARKER, np.where(at_top, _TOP_EDGE_MARKER, "o")
    )
    size, edge_size = _point_sizes(n_drawn)
    rasterized = n_drawn > _ASSOCIATION_RASTERIZE_THRESHOLD
    n_significant = int(np.count_nonzero(significant))
    title = [
        "Association tests of the discovery pairs",
        f"{n_significant:,} of {n_drawn:,} tested pair{'' if n_drawn == 1 else 's'} "
        f"significant at FDR {alpha:g}",
    ]
    if n_untested:
        title.append(
            f"Not plotted: {n_untested:,} untested pair{'' if n_untested == 1 else 's'}"
        )
    if figsize is None and target is None:
        figsize = (5.4 + (3.0 if show_legend else 0.0), 4.4 + 0.25 * len(title))
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=[_VOLCANO_PANEL], figsize=figsize
        )
        ax = axes[_VOLCANO_PANEL]
        foreground = foreground_color(theme)
        ax.axvline(0.0, color=foreground, linewidth=0.8, zorder=1)
        # The significant pairs are drawn over the others.
        for chosen_significance, color, order in (
            (False, _MUTED_COLOR, 2.0),
            (True, _HIGHLIGHT_COLOR, 3.0),
        ):
            for shape in ("o", _TOP_EDGE_MARKER, _MINUS_INFINITY_MARKER):
                chosen = (significant == chosen_significance) & (shapes == shape)
                if chosen.any():
                    ax.scatter(
                        x[chosen],
                        y[chosen],
                        s=size if shape == "o" else edge_size,
                        color=color,
                        marker=shape,
                        linewidths=0,
                        alpha=0.75,
                        zorder=order,
                        rasterized=rasterized,
                    )
        if cut_height is not None:
            ax.axhline(
                top_edge if np.isinf(cut_height) else cut_height,
                color=_REFERENCE_COLOR,
                linestyle="--",
                linewidth=1.2,
                zorder=3.5,
            )
        labelled = drawn.head(label_top)
        # Labels name genes by name, such as IFNGR1, rather than by ID.
        texts = [
            f"{target}:{feature}"
            for target, feature in zip(
                labelled["target"].tolist(),
                labelled["feature_name"].tolist(),
                strict=True,
            )
        ]
        n_labels = len(texts)
        labels = _callouts(
            ax,
            texts,
            x[:n_labels],
            y[:n_labels],
            middle=0.0,
            color=foreground,
            zorder=4,
        )
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        # At high MOI a target's cells are compared with all other cells.
        compared = "all other cells" if str(loaded.moi) == "high" else "control cells"
        ax.set_xlabel(f"Log2 fold change against {compared}")
        ax.set_ylabel(f"-log10 p-value{_SIDE_WORDS.get(side, '')}")
        ax.set_title("\n".join(title))
        handles, texts = _volcano_legend(
            significant,
            at_top & ~no_counts,
            no_counts,
            cut=cut_height is not None,
            alpha=alpha,
            foreground=foreground,
        )
        _place_legend(figure, ax, handles, texts, owns=owns, show_legend=show_legend)
        apply_figure_chrome(figure, theme)
        if len(labels) > 1:
            _place_labels(figure, ax, labels)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"volcano": drawn},
        legends=(
            LegendSpec(kind="categorical", label="Significance"),
            LegendSpec(kind="line", label=_CUT_LABEL),
        ),
        scales=(),
        provenance=PlotProvenance(
            assay=tests.assay,
            cell_key=None,
            # The cells of the drawn targets and the control cells; at high
            # MOI every pair's cells are all the cells.
            n_cells=int((drawn["n_cells"] + drawn["n_control_cells"]).max())
            if str(loaded.moi) == "high"
            else int(drawn.groupby("target", sort=False)["n_cells"].max().sum())
            + int(drawn["n_control_cells"].max()),
            renderer="matplotlib",
            notes=("perturbation_associations",),
            extras={
                "tests": tests.to_dict(),
                "side": side,
                "integration": str(loaded.integration),
                "moi": str(loaded.moi),
                "from_assay": str(loaded.from_assay),
                "alpha": alpha,
                "targets": requested,
                "label_top": label_top,
                "n_tested": n_drawn,
                "n_significant": n_significant,
                "largest_significant_p_value": cut,
                "rasterized": rasterized,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _set_rows(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the ``target``, ``feature`` and ``p_value`` of each set's pairs.

    The rows come set by set, each in its order, after a ``set`` column that
    names their set.
    """
    frames = list(sets.values())
    return pd.DataFrame(
        {
            "set": np.repeat(
                np.asarray(list(sets), dtype=object), [len(frame) for frame in frames]
            ),
            **{
                column: np.concatenate(
                    [np.zeros(0, dtype=object)]
                    + [frame[column].to_numpy(dtype=object) for frame in frames]
                )
                for column in ("target", "feature")
            },
            "p_value": np.concatenate(
                [np.zeros(0)]
                + [frame["p_value"].to_numpy(dtype=np.float64) for frame in frames]
            ),
        }
    )


def _qq_band(n: int) -> pd.DataFrame:
    """Return the 95% band of the i-th smallest of ``n`` uniform p-values.

    The i-th smallest follows Beta(i, n - i + 1), whose mean is i / (n + 1),
    so its 2.5% and 97.5% quantiles bound the band, at the ranks of
    :func:`_log_spread_integers`.
    """
    ranks = _log_spread_integers(n)
    first = ranks.astype(np.float64)
    second = n - first + 1.0
    tail = (1.0 - _BAND_PROBABILITY) / 2.0
    return pd.DataFrame(
        {
            "rank": ranks,
            "expected": first / (n + 1.0),
            "lower": betaincinv(first, second, tail),
            "upper": betaincinv(first, second, 1.0 - tail),
        }
    )


def _qq_points(calibration: pd.DataFrame, discovery: pd.DataFrame) -> pd.DataFrame:
    """Return the QQ points of the tested negative-control and discovery pairs.

    Each set runs from its smallest p-value to its largest, and ``expected``
    is the mean of the i-th smallest of the set's n uniform p-values,
    i / (n + 1).
    """
    sets = {
        "calibration": calibration.sort_values("p_value", kind="stable"),
        "discovery": discovery.sort_values("p_value", kind="stable"),
    }
    points = _set_rows(sets)
    points["expected"] = np.concatenate(
        [np.arange(1, len(frame) + 1) / (len(frame) + 1.0) for frame in sets.values()]
    )
    return points


def _qq_panel(
    ax: Any,
    points: pd.DataFrame,
    band: pd.DataFrame,
    *,
    side: str,
    foreground: str,
    show_legend: bool,
) -> bool:
    """Draw the QQ plot of the negative-control and discovery p-values.

    Returns whether the points are rasterized.
    """
    from matplotlib.patches import Patch

    sets = points["set"].to_numpy(dtype=object)
    x = -np.log10(points["expected"].to_numpy(dtype=np.float64))
    heights = _p_value_heights(points["p_value"].to_numpy(dtype=np.float64))
    calibration = sets == "calibration"
    band_x = -np.log10(band["expected"].to_numpy(dtype=np.float64))
    band_low = -np.log10(band["upper"].to_numpy(dtype=np.float64))
    band_high = -np.log10(band["lower"].to_numpy(dtype=np.float64))
    # The axis holds the identity line, the band and the negative controls,
    # so a discovery p-value above them, like a p-value below 1e-300, sits at
    # the top edge.
    high = max(
        float(x.max()),
        float(band_high.max()),
        float(heights[calibration & np.isfinite(heights)].max(initial=0.0)),
    )
    beyond = ~(heights <= high)
    bottom, top, edge = _height_limits(high, bool(beyond.any()))
    y = np.where(beyond, edge, heights)
    increasing = np.argsort(band_x)
    ax.fill_between(
        band_x[increasing],
        band_low[increasing],
        band_high[increasing],
        color=foreground,
        alpha=0.15,
        linewidth=0,
        zorder=1,
    )
    ax.axline((0.0, 0.0), slope=1.0, color=foreground, linewidth=0.8, zorder=1.5)
    size, edge_size = _point_sizes(len(points))
    rasterized = len(points) > _ASSOCIATION_RASTERIZE_THRESHOLD
    # The discovery p-values are drawn over the negative controls.
    for name, color, order in (
        ("calibration", _REFERENCE_COLOR, 2.0),
        ("discovery", _HIGHLIGHT_COLOR, 3.0),
    ):
        in_set = sets == name
        for shape, chosen in (
            ("o", in_set & ~beyond),
            (_TOP_EDGE_MARKER, in_set & beyond),
        ):
            if chosen.any():
                ax.scatter(
                    x[chosen],
                    y[chosen],
                    s=size if shape == "o" else edge_size,
                    color=color,
                    marker=shape,
                    linewidths=0,
                    alpha=0.75,
                    zorder=order,
                    rasterized=rasterized,
                )
    left, right, _ = _height_limits(float(x.max()), False)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Expected -log10 p-value")
    ax.set_ylabel(f"Observed -log10 p-value{_SIDE_WORDS.get(side, '')}")
    ax.set_title("Negative-control p-values")
    if show_legend:
        # Short labels keep the legend clear of the points.
        entries = [
            (
                _legend_marker("o", _REFERENCE_COLOR, foreground),
                "Negative controls",
            )
        ]
        if not calibration.all():
            entries.append(
                (_legend_marker("o", _HIGHLIGHT_COLOR, foreground), "Discovery pairs")
            )
        entries.append((_legend_line(foreground, "-", 0.8), "Null expectation"))
        entries.append(
            (
                Patch(facecolor=foreground, alpha=0.15, linewidth=0),
                f"{_BAND_PROBABILITY:.0%} null band",
            )
        )
        if beyond.any():
            entries.append(
                (
                    _legend_marker(_TOP_EDGE_MARKER, None, foreground),
                    _BEYOND_AXIS_LABEL,
                )
            )
        # Points rarely fall below the identity line far from the origin.
        ax.legend(
            [handle for handle, _ in entries],
            [label for _, label in entries],
            loc="lower right",
            frameon=False,
            fontsize="small",
        )
    return rasterized


def _fold_change_panel(
    ax: Any, calibration: pd.DataFrame, *, compared: str, foreground: str
) -> tuple[pd.DataFrame, float | None]:
    """Draw the histogram of the negative-control fold changes and their mean.

    ``compared`` names the cells that each group's fold change compares its
    cells with. Returns the histogram and the mean, None without a finite
    fold change.
    """
    from matplotlib.ticker import MaxNLocator

    fold_changes = calibration["log2_fold_change"].to_numpy(dtype=np.float64)
    finite = np.isfinite(fold_changes)
    values = fold_changes[finite]
    title = ["Negative-control fold changes"]
    mean: float | None = None
    counts = np.zeros(0, dtype=np.int64)
    edges = np.zeros(1)
    if len(values):
        counts, edges = np.histogram(values, bins=_FOLD_CHANGE_BINS)
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=_REFERENCE_COLOR,
            linewidth=0,
        )
        mean = float(values.mean())
        ax.axvline(mean, color=foreground, linestyle="--", linewidth=1.2)
        title.append(f"Dashed line at the mean, {mean:.3g}")
    n_infinite = int(np.count_nonzero(~finite))
    if n_infinite:
        title.append(f"Not drawn: {n_infinite:,} without a finite fold change")
    # Bars count pairs, so the ticks fall on whole numbers.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(f"Log2 fold change against {compared}")
    ax.set_ylabel("Negative-control pairs")
    ax.set_title("\n".join(title))
    histogram = pd.DataFrame(
        {"bin_start": edges[:-1], "bin_stop": edges[1:], "pairs": counts}
    )
    return histogram, mean


def _controls_panel(
    ax: Any,
    positive: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    n_positive_pairs: int,
    discoveries: str,
    side: str,
    foreground: str,
    show_legend: bool,
) -> tuple[pd.DataFrame, bool]:
    """Draw the p-values of the positive and negative controls as two strips.

    Returns the drawn controls and whether their points are rasterized.
    """
    controls = _set_rows({"positive": positive, "calibration": calibration})
    sets = controls["set"].to_numpy(dtype=object)
    positions = np.where(sets == "positive", 0.0, 1.0)
    random = np.random.default_rng(_STRIP_SEED)
    x = positions + random.uniform(-_STRIP_HALF_WIDTH, _STRIP_HALF_WIDTH, len(sets))
    heights = _p_value_heights(controls["p_value"].to_numpy(dtype=np.float64))
    at_top = ~np.isfinite(heights)
    bottom, top, edge = _height_limits(
        float(heights[~at_top].max(initial=0.0)), bool(at_top.any())
    )
    y = np.where(at_top, edge, heights)
    size, edge_size = _point_sizes(len(controls))
    rasterized = len(controls) > _ASSOCIATION_RASTERIZE_THRESHOLD
    for name, color in (
        ("positive", _HIGHLIGHT_COLOR),
        ("calibration", _REFERENCE_COLOR),
    ):
        in_set = sets == name
        for shape, chosen in (
            ("o", in_set & ~at_top),
            (_TOP_EDGE_MARKER, in_set & at_top),
        ):
            if chosen.any():
                ax.scatter(
                    x[chosen],
                    y[chosen],
                    s=size if shape == "o" else edge_size,
                    color=color,
                    marker=shape,
                    linewidths=0,
                    alpha=0.75,
                    zorder=2,
                    rasterized=rasterized,
                )
    if positive.empty:
        ax.text(
            0.0,
            0.5,
            "No positive controls"
            if n_positive_pairs == 0
            else "No positive control passed QC",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            color=foreground,
        )
    labels = [
        f"{name}\n{count:,} pair{'' if count == 1 else 's'}"
        for name, count in (
            ("Positive controls", len(positive)),
            ("Negative controls", len(calibration)),
        )
    ]
    ax.set_xticks([0.0, 1.0], labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(bottom, top)
    ax.set_ylabel(f"-log10 p-value{_SIDE_WORDS.get(side, '')}")
    ax.set_title("\n".join(["Positive and negative controls", discoveries]))
    if show_legend and at_top.any():
        # The negative controls rarely rise far above the bottom.
        ax.legend(
            [_legend_marker(_TOP_EDGE_MARKER, None, foreground)],
            [_TOP_EDGE_LABEL],
            loc="center right",
            frameon=False,
            fontsize="small",
        )
    return controls, rasterized


@close_figures_on_error
def association_calibration(
    store: Any,
    tests: ArtifactRef,
    *,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot whether the association tests keep their error rate, and find effects.

    ``tests`` is a ``perturbation_associations`` artifact of
    ``ds.perturbation.associations``, as ``ds.perturbation.load_associations``
    reads it. Its negative-control pairs test groups of non-targeting guides
    against the other control cells, or against all other cells in tests at
    high MOI, so no gene should change, and its positive-control pairs, such
    as each target with its own gene, should change. Three panels in a row,
    whose axes take the keys ``"qq"``, ``"fold_changes"`` and
    ``"controls"``, follow the calibration check of the SCEPTRE book.

    ``"qq"`` draws -log10 of the negative-control p-values against the
    quantiles of as many uniform p-values. Calibrated p-values follow the
    identity line and mostly stay within the shaded band, between the 2.5%
    and 97.5% quantiles of Beta(i, n - i + 1), which the i-th smallest of n
    uniform p-values follows. The discovery p-values are drawn in blue over
    them, against the quantiles of their own number, and rise above the line
    where perturbations change genes. The axis holds the band and the
    negative controls, so a discovery p-value above it, like any p-value
    below 1e-300, is drawn as an upward triangle at the top edge.

    ``"fold_changes"`` is a histogram of the negative-control log2 fold
    changes, with a dashed line at their mean, which lies near 0 when the
    tests are calibrated. ``"controls"`` draws -log10 of the positive-control
    and negative-control p-values as two strips, each point moved sideways at
    random with a fixed seed so that the points stay apart. Its title counts
    the negative-control pairs that the tests call significant, the false
    discoveries, with their grade. Without a tested positive-control pair,
    the panel says so. A p-value below 1e-300 is drawn at the top edge.

    When the tests are one-sided, the labels of the p-value axes name the
    side tested. A panel rasterizes its points when it draws more than
    100,000. The plot raises ValueError when no negative-control pair passed
    QC. The table ``"qq"`` holds the drawn p-values with ``set``,
    ``target``, ``feature``, ``p_value`` and ``expected``, the uniform
    quantile i / (n + 1); ``"band"`` holds the band's ``rank``, ``expected``,
    ``lower`` and ``upper`` p-values, at every rank of at most 1,000
    negative-control pairs and at 1,000 ranks spread on the log scale
    otherwise; ``"fold_changes"`` holds the histogram's ``bin_start``,
    ``bin_stop`` and ``pairs``; and ``"controls"`` holds the drawn controls
    with ``set``, ``target``, ``feature`` and ``p_value``.
    """
    loaded = _load(store, "load_associations", tests)
    side = str(loaded.side)
    alpha = float(loaded.alpha)
    calibration = _tested_pairs(loaded.calibration)
    if calibration.empty:
        raise ValueError(
            "None of the negative-control pairs passed QC, so there is no "
            "calibration to plot"
        )
    discovery = _tested_pairs(loaded.discovery)
    positive = _tested_pairs(loaded.positive)
    points = _qq_points(calibration, discovery)
    band = _qq_band(len(calibration))
    n_false = int(loaded.n_false_discoveries)
    grade = str(loaded.calibration_grade)
    discoveries = (
        f"{n_false:,} false discover{'y' if n_false == 1 else 'ies'} "
        f"at FDR {alpha:g} ({grade})"
    )
    if figsize is None and target is None:
        figsize = (13.0, 4.6)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=list(_CALIBRATION_PANELS), figsize=figsize, n_columns=3
        )
        foreground = foreground_color(theme)
        qq_rasterized = _qq_panel(
            axes["qq"],
            points,
            band,
            side=side,
            foreground=foreground,
            show_legend=show_legend,
        )
        fold_changes, mean = _fold_change_panel(
            axes["fold_changes"],
            calibration,
            # At high MOI a group's cells are compared with all other cells.
            compared=(
                "all other cells"
                if str(loaded.moi) == "high"
                else "the other control cells"
            ),
            foreground=foreground,
        )
        controls, controls_rasterized = _controls_panel(
            axes["controls"],
            positive,
            calibration,
            n_positive_pairs=len(loaded.positive),
            discoveries=discoveries,
            side=side,
            foreground=foreground,
            show_legend=show_legend,
        )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "qq": points,
            "band": band,
            "fold_changes": fold_changes,
            "controls": controls,
        },
        legends=(
            LegendSpec(kind="categorical", label="Pairs"),
            LegendSpec(kind="line", label="Expected under the null"),
        ),
        scales=(),
        provenance=PlotProvenance(
            assay=tests.assay,
            cell_key=None,
            # The cells that each negative-control pair splits into its
            # group's cells and the others: the control cells, or at high MOI
            # every labelled cell.
            n_cells=int(
                (calibration["n_cells"] + calibration["n_control_cells"]).max()
            ),
            renderer="matplotlib",
            notes=("perturbation_associations",),
            extras={
                "tests": tests.to_dict(),
                "side": side,
                "integration": str(loaded.integration),
                "moi": str(loaded.moi),
                "from_assay": str(loaded.from_assay),
                "alpha": alpha,
                "n_false_discoveries": n_false,
                "calibration_grade": grade,
                "mean_log2_fold_change": mean,
                "n_tested_pairs": {
                    "discovery": len(discovery),
                    "calibration": len(calibration),
                    "positive": len(positive),
                },
                # Whether either panel of points is rasterized.
                "rasterized": qq_rasterized or controls_rasterized,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _cluster_palette(codes: np.ndarray) -> dict[int, str]:
    """Colour the clusters in the order in which they first appear in ``codes``.

    ``codes`` holds the cluster of every clustered perturbation in leaf order,
    so a cluster keeps its colour whatever a plot draws.
    """
    first = pd.unique(codes[codes != _UNASSIGNED])
    return {
        int(cluster): _CLUSTER_COLORS[rank % len(_CLUSTER_COLORS)]
        for rank, cluster in enumerate(first.tolist())
    }


def _cluster_runs(codes: np.ndarray) -> pd.DataFrame:
    """Return the longest run of rows of each cluster, ties to the first run.

    The rows of ``codes`` run in drawing order. The columns are ``cluster``,
    ``row``, the middle of the run, and ``length``, one row per cluster in
    order of cluster.
    """
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(codes)]])
    runs = pd.DataFrame(
        {
            "cluster": codes[starts],
            "row": (starts + stops - 1) / 2.0,
            "length": stops - starts,
        }
    )
    runs = runs[runs["cluster"] != _UNASSIGNED]
    return (
        runs.sort_values(
            ["cluster", "length", "row"], ascending=[True, False, True], kind="stable"
        )
        .drop_duplicates("cluster")
        .reset_index(drop=True)
    )


def _number_clusters(figure: Any, bar: Any, runs: pd.DataFrame) -> np.ndarray:
    """Number the clusters beside the bar, longer runs first, without overlaps.

    Each number sits beside the middle of its cluster's longest run. Once the
    layout settles, the numbers are kept from the longest run to the
    shortest, ties broken by row, and a number that would overlap a kept
    number is left out. Returns whether each run's number is kept.
    """
    from matplotlib.text import Text

    rows = runs["row"].to_numpy(dtype=np.float64)
    texts = [str(cluster) for cluster in runs["cluster"].tolist()]
    bar.set_yticks(rows, texts)
    # The layout settles, and the numbers get their sizes, without rendering.
    figure.draw_without_rendering()
    height = max(
        Text.get_window_extent(number).height for number in bar.get_yticklabels()
    )
    middles = bar.transData.transform(np.column_stack([np.zeros(len(rows)), rows]))
    kept = np.zeros(len(rows), dtype=bool)
    for index in np.lexsort((rows, -runs["length"].to_numpy())):
        if not (np.abs(middles[kept, 1] - middles[index, 1]) < height).any():
            kept[index] = True
    bar.set_yticks(
        rows[kept], [text for text, keep in zip(texts, kept, strict=True) if keep]
    )
    return kept


def _square_heatmap(figure: Any, ax: Any) -> None:
    """Change the figure's width until the heatmap's box comes out square.

    The heatmap's height does not depend on the figure's width, and its width
    follows the figure's nearly one for one: the colorbar's pad grows with
    it. So each pass, after the layout settles, widens the figure by the
    box's shortfall, and stops within half a pixel or after three passes.
    """
    for _ in range(_SQUARE_PASSES):
        figure.draw_without_rendering()
        box = ax.get_window_extent()
        shortfall = box.height - box.width
        if abs(shortfall) < 0.5:
            return
        width, height = figure.get_size_inches()
        figure.set_size_inches(width + shortfall / figure.dpi, height)


def _cluster_figsize(
    panel: str, names: list[str] | None, *, n_title: int, show_legend: bool
) -> tuple[float, float]:
    """Return the size of a figure that holds the panel, its names and legend.

    ``names`` are the perturbations that a heatmap names, or None.
    """
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties

    title = 0.25 * n_title + 0.1
    if panel == _EMBEDDING_PANEL:
        return (_HEATMAP_INCHES, _HEATMAP_INCHES + title)
    side = _HEATMAP_INCHES
    names_inches = 0.0
    if names is not None:
        side = max(_HEATMAP_MIN_INCHES, _NAMED_ROW_INCHES * len(names))
        points = FontProperties(size=rcParams["ytick.labelsize"]).get_size_in_points()
        names_inches = 0.15 + 0.62 * points / 72.0 * max(map(len, names))
    # The bar and its numbers, and the colorbar with its label.
    width = side + 0.6 + names_inches + (1.0 if show_legend else 0.0)
    return (width, side + names_inches + title)


def _cluster_heatmap(
    ax: Any,
    loaded: Any,
    drawn: pd.DataFrame,
    colors: list[str],
    *,
    correlation: bool,
    named: bool,
) -> tuple[Any, Any, pd.DataFrame, tuple[float, float]]:
    """Draw the profiles' correlations or distances and the cluster bar.

    Returns the image, the bar's axes, the drawn matrix and its colour
    limits.
    """
    from matplotlib.colors import to_rgba

    n_drawn = len(drawn)
    positions = loaded.distances.index.get_indexer(drawn.index)
    matrix = loaded.distances.to_numpy(dtype=np.float64)[np.ix_(positions, positions)]
    limits: tuple[float, float]
    if correlation:
        np.subtract(1.0, matrix, out=matrix)
        limits = (-1.0, 1.0)
    else:
        # Distances run from 0 to the largest drawn, or to 1 when every drawn
        # distance is 0.
        limits = (0.0, float(matrix.max()) or 1.0)
    # A heatmap with more perturbations than pixels averages the values that
    # each pixel covers before colouring it, and a smaller one draws each cell
    # as a block. The image fills the axes, whose box the plot's own figure
    # makes square: a box of fixed aspect would leave the layout to place the
    # bar's numbers against a box it then moves.
    image = ax.imshow(
        matrix,
        cmap=_CORRELATION_CMAP if correlation else _DISTANCE_CMAP,
        vmin=limits[0],
        vmax=limits[1],
        aspect="auto",
        interpolation="antialiased",
        interpolation_stage="data",
        rasterized=True,
    )
    if named:
        ticks = np.arange(n_drawn)
        names = [str(name) for name in drawn.index.tolist()]
        ax.set_xticks(ticks, names, rotation=90)
        ax.set_yticks(ticks, names)
        ax.yaxis.tick_right()
        ax.tick_params(length=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    # The bar sits left of the heatmap, one row per drawn perturbation.
    bar = ax.inset_axes([-(_BAR_GAP + _BAR_WIDTH), 0.0, _BAR_WIDTH, 1.0])
    rgba = {color: to_rgba(color) for color in set(colors)}
    bar.imshow(
        np.asarray([rgba[color] for color in colors]).reshape(n_drawn, 1, 4),
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, 1.0, n_drawn - 0.5, -0.5),
        rasterized=True,
    )
    bar.set_xlim(0.0, 1.0)
    bar.set_ylim(n_drawn - 0.5, -0.5)
    bar.set_xticks([])
    bar.set_yticks([])
    bar.tick_params(axis="y", length=2.0, pad=1.0)
    for spine in (*ax.spines.values(), *bar.spines.values()):
        spine.set_visible(False)
    table = pd.DataFrame(
        matrix, index=drawn.index.copy(), columns=drawn.index.copy(), copy=False
    )
    return image, bar, table, limits


def _cluster_map(
    ax: Any,
    drawn: pd.DataFrame,
    colors: list[str],
    *,
    named: list[str] | None,
    show_legend: bool,
    theme: str,
) -> tuple[pd.DataFrame, list[Any], list[Any], bool]:
    """Draw each perturbation at its place in the layout, coloured by cluster.

    Returns the drawn clusters with their medians, the clusters' numbers, the
    labels of the named perturbations, and whether the points are
    rasterized.
    """
    from matplotlib import patheffects

    x = drawn["x"].to_numpy(dtype=np.float64)
    y = drawn["y"].to_numpy(dtype=np.float64)
    assigned = drawn["cluster"].to_numpy(dtype=np.int64) != _UNASSIGNED
    size, _ = _point_sizes(len(drawn))
    rasterized = len(drawn) > DEFAULT_RASTERIZE_THRESHOLD
    point_colors = np.asarray(colors, dtype=object)
    # The unassigned perturbations lie beneath the clusters.
    for chosen, order in ((~assigned, 2.0), (assigned, 3.0)):
        if chosen.any():
            ax.scatter(
                x[chosen],
                y[chosen],
                s=size,
                c=point_colors[chosen].tolist(),
                linewidths=0,
                zorder=order,
                rasterized=rasterized,
            )
    xlim, ylim = padded_square_limits(x, y)
    finish_embedding_axes(ax, xlim=xlim, ylim=ylim, frame="minimal")
    members = drawn[assigned].groupby("cluster", sort=True)
    medians = members[["x", "y"]].median()
    clusters = pd.DataFrame(
        {
            "cluster": medians.index.to_numpy(dtype=np.int64),
            "n_perturbations": members.size().to_numpy(dtype=np.int64),
            "x": medians["x"].to_numpy(dtype=np.float64),
            "y": medians["y"].to_numpy(dtype=np.float64),
        }
    )
    numbers: list[Any] = []
    if show_legend and len(clusters):
        # Each number starts at its cluster's median, as the on-data labels
        # of an embedding do, and moves aside when it overlaps another.
        text_color = "#f5f5f5" if theme == "dark" else "#222222"
        stroke = "#222222" if theme == "dark" else "#ffffff"
        centres = clusters[["x", "y"]].to_numpy(dtype=np.float64)
        numbers = [
            ax.text(
                centre_x,
                centre_y,
                str(cluster),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                path_effects=[patheffects.withStroke(linewidth=2.0, foreground=stroke)],
                zorder=5,
                in_layout=False,
            )
            for cluster, (centre_x, centre_y) in zip(
                clusters["cluster"].tolist(), centres.tolist(), strict=True
            )
        ]
        register_layout_labels(ax, numbers, centres)
    labels: list[Any] = []
    if named:
        foreground = foreground_color(theme)
        rows = drawn.index.get_indexer(named)
        ax.scatter(
            x[rows],
            y[rows],
            s=_RING_AREA * size,
            facecolors="none",
            edgecolors=foreground,
            linewidths=0.8,
            zorder=4,
            rasterized=rasterized,
        )
        labels = _callouts(
            ax,
            named,
            x[rows],
            y[rows],
            middle=0.5 * (xlim[0] + xlim[1]),
            color=foreground,
            zorder=6,
        )
    return clusters, numbers, labels, rasterized


@close_figures_on_error
def perturbation_clusters(
    store: Any,
    clusters: ArtifactRef,
    *,
    panel: Literal["heatmap", "embedding"] = "heatmap",
    perturbations: Sequence[str] | None = None,
    cluster_ids: Sequence[int] | None = None,
    label: Sequence[str] | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot clusters of perturbations as a heatmap of their profiles or as a map.

    ``clusters`` is a ``perturbation_clusters`` artifact of
    ``ds.perturbation.cluster``, as ``ds.perturbation.load_clusters`` reads
    it.

    With ``panel="heatmap"``, the perturbations run in the leaf order of
    average linkage, from the top down and from the left, and each cell holds
    the Pearson correlation of two profiles, on a diverging scale from -1 in
    blue through 0 in white to 1 in red. When the clusters were found by
    Euclidean distance, each cell holds the distance instead, from 0 in dark
    blue to the largest drawn distance in white. A bar on the left colours
    each perturbation by its cluster, grey when HDBSCAN left it unassigned,
    and numbers each cluster beside the middle of the longest run of its
    rows; a number that would overlap the number of a longer run is left
    out. The names of the perturbations run beside the rows and below the
    columns when the heatmap draws at most 50, and are left out of a larger
    heatmap, which is too dense to name; ``perturbations`` or
    ``cluster_ids`` then draw fewer. The matrix is drawn as one image, so
    that thousands of perturbations render quickly and save to a small file.
    The axes take the keys ``"heatmap"``, ``"cluster_bar"`` and
    ``"colorbar"``. The table ``"matrix"`` holds the drawn values with the
    perturbations in drawing order, and ``"clusters"`` holds each drawn
    cluster's ``n_perturbations``, the ``row`` of its number and whether it
    is ``numbered``.

    With ``panel="embedding"``, each perturbation is a point at its place in
    the clustering's two-dimensional layout, coloured by its cluster, with
    the unassigned perturbations in grey beneath the others. Each cluster is
    numbered at the median of its points, and a number that overlaps another
    moves aside by at most about its height. ``label`` names those
    perturbations: each gets a ring around its point and its name beside it,
    and a name that would overlap another name or a number moves to the other
    side, or above or below its point with a thin line to it. The points are
    rasterized when there are more than 50,000. The axes take the key
    ``"embedding"``. The table ``"clusters"`` holds each drawn cluster's
    ``n_perturbations`` and the median ``x`` and ``y`` of its points, where
    its number starts.

    A cluster has the same colour in both panels and whatever the plot
    draws. The clusters take eight colours, CyteArc's colourblind-safe
    palette without its black, its yellow and its two greyish colours, in the
    order in which they first appear in the leaf order; the ninth takes the
    first colour again, and its number tells it apart.

    ``perturbations`` keeps those perturbations, and ``cluster_ids`` keeps
    the perturbations of those clusters, -1 for the unassigned ones; they
    cannot be combined. The plot raises ValueError when either names a
    perturbation or a cluster that the clusters lack, and when ``label``
    names a perturbation that the plot does not draw or comes with
    ``panel="heatmap"``, and when the clusters hold no perturbation. The
    title counts the drawn perturbations, out of
    the clustered ones when some are left out, and the drawn clusters and
    unassigned perturbations. ``show_legend=False`` leaves out the colorbar
    and the cluster numbers. The table ``"perturbations"`` holds the drawn
    rows of the record's table, indexed by perturbation, in leaf order.
    """
    if not isinstance(panel, str) or panel not in (_HEATMAP_PANEL, _EMBEDDING_PANEL):
        raise ValueError(f"panel must be 'heatmap' or 'embedding', not {panel!r}")
    requested = _distinct_values(perturbations, "perturbations", "perturbation")
    chosen = _distinct_values(
        cluster_ids, "cluster_ids", "cluster number", numbers=True
    )
    if requested is not None and chosen is not None:
        raise ValueError("Pass perturbations or cluster_ids, not both")
    named = _distinct_values(label, "label", "perturbation")
    if named is not None and panel == _HEATMAP_PANEL:
        raise ValueError(
            "label names perturbations in the embedding; the heatmap names every "
            f"perturbation when it draws at most {_NAMED_PERTURBATIONS}"
        )
    loaded = _load(store, "load_clusters", clusters)
    table = loaded.table
    if table.empty:
        raise ValueError(
            "The clusters hold no perturbation, so there is nothing to plot"
        )
    # Every clustered perturbation, in leaf order.
    in_order = table.reindex(pd.Index(loaded.order, name=table.index.name))
    every_code = in_order["cluster"].to_numpy(dtype=np.int64)
    palette = _cluster_palette(every_code)
    for names in (requested, named):
        _require_known(
            names, set(in_order.index), holder="The clusters", noun="perturbation"
        )
    if chosen is not None:
        present = set(every_code.tolist())
        missing = [cluster for cluster in chosen if cluster not in present]
        if missing:
            largest = int(every_code.max())
            raise ValueError(
                f"No perturbation is in cluster{'s' if len(missing) > 1 else ''} "
                f"{_quoted(missing)}; "
                + (
                    f"the clusters run from 0 to {largest}, with -1 for "
                    "unassigned perturbations"
                    if largest >= 0
                    else "every perturbation is unassigned, in cluster -1"
                )
            )
        keep = np.isin(every_code, chosen)
    elif requested is not None:
        keep = in_order.index.isin(requested)
    else:
        keep = np.ones(len(in_order), dtype=bool)
    drawn = in_order[keep]
    hidden = [name for name in named or () if name not in drawn.index]
    if hidden:
        raise ValueError(
            f"label names perturbations that the plot does not draw: {_quoted(hidden)}"
        )
    n_drawn = len(drawn)
    n_clustered = len(in_order)
    codes = drawn["cluster"].to_numpy(dtype=np.int64)
    assigned = codes != _UNASSIGNED
    n_unassigned = int(np.count_nonzero(~assigned))
    drawn_clusters = sorted(set(codes[assigned].tolist()))
    n_clusters = len(drawn_clusters)
    unassigned = _DARK_UNASSIGNED_COLOR if theme == "dark" else _UNASSIGNED_COLOR
    colors = [palette.get(code, unassigned) for code in codes.tolist()]
    metric = str(loaded.metric)
    correlation = metric == "correlation"
    counts = f"{n_clusters:,} cluster{'' if n_clusters == 1 else 's'}"
    if n_unassigned:
        counts += f", {n_unassigned:,} unassigned in grey"
    kept = "" if n_drawn == n_clustered else f"{n_drawn:,} of "
    title = [
        (
            (
                "Correlation between perturbation profiles"
                if correlation
                else "Euclidean distance between perturbation profiles"
            )
            if panel == _HEATMAP_PANEL
            else "UMAP of perturbation profiles"
        ),
        f"{kept}{n_clustered:,} perturbation{'' if n_clustered == 1 else 's'}: "
        f"{counts}",
    ]
    scale_label = "Pearson correlation" if correlation else "Euclidean distance"
    category_scale = CategoricalScale(
        order=tuple(drawn_clusters),
        palette={cluster: palette[cluster] for cluster in drawn_clusters},
        missing_color=unassigned,
        missing_label=_UNASSIGNED_LABEL,
    )
    named_axes = n_drawn <= _NAMED_PERTURBATIONS
    tables: dict[str, pd.DataFrame]
    legends: tuple[LegendSpec, ...]
    scales: tuple[Any, ...]
    extras: dict[str, Any]
    require_matplotlib()
    with theme_context(theme):
        sized = figsize is None and target is None
        if sized:
            figsize = _cluster_figsize(
                panel,
                [str(name) for name in drawn.index.tolist()] if named_axes else None,
                n_title=len(title),
                show_legend=show_legend,
            )
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=[panel], figsize=figsize
        )
        ax = axes[panel]
        if panel == _HEATMAP_PANEL:
            image, bar, matrix, limits = _cluster_heatmap(
                ax,
                loaded,
                drawn,
                colors,
                correlation=correlation,
                named=named_axes,
            )
            axes["cluster_bar"] = bar
            ax.set_title("\n".join(title))
            if show_legend:
                colorbar = figure.colorbar(image, ax=ax, shrink=0.5, aspect=15)
                colorbar.set_label(scale_label)
                if correlation:
                    colorbar.set_ticks(np.linspace(-1.0, 1.0, 5))
                axes["colorbar"] = colorbar.ax
            apply_figure_chrome(figure, theme)
            runs = _cluster_runs(codes)
            numbered = (
                _number_clusters(figure, bar, runs)
                if show_legend and len(runs)
                else np.zeros(len(runs), dtype=bool)
            )
            if sized:
                _square_heatmap(figure, ax)
            sizes = pd.Series(codes[assigned]).value_counts()
            tables = {
                "perturbations": drawn,
                "matrix": matrix,
                "clusters": pd.DataFrame(
                    {
                        "cluster": runs["cluster"].to_numpy(dtype=np.int64),
                        "n_perturbations": sizes.reindex(runs["cluster"]).to_numpy(
                            dtype=np.int64
                        ),
                        "row": runs["row"].to_numpy(dtype=np.float64),
                        "numbered": numbered,
                    }
                ),
            }
            legends = (
                LegendSpec(
                    kind="colorbar",
                    label=scale_label,
                    extras={"vmin": limits[0], "vmax": limits[1]},
                ),
                LegendSpec(kind="categorical", label="Cluster"),
            )
            scales = (
                ColorScale(
                    cmap=_CORRELATION_CMAP if correlation else _DISTANCE_CMAP,
                    vmin=limits[0],
                    vmax=limits[1],
                    vcenter=0.0 if correlation else None,
                ),
                category_scale,
            )
            # The matrix and the cluster bar are always drawn as images.
            extras = {
                "named": named_axes,
                "color_limits": list(limits),
                "rasterized": True,
            }
        else:
            cluster_table, numbers, labels, rasterized = _cluster_map(
                ax,
                drawn,
                colors,
                named=named,
                show_legend=show_legend,
                theme=theme,
            )
            ax.set_title("\n".join(title))
            apply_figure_chrome(figure, theme)
            # The numbers settle first, and the names then keep clear of them.
            place_layout_labels(figure)
            if labels:
                _place_labels(figure, ax, labels, avoid=numbers)
            tables = {"perturbations": drawn, "clusters": cluster_table}
            legends = (LegendSpec(kind="categorical", label="Cluster"),)
            scales = (category_scale,)
            extras = {"rasterized": rasterized}
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables=tables,
        legends=legends,
        scales=scales,
        provenance=PlotProvenance(
            assay=clusters.assay,
            cell_key=None,
            # The cells of the drawn perturbations' profiles.
            n_cells=int(drawn["n_cells"].sum()),
            renderer="matplotlib",
            notes=("perturbation_clusters",),
            extras={
                "clusters": clusters.to_dict(),
                "panel": panel,
                "metric": metric,
                "min_cluster_size": int(loaded.min_cluster_size),
                "min_samples": int(loaded.min_samples),
                "selection": str(loaded.selection),
                "seed": int(loaded.seed),
                "from_assay": str(loaded.from_assay),
                "perturbations": requested,
                "cluster_ids": chosen,
                "label": named,
                "n_perturbations": n_drawn,
                "n_clustered": n_clustered,
                "n_clusters": n_clusters,
                "n_unassigned": n_unassigned,
                **extras,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
