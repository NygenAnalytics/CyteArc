"""Figures of pooled CRISPR screens."""

from collections.abc import Callable, Hashable, Sequence
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import betaincinv

from ..perturbation.edistance import (
    EDISTANCE_STATUSES,
    TESTED as EDISTANCE_TESTED,
)
from ..perturbation.efficiency import EFFICIENCY_STATUSES, TESTED
from ..perturbation.mixscape import MIXSCAPE_STATUSES
from ..perturbation.mixtures import (
    FIT_STATUSES,
    gaussian_posterior,
    poisson_gaussian_posterior,
)
from ..storage.refs import ArtifactRef
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._style import (
    DEFAULT_RASTERIZE_THRESHOLD,
    apply_figure_chrome,
    default_point_size,
    foreground_color,
    theme_context,
)

_PANELS = ("guides_per_cell", "cells_per_guide", "top_fraction")
# A library of at most this many guides gets one labelled bar per guide; a
# larger one is drawn as guides ranked by their cells.
_LABELLED_GUIDES = 40
_NEGATIVE_COLOR = "#bdbdbd"
_ONE_GUIDE_COLOR = "#2b6cb0"
_MULTIPLE_COLOR = "#d1495b"
_CALL_GROUPS = (
    ("No guide called", _NEGATIVE_COLOR),
    ("One guide called", _ONE_GUIDE_COLOR),
    ("Several guides called", _MULTIPLE_COLOR),
)

_OVERVIEW_PANELS = ("thresholds", "statuses")
_THRESHOLD_COLOR = "#d55e00"
_OTHER_STATUS_COLOR = "#9e9e9e"
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
# A panel draws the posterior at every count of a guide whose largest count is
# at most this, and at this many counts spread on the log scale otherwise.
_POSTERIOR_COUNTS = 1000
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
_SIGNIFICANT_COLOR = _ONE_GUIDE_COLOR
_NOT_SIGNIFICANT_COLOR = _OTHER_STATUS_COLOR
# A point's area, in points squared, grows linearly with the row's perturbed
# cells from the first area to the second, which the row with the most takes.
_POINT_AREAS = (16.0, 144.0)
# A fold change of minus infinity is drawn at the left edge with this marker.
_NO_UMI_MARKER = "<"
_NO_UMI_LABEL = "No UMI remains in the perturbed cells"

_EDISTANCE_PANEL = "edistance"

_CLASSES_PANEL = "classes"
_SCORES_PANEL = "scores"
_KO_COLOR = _SIGNIFICANT_COLOR
_NP_COLOR = _NOT_SIGNIFICANT_COLOR
_NP_LABEL = "NP (not perturbed)"
# A bar's share of the space between two neighbouring targets.
_CLASS_BAR_HEIGHT = 0.7
_CONTROL_CURVE_LABEL = "Fitted control component"
_PERTURBED_CURVE_LABEL = "Fitted perturbed component"
_PERTURBED_CURVE_COLOR = _THRESHOLD_COLOR
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
_TARGETING_COLOR = _SIGNIFICANT_COLOR
_NEGATIVE_CONTROL_COLOR = _THRESHOLD_COLOR
# The label of a p-value axis names the side of a one-sided test.
_SIDE_WORDS = {"left": " for a decrease", "right": " for an increase"}
# A volcano label sits this many points beside its point. A label that would
# overlap another steps up or down by its height and this gap, in points, at
# most this many times each way.
_LABEL_OFFSET = 5.0
_LABEL_GAP = 2.0
_LABEL_STEPS = 8
# A QQ band is drawn at every rank of at most this many p-values, and at this
# many ranks spread evenly on the log scale otherwise.
_BAND_RANKS = 1000
_BAND_PROBABILITY = 0.95
# The points of a strip of control p-values spread at random, with a fixed
# seed, over this distance on each side of the strip's position.
_STRIP_HALF_WIDTH = 0.25
_STRIP_SEED = 0
_FOLD_CHANGE_BINS = 30


def _guide_calls(store: Any, calls: ArtifactRef) -> Any:
    """Load saved guide calls through the store's narrow loader."""
    loader = getattr(getattr(store, "perturbation", None), "load_guide_calls", None)
    if not callable(loader):
        raise TypeError("store does not provide guide calls")
    return loader(calls)


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
    """
    if (
        isinstance(max_guides_per_cell, bool)
        or not isinstance(max_guides_per_cell, int)
        or max_guides_per_cell < 2
    ):
        raise ValueError("max_guides_per_cell must be an integer of at least 2")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    loaded = _guide_calls(store, calls)
    cells = loaded.cells
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
        bar_colors = [_NEGATIVE_COLOR, _ONE_GUIDE_COLOR] + [_MULTIPLE_COLOR] * (
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
                color=_ONE_GUIDE_COLOR,
                label="Only guide called",
            )
            ax.barh(
                positions,
                cells_per_guide["several_guide_cells"],
                left=cells_per_guide["one_guide_cells"],
                color=_MULTIPLE_COLOR,
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
                color=_ONE_GUIDE_COLOR,
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
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _guide_thresholds(
    store: Any, thresholds: ArtifactRef, guides: list[str] | None
) -> Any:
    """Load saved guide thresholds through the store's narrow loader."""
    loader = getattr(
        getattr(store, "perturbation", None), "load_guide_thresholds", None
    )
    if not callable(loader):
        raise TypeError("store does not provide guide thresholds")
    return loader(thresholds, guides=guides)


def _distinct_labels(
    values: Sequence[str] | None, name: str, noun: str
) -> list[str] | None:
    """Return the distinct labels of ``values`` in their order, or None."""
    if values is None:
        return None
    if (
        isinstance(values, str)
        or not isinstance(values, Sequence)
        or not all(isinstance(value, str) for value in values)
    ):
        raise TypeError(f"{name} must be a sequence of {noun}s or None")
    if not values:
        raise ValueError(f"{name} must name at least one {noun}")
    return list(dict.fromkeys(values))


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


def _posterior_counts(max_umi: int) -> np.ndarray:
    """Return the counts from 1 to ``max_umi`` at which a panel draws the posterior.

    Every count when ``max_umi`` is at most 1,000, and otherwise 1,000 counts
    spread evenly on the log scale and rounded, without repeats.
    """
    if max_umi <= _POSTERIOR_COUNTS:
        return np.arange(1, max_umi + 1, dtype=np.int64)
    spread = np.rint(np.geomspace(1.0, float(max_umi), _POSTERIOR_COUNTS))
    counts: np.ndarray = np.unique(spread.astype(np.int64))
    return counts


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
    entries: dict[str, Any] = {}
    for each in [*axes.values(), *twins.values()]:
        for handle, label in zip(*each.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    labels = [label for label in _LEGEND_ORDER if label in entries]
    if show_legend and labels:
        handles = [entries[label] for label in labels]
        if owns:
            figure.legend(handles, labels, loc="outside right center", frameon=False)
        else:
            # The posterior axes lies over its panel, so the legend goes on top.
            first = twins.get(keys[0], axes[keys[0]])
            first.legend(handles, labels, loc="best", frameon=False, fontsize="small")
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
        (called, _ONE_GUIDE_COLOR, _CALLED_LABEL),
        (~called, _NEGATIVE_COLOR, _NOT_CALLED_LABEL),
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
            color=_THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=_THRESHOLD_LABEL,
        )
    if limit != threshold:
        ax.axvline(
            limit,
            color=_THRESHOLD_COLOR,
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
    posterior_counts = _posterior_counts(int(row["max_umi"]))
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
        (fitted, _ONE_GUIDE_COLOR, "fitted", 3.0),
        (~fitted, _OTHER_STATUS_COLOR, "other statuses", 2.0),
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
                rasterized=len(selected) > DEFAULT_RASTERIZE_THRESHOLD,
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
            _ONE_GUIDE_COLOR if status == "fitted" else _OTHER_STATUS_COLOR
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
    each status.

    ``samples`` keeps the panels or points of those sample labels. When
    ``guides`` and ``samples`` select more than ``max_panels`` panels, the plot
    raises ValueError rather than draw a crowded figure. A
    panel's axes take the guide as key, or ``(sample, guide)`` when the fits
    have a sample column, and its posterior axes take ``(key, "posterior")``.
    The tables ``"thresholds"`` and ``"histograms"`` hold the selected rows of
    the record and, with ``guides``, ``"posterior"`` holds the posterior at
    each drawn count: every count when the guide's largest is at most 1,000,
    and otherwise 1,000 counts spread evenly on the log scale.
    """
    if (
        isinstance(max_panels, bool)
        or not isinstance(max_panels, int)
        or max_panels < 1
    ):
        raise ValueError("max_panels must be a positive integer")
    requested = _distinct_labels(guides, "guides", "guide ID")
    kept = _distinct_labels(samples, "samples", "sample label")
    loaded = _guide_thresholds(store, thresholds, requested)
    table = loaded.table
    histograms = loaded.histograms
    stored = list(dict.fromkeys(table["sample"].tolist()))
    sample_order = stored if kept is None else kept
    if kept is not None:
        unknown = [label for label in kept if label not in stored]
        if unknown:
            raise ValueError(
                "The fits have no samples named "
                + ", ".join(repr(label) for label in unknown[:5])
            )
        table = table[table["sample"].isin(kept)]
        histograms = histograms[histograms["sample"].isin(kept)]
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
    require_matplotlib()
    legends: tuple[LegendSpec, ...]
    with theme_context(theme):
        if requested is None:
            figure, axes, owns = _threshold_overview(
                selected,
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
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _perturbation_efficiency(store: Any, efficiency: ArtifactRef) -> Any:
    """Load a saved perturbation efficiency table through the store's narrow loader."""
    loader = getattr(getattr(store, "perturbation", None), "load_efficiency", None)
    if not callable(loader):
        raise TypeError("store does not provide perturbation efficiency")
    return loader(efficiency)


def _point_areas(n_cells: np.ndarray, most: int) -> np.ndarray:
    """Return point areas that grow linearly with ``n_cells``, largest at ``most``."""
    smallest, largest = _POINT_AREAS
    areas: np.ndarray = smallest + (largest - smallest) * n_cells / max(most, 1)
    return areas


def _fold_change_limits(fold_changes: np.ndarray) -> tuple[float, float, float]:
    """Return the limits of the fold change axis and the position of minus infinity.

    The axis holds 0 and every finite fold change, with a margin of 5% of their
    range on each side; the range runs from -1 to 1 when no finite fold change
    differs from 0. Minus infinity sits at the left edge, three margins left of
    the smallest fold change, and the axis reaches one margin past it when a
    fold change is minus infinity.
    """
    finite = fold_changes[np.isfinite(fold_changes)]
    low = float(finite.min(initial=0.0))
    high = float(finite.max(initial=0.0))
    if low == high:
        low, high = -1.0, 1.0
    margin = 0.05 * (high - low)
    edge = low - 3.0 * margin
    left = edge - margin if np.isneginf(fold_changes).any() else low - margin
    return left, high + margin, edge


def _efficiency_legend(
    significant: np.ndarray,
    no_umi: np.ndarray,
    n_cells: np.ndarray,
    *,
    alpha: float,
    foreground: str,
) -> tuple[list[Any], list[str]]:
    """Return the legend handles and labels of the drawn rows.

    Filled points give the colours of the drawn rows. Hollow markers give the
    marker of minus infinity and the point areas of the smallest, middle and
    largest perturbed cell counts.
    """
    from matplotlib.lines import Line2D

    def marker(shape: str, color: str | None, size: float) -> Any:
        return Line2D(
            [],
            [],
            marker=shape,
            linestyle="",
            markerfacecolor="none" if color is None else color,
            markeredgecolor=foreground if color is None else "none",
            markersize=size,
        )

    handles: list[Any] = []
    labels: list[str] = []
    for chosen, color, label in (
        (significant, _SIGNIFICANT_COLOR, f"Adjusted p-value below {alpha:g}"),
        (
            ~significant,
            _NOT_SIGNIFICANT_COLOR,
            f"Adjusted p-value of {alpha:g} or more",
        ),
    ):
        if chosen.any():
            handles.append(marker("o", color, 7.0))
            labels.append(label)
    if no_umi.any():
        handles.append(marker(_NO_UMI_MARKER, None, 7.0))
        labels.append(_NO_UMI_LABEL)
    if len(n_cells):
        ordered = np.sort(n_cells)
        counts = np.unique(ordered[[0, len(ordered) // 2, -1]])
        for count, area in zip(
            counts.tolist(), _point_areas(counts, int(ordered[-1])), strict=True
        ):
            handles.append(marker("o", None, float(np.sqrt(area))))
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
    the line at 0 and an activation right of it. A point is blue when its
    adjusted p-value is below ``alpha`` and grey otherwise, and its area grows
    with the cells that carry the perturbation. When no UMI of the gene remains
    in those cells, the fold change is minus infinity: a triangle at the left
    edge marks it, and the row's label says that no UMI remains.

    When more than ``max_rows`` rows were tested, the plot keeps the
    ``max_rows`` rows with the smallest adjusted p-values, ties broken by name.
    The rows run from the strongest knockdown at the top to the strongest
    activation at the bottom. Rows whose status is not ``"tested"`` are left
    out, and the title counts them by status. The table ``"efficiency"`` holds
    the drawn rows of the record, with every column, in drawing order.
    """
    if (
        isinstance(max_rows, bool)
        or not isinstance(max_rows, int | np.integer)
        or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be a number above 0 and below 1")
    max_rows = int(max_rows)
    alpha = float(alpha)
    loaded = _perturbation_efficiency(store, efficiency)
    table = loaded.table
    by = str(loaded.by)
    tested_status = EFFICIENCY_STATUSES[TESTED]
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
    others = [
        f"{int(statuses[status]):,} {status}"
        for status in EFFICIENCY_STATUSES
        if status != tested_status and status in statuses.index
    ]
    if others:
        title.append("Not plotted: " + ", ".join(others))
    fold_changes = drawn["log2_fold_change"].to_numpy(dtype=np.float64)
    n_cells = drawn["n_cells"].to_numpy(dtype=np.int64)
    significant = drawn["adjusted_p_value"].to_numpy(dtype=np.float64) < alpha
    no_umi = np.isneginf(fold_changes)
    labels = [
        f"{name} (no UMI remains)" if missing else str(name)
        for name, missing in zip(drawn[by].tolist(), no_umi.tolist(), strict=True)
    ]
    positions = np.arange(len(drawn), dtype=np.float64)[::-1]
    colors = np.where(significant, _SIGNIFICANT_COLOR, _NOT_SIGNIFICANT_COLOR)
    areas = _point_areas(n_cells, int(n_cells.max(initial=0)))
    left, right, edge = _fold_change_limits(fold_changes)
    x = np.where(no_umi, edge, fold_changes)
    if figsize is None and target is None:
        figsize = (
            6.2 + (2.6 if show_legend else 0.0),
            max(3.0, 0.22 * len(drawn) + 1.0 + 0.25 * len(title)),
        )
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=[_EFFICIENCY_PANEL], figsize=figsize
        )
        ax = axes[_EFFICIENCY_PANEL]
        foreground = foreground_color(theme)
        ax.axvline(0.0, color=foreground, linewidth=0.8, zorder=1)
        for chosen, shape in ((~no_umi, "o"), (no_umi, _NO_UMI_MARKER)):
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
        ax.set_ylim(-0.7, max(len(drawn), 1) - 0.3)
        # Faint lines lead from each label to its point.
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", color=foreground, alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Log2 fold change against control cells")
        ax.set_title("\n".join(title))
        handles, texts = _efficiency_legend(
            significant, no_umi, n_cells, alpha=alpha, foreground=foreground
        )
        if show_legend and handles:
            if owns:
                # Centred beside the axes, so the multi-line title never runs
                # into it.
                figure.legend(handles, texts, loc="outside right center", frameon=False)
            else:
                ax.legend(handles, texts, loc="best", frameon=False, fontsize="small")
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


def _perturbation_edistance(store: Any, edistance: ArtifactRef) -> Any:
    """Load saved perturbation E-distances through the store's narrow loader."""
    loader = getattr(getattr(store, "perturbation", None), "load_edistance", None)
    if not callable(loader):
        raise TypeError("store does not provide perturbation E-distances")
    return loader(edistance)


def _edistance_limits(distances: np.ndarray) -> tuple[float, float]:
    """Return the limits of the E-distance axis.

    The axis holds 0 and every finite E-distance, with a margin of 5% of their
    range on each side; the range runs from -1 to 1 when no E-distance differs
    from 0.
    """
    finite = distances[np.isfinite(distances)]
    low = float(finite.min(initial=0.0))
    high = float(finite.max(initial=0.0))
    if low == high:
        low, high = -1.0, 1.0
    margin = 0.05 * (high - low)
    return low - margin, high + margin


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
    its adjusted p-value is below ``alpha`` and grey otherwise, and its area
    grows with the cells that carry the perturbation. When the E-distances
    were measured without permutation tests, no row has a p-value: every point
    is blue, and the title says so.

    When more than ``max_rows`` rows were tested, the plot keeps the
    ``max_rows`` rows with the largest E-distances, ties broken by name. The
    rows run from the largest E-distance at the top to the smallest at the
    bottom. Rows whose status is not ``"tested"`` are left out, and the title
    counts them by status. The table ``"edistance"`` holds the drawn rows of the
    record, with every column, in drawing order.
    """
    if (
        isinstance(max_rows, bool)
        or not isinstance(max_rows, int | np.integer)
        or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be a number above 0 and below 1")
    max_rows = int(max_rows)
    alpha = float(alpha)
    loaded = _perturbation_edistance(store, edistance)
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
    significant = (adjusted < alpha) if permuted else np.ones(len(drawn), dtype=bool)
    statuses = table["status"].value_counts()
    title = [f"E-distance by {by}"]
    if n_tested > max_rows:
        title.append(
            f"The {max_rows:,} of {n_tested:,} tested {by}s with the largest "
            "E-distances"
        )
    if len(drawn) and not permuted:
        title.append("Without permutation tests")
    others = [
        f"{int(statuses[status]):,} {status}"
        for status in EDISTANCE_STATUSES
        if status != tested_status and status in statuses.index
    ]
    if others:
        title.append("Not plotted: " + ", ".join(others))
    positions = np.arange(len(drawn), dtype=np.float64)[::-1]
    colors = np.where(significant, _SIGNIFICANT_COLOR, _NOT_SIGNIFICANT_COLOR)
    areas = _point_areas(n_cells, int(n_cells.max(initial=0)))
    left, right = _edistance_limits(distances)
    if figsize is None and target is None:
        figsize = (
            6.2 + (2.6 if show_legend else 0.0),
            max(3.0, 0.22 * len(drawn) + 1.0 + 0.25 * len(title)),
        )
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=[_EDISTANCE_PANEL], figsize=figsize
        )
        ax = axes[_EDISTANCE_PANEL]
        foreground = foreground_color(theme)
        ax.axvline(0.0, color=foreground, linewidth=0.8, zorder=1)
        if len(drawn):
            ax.scatter(
                distances,
                positions,
                s=areas,
                c=colors.tolist(),
                marker="o",
                linewidths=0,
                zorder=3,
            )
        ax.set_xlim(left, right)
        ax.set_yticks(positions, [str(name) for name in drawn[by].tolist()])
        ax.set_ylim(-0.7, max(len(drawn), 1) - 0.3)
        # Faint lines lead from each label to its point.
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", color=foreground, alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("E-distance to control cells")
        ax.set_title("\n".join(title))
        # The entries of the efficiency legend, without minus infinity, and
        # without the p-values when no row has one.
        handles, texts = _efficiency_legend(
            significant if permuted else significant[:0],
            np.zeros(len(drawn), dtype=bool),
            n_cells,
            alpha=alpha,
            foreground=foreground,
        )
        if show_legend and handles:
            if owns:
                # Centred beside the axes, so the multi-line title never runs
                # into it.
                figure.legend(handles, texts, loc="outside right center", frameon=False)
            else:
                ax.legend(handles, texts, loc="best", frameon=False, fontsize="small")
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


def _mixscape_classes(store: Any, classes: ArtifactRef) -> Any:
    """Load saved Mixscape classes through the store's narrow loader."""
    loader = getattr(getattr(store, "perturbation", None), "load_mixscape", None)
    if not callable(loader):
        raise TypeError("store does not provide Mixscape classes")
    return loader(classes)


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
    return [
        f"{int(counts[status]):,} {status}"
        for status in MIXSCAPE_STATUSES
        if status in counts.index
    ]


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
        figsize = (
            6.2 + (2.2 if show_legend else 0.0),
            max(3.0, 0.22 * len(drawn) + 1.0 + 0.25 * len(title)),
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
                color=_KO_COLOR,
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
                color=_NP_COLOR,
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
    if show_legend and handles:
        texts = [container.get_label() for container in handles]
        if owns:
            figure.legend(handles, texts, loc="outside right center", frameon=False)
        else:
            # Every bar spans the axes, so the legend goes beside them.
            ax.legend(
                handles,
                texts,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize="small",
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
            color=_KO_COLOR,
            label=_ko_label(perturbation_type),
        )
    if np_counts.any():
        ax.bar(
            edges[:-1],
            np_counts,
            width=widths,
            bottom=ko_counts,
            align="edge",
            color=_NP_COLOR,
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
            _PERTURBED_CURVE_COLOR,
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
    entries: dict[str, Any] = {}
    for ax in axes.values():
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    order = (
        _ko_label(perturbation_type),
        _NP_LABEL,
        _CONTROL_CURVE_LABEL,
        _PERTURBED_CURVE_LABEL,
    )
    labels = [label for label in order if label in entries]
    first = axes[names[0]]
    if show_legend:
        handles = [entries[label] for label in labels]
        if owns:
            figure.legend(handles, labels, loc="outside right center", frameon=False)
        else:
            first.legend(handles, labels, loc="best", frameon=False, fontsize="small")
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
    panel: str = "classes",
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
    order.

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
    if (
        isinstance(max_targets, bool)
        or not isinstance(max_targets, int | np.integer)
        or max_targets < 1
    ):
        raise ValueError("max_targets must be a positive integer")
    max_targets = int(max_targets)
    requested = _distinct_labels(targets, "targets", "target")
    if requested is not None and len(requested) > max_targets:
        raise ValueError(
            f"targets names {len(requested)} targets, more than "
            f"max_targets={max_targets}"
        )
    loaded = _mixscape_classes(store, classes)
    rows = loaded.targets
    perturbation_type = str(loaded.perturbation_type)
    pooled = _pooled_targets(rows)
    if requested is not None:
        known = set(pooled["target"].tolist())
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise ValueError(
                f"The Mixscape classes have no target named {unknown[0]!r}"
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


def _perturbation_associations(store: Any, tests: ArtifactRef) -> Any:
    """Load saved association tests through the store's narrow loader."""
    loader = getattr(getattr(store, "perturbation", None), "load_associations", None)
    if not callable(loader):
        raise TypeError("store does not provide perturbation associations")
    return loader(tests)


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


def _legend_marker(shape: str, color: str | None, foreground: str) -> Any:
    """Return the legend handle of a marker filled with ``color``, or hollow."""
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        marker=shape,
        linestyle="",
        markerfacecolor="none" if color is None else color,
        markeredgecolor=foreground if color is None else "none",
        markersize=7.0,
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


def _place_labels(figure: Any, ax: Any, labels: list[Any]) -> None:
    """Move each label, in order, to the first free slot beside its point.

    The slots lie beside the point, on the side the label starts on and then
    on the other, at the point's height, then one label height above and
    below it, then two, and so on up to eight. A label takes the first slot
    inside the axes that no earlier label overlaps, and keeps its first slot
    when none is free. A thin line leads to the point from a label above or
    below it.
    """
    from matplotlib.text import Text

    # The layout settles, and the labels get their sizes, without rendering.
    figure.draw_without_rendering()
    scale = figure.dpi / 72.0
    frame = ax.get_window_extent()
    steps = [0] + [
        sign * step for step in range(1, _LABEL_STEPS + 1) for sign in (1, -1)
    ]
    placed: list[tuple[float, float, float, float]] = []
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
                _legend_marker("o", _SIGNIFICANT_COLOR, foreground),
                f"Significant at FDR {alpha:g}",
            )
        )
    if not significant.all():
        entries.append(
            (_legend_marker("o", _NOT_SIGNIFICANT_COLOR, foreground), "Not significant")
        )
    if at_top.any():
        entries.append(
            (_legend_marker(_TOP_EDGE_MARKER, None, foreground), _TOP_EDGE_LABEL)
        )
    if no_counts.any():
        entries.append(
            (_legend_marker(_NO_UMI_MARKER, None, foreground), _NO_COUNTS_LABEL)
        )
    if cut:
        entries.append((_legend_line(_THRESHOLD_COLOR, "--", 1.2), _CUT_LABEL))
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
    against the counts that its fit on the control cells predicts for them,
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
    labelled ``target:feature`` beside their points, and ``label_top=0``
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
    if (
        isinstance(label_top, bool)
        or not isinstance(label_top, int | np.integer)
        or label_top < 0
    ):
        raise ValueError("label_top must be a non-negative integer")
    label_top = int(label_top)
    requested = _distinct_labels(targets, "targets", "target")
    loaded = _perturbation_associations(store, tests)
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
        known = set(discovery["target"].tolist())
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise ValueError(
                "The discovery pairs have no target named "
                + ", ".join(repr(name) for name in unknown)
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
    left, right, left_edge = _fold_change_limits(fold_changes)
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
        no_counts, _NO_UMI_MARKER, np.where(at_top, _TOP_EDGE_MARKER, "o")
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
            (False, _NOT_SIGNIFICANT_COLOR, 2.0),
            (True, _SIGNIFICANT_COLOR, 3.0),
        ):
            for shape in ("o", _TOP_EDGE_MARKER, _NO_UMI_MARKER):
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
                color=_THRESHOLD_COLOR,
                linestyle="--",
                linewidth=1.2,
                zorder=3.5,
            )
        targets_drawn = drawn["target"].tolist()
        # Labels name genes by name, such as IFNGR1, rather than by ID.
        features_drawn = drawn[
            "feature_name" if "feature_name" in drawn else "feature"
        ].tolist()
        labels = []
        for row in range(min(label_top, n_drawn)):
            # A label starts on the side towards the middle, where a volcano
            # has room.
            towards = 1.0 if x[row] < 0 else -1.0
            label = ax.annotate(
                f"{targets_drawn[row]}:{features_drawn[row]}",
                (float(x[row]), float(y[row])),
                xytext=(towards * _LABEL_OFFSET, 0.0),
                textcoords="offset points",
                ha="left" if towards > 0 else "right",
                va="center",
                fontsize="small",
                color=foreground,
                zorder=4,
                arrowprops={
                    "arrowstyle": "-",
                    "color": foreground,
                    "linewidth": 0.6,
                    "shrinkA": 0.0,
                    "shrinkB": 2.0,
                },
            )
            label.arrow_patch.set_visible(False)
            labels.append(label)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_xlabel("Log2 fold change against control cells")
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
        if show_legend:
            if owns:
                figure.legend(handles, texts, loc="outside right center", frameon=False)
            else:
                ax.legend(handles, texts, loc="best", frameon=False, fontsize="small")
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
            # The cells of the drawn targets and the control cells.
            n_cells=int(drawn.groupby("target", sort=False)["n_cells"].max().sum())
            + int(drawn["n_control_cells"].max()),
            renderer="matplotlib",
            notes=("perturbation_associations",),
            extras={
                "tests": tests.to_dict(),
                "side": side,
                "integration": str(loaded.integration),
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


def _band_ranks(n: int) -> np.ndarray:
    """Return the ranks at which the band of ``n`` uniform p-values is drawn.

    Every rank when ``n`` is at most 1,000, and otherwise 1,000 ranks spread
    evenly on the log scale and rounded, without repeats.
    """
    if n <= _BAND_RANKS:
        return np.arange(1, n + 1, dtype=np.int64)
    spread = np.rint(np.geomspace(1.0, float(n), _BAND_RANKS))
    ranks: np.ndarray = np.unique(spread.astype(np.int64))
    return ranks


def _qq_band(n: int) -> pd.DataFrame:
    """Return the 95% band of the i-th smallest of ``n`` uniform p-values.

    The i-th smallest follows Beta(i, n - i + 1), whose mean is i / (n + 1),
    so its 2.5% and 97.5% quantiles bound the band, at the ranks of
    :func:`_band_ranks`.
    """
    ranks = _band_ranks(n)
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
) -> None:
    """Draw the QQ plot of the negative-control and discovery p-values."""
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
        ("calibration", _NEGATIVE_CONTROL_COLOR, 2.0),
        ("discovery", _TARGETING_COLOR, 3.0),
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
                _legend_marker("o", _NEGATIVE_CONTROL_COLOR, foreground),
                "Negative controls",
            )
        ]
        if not calibration.all():
            entries.append(
                (_legend_marker("o", _TARGETING_COLOR, foreground), "Discovery pairs")
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


def _fold_change_panel(
    ax: Any, calibration: pd.DataFrame, *, foreground: str
) -> tuple[pd.DataFrame, float | None]:
    """Draw the histogram of the negative-control fold changes and their mean.

    Returns the histogram and the mean, None without a finite fold change.
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
            color=_NEGATIVE_CONTROL_COLOR,
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
    ax.set_xlabel("Log2 fold change against the other control cells")
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
) -> pd.DataFrame:
    """Draw the p-values of the positive and negative controls as two strips.

    Returns the drawn controls.
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
        ("positive", _TARGETING_COLOR),
        ("calibration", _NEGATIVE_CONTROL_COLOR),
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
    return controls


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
    against the other control cells, so no gene should change, and its
    positive-control pairs, such as each target with its own gene, should
    change. Three panels in a row, whose axes take the keys ``"qq"``,
    ``"fold_changes"`` and ``"controls"``, follow the calibration check of
    the SCEPTRE book.

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
    loaded = _perturbation_associations(store, tests)
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
        _qq_panel(
            axes["qq"],
            points,
            band,
            side=side,
            foreground=foreground,
            show_legend=show_legend,
        )
        fold_changes, mean = _fold_change_panel(
            axes["fold_changes"], calibration, foreground=foreground
        )
        controls = _controls_panel(
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
            # The control cells, which each negative-control pair splits into
            # its group's cells and the others.
            n_cells=int(
                (calibration["n_cells"] + calibration["n_control_cells"]).max()
            ),
            renderer="matplotlib",
            notes=("perturbation_associations",),
            extras={
                "tests": tests.to_dict(),
                "side": side,
                "integration": str(loaded.integration),
                "from_assay": str(loaded.from_assay),
                "alpha": alpha,
                "n_false_discoveries": n_false,
                "calibration_grade": grade,
                "mean_log2_fold_change": mean,
                "n_tested": {
                    "discovery": len(discovery),
                    "calibration": len(calibration),
                    "positive": len(positive),
                },
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
