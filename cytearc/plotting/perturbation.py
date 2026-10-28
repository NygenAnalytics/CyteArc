"""Figures of pooled CRISPR screens."""

from collections.abc import Callable, Hashable, Sequence
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from ..perturbation.efficiency import EFFICIENCY_STATUSES, TESTED
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
