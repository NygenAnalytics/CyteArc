"""Plots of differential-abundance tests: propeller and Milo."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..abundance.results import CategoryTestResult, NeighborhoodTestResult
from ..metadata.selection import CellField
from ..storage.refs import ArtifactRef
from ..utils.arrays import sort_categories
from ._contracts import CategoricalScale, ColorScale, PlotProvenance, SizeScale
from ._data import _resolve_layout
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._style import (
    apply_figure_chrome,
    category_label,
    continuous_norm,
    finish_embedding_axes,
    padded_square_limits,
    resolve_category_scale,
    scatter_edgecolor,
    theme_context,
)

_DIVERGING_CMAP = "RdBu_r"


def _abundance_loader(store: Any, name: str) -> Any:
    loader = getattr(getattr(store, "abundance", None), name, None)
    if not callable(loader):
        raise TypeError(f"store does not provide abundance.{name}")
    return loader


def _category_tests(
    store: Any, tests: ArtifactRef | CategoryTestResult
) -> CategoryTestResult:
    """Return a category test, loading a saved one through the store."""
    if isinstance(tests, CategoryTestResult):
        return tests
    if not isinstance(tests, ArtifactRef):
        raise TypeError("tests must be a CategoryTestResult or an ArtifactRef")
    loaded = _abundance_loader(store, "load_category_tests")(tests)
    if not isinstance(loaded, CategoryTestResult):
        raise TypeError("store.abundance.load_category_tests returned another type")
    return loaded


def _neighborhood_tests(
    store: Any, tests: ArtifactRef | NeighborhoodTestResult
) -> NeighborhoodTestResult:
    """Return a Milo test, loading a saved one through the store."""
    if isinstance(tests, NeighborhoodTestResult):
        return tests
    if not isinstance(tests, ArtifactRef):
        raise TypeError("tests must be a NeighborhoodTestResult or an ArtifactRef")
    loaded = _abundance_loader(store, "load_neighborhood_tests")(tests)
    if not isinstance(loaded, NeighborhoodTestResult):
        raise TypeError("store.abundance.load_neighborhood_tests returned another type")
    return loaded


def _artifact_record(ref: ArtifactRef | None) -> dict[str, Any] | None:
    return ref.to_dict() if ref is not None else None


def _check_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 < value <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return value


def _significant(table: pd.DataFrame, alpha: float) -> np.ndarray:
    """Tested neighbourhoods whose spatial FDR is below ``alpha``."""
    spatial_fdr = table["spatial_fdr"].to_numpy(dtype=np.float64)
    tested = table["tested"].to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        significant: np.ndarray = tested & (spatial_fdr < alpha)
    return significant


def _log_fc_limits(values: np.ndarray, color_scale: ColorScale) -> tuple[float, float]:
    """Color limits for log fold changes, symmetric about zero by default."""
    finite = np.abs(values[np.isfinite(values)])
    if color_scale.vmin is not None or color_scale.vmax is not None:
        low = color_scale.vmin
        high = color_scale.vmax
        bound = float(np.max(finite, initial=1.0))
        return (
            float(low) if low is not None else -bound,
            float(high) if high is not None else bound,
        )
    if finite.size == 0:
        return -1.0, 1.0
    if color_scale.quantiles is not None:
        bound = float(np.quantile(finite, color_scale.quantiles[1]))
    else:
        bound = float(finite.max())
    bound = bound if bound > 0 else 1.0
    return -bound, bound


def _van_der_corput(n: int) -> np.ndarray:
    """The first ``n`` terms of the base-2 van der Corput sequence."""
    values = np.zeros(n, dtype=np.float64)
    for index in range(n):
        number, denominator, value = index + 1, 1.0, 0.0
        while number:
            denominator *= 2.0
            number, remainder = divmod(number, 2)
            value += remainder / denominator
        values[index] = value
    return values


def _quasirandom_offsets(values: np.ndarray, width: float) -> np.ndarray:
    """Deterministic beeswarm offsets in the style of ``geom_quasirandom``.

    Points are spread by a van der Corput sequence taken in the order of
    their values, scaled by a Gaussian kernel density of the values, so
    dense ranges spread wide and sparse tails stay near the axis.
    """
    n = values.size
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    spread = float(np.std(values, ddof=1))
    quartiles = np.quantile(values, [0.25, 0.75])
    iqr = float(quartiles[1] - quartiles[0]) / 1.34
    scale = min(spread, iqr) if iqr > 0 else spread
    if scale <= 0:
        density = np.ones(n, dtype=np.float64)
    else:
        bandwidth = 0.9 * scale * n ** (-0.2)
        density = np.zeros(n, dtype=np.float64)
        for start in range(0, n, 2048):
            block = values[start : start + 2048]
            density[start : start + block.size] = np.exp(
                -0.5 * ((block[:, None] - values[None, :]) / bandwidth) ** 2
            ).sum(axis=1)
        density /= density.max()
    sequence = np.empty(n, dtype=np.float64)
    sequence[np.argsort(values, kind="stable")] = _van_der_corput(n)
    return (sequence - 0.5) * 2.0 * width * density


@close_figures_on_error
def category_proportions(
    store: Any,
    tests: ArtifactRef | CategoryTestResult,
    *,
    categories: Sequence[str] | None = None,
    transformed: bool = False,
    categorical_scale: CategoricalScale | None = None,
    point_size: float = 36.0,
    n_columns: int | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot per-sample category proportions by condition with their FDR.

    ``tests`` is a ``category_tests`` artifact from
    ``ds.abundance.test_categories``, or the returned result. Each panel shows
    one category: one point per sample, a bar at each condition's mean, and the
    test's Benjamini-Hochberg FDR in the title. ``transformed=True`` plots the
    logit or arcsine square root proportions that the test modelled.
    """
    result = _category_tests(store, tests)
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    available = list(result.fit.categories)
    selected = available if categories is None else [str(c) for c in categories]
    if not selected:
        raise ValueError("categories must be non-empty")
    unknown = [category for category in selected if category not in available]
    if unknown:
        raise KeyError("Unknown categories: " + ", ".join(unknown))
    if len(set(selected)) != len(selected):
        raise ValueError("categories cannot contain duplicates")
    raw = result.proportions()
    sample_groups = result.sample_groups()
    scale = resolve_category_scale(
        sample_groups.to_numpy(dtype=object),
        categorical_scale or CategoricalScale(order=result.groups),
    )
    group_order = list(scale.order or ())
    palette = scale.palette or {}
    long_frame = (
        raw.loc[selected]
        .rename_axis(index="category", columns="sample")
        .stack()
        .rename("proportion")
        .reset_index()
    )
    long_frame["transformed"] = (
        result.proportions(transformed=True)
        .loc[selected]
        .stack()
        .to_numpy(dtype=np.float64)
    )
    long_frame["group"] = sample_groups.loc[long_frame["sample"]].to_numpy()
    plotted = "transformed" if transformed else "proportion"
    y_label = (
        {"logit": "Logit proportion", "asin": "Arcsine sqrt proportion"}[
            result.fit.transform
        ]
        if transformed
        else "Proportion"
    )
    tests_table = result.table.loc[selected].rename_axis("category").reset_index()
    columns = n_columns if n_columns is not None else min(len(selected), 4)
    if figsize is None and target is None:
        n_rows = int(np.ceil(len(selected) / max(columns, 1)))
        figsize = (2.6 * min(len(selected), columns) + 1.2, 2.8 * n_rows)
    plt, mpl = require_matplotlib()
    edgecolor = scatter_edgecolor(theme)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=selected,
            figsize=figsize,
            n_columns=columns,
        )
        for category in selected:
            ax = axes[category]
            rows = long_frame[long_frame["category"] == category]
            for position, group in enumerate(group_order):
                values = rows.loc[rows["group"] == group, plotted].to_numpy(
                    dtype=np.float64
                )
                if values.size == 0:
                    continue
                jitter = (np.arange(values.size) - (values.size - 1) / 2) * 0.04
                ax.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    color=palette[group],
                    s=point_size,
                    edgecolors=edgecolor,
                    linewidths=0.3,
                    zorder=2,
                )
                mean = float(np.mean(values))
                ax.hlines(
                    mean,
                    position - 0.28,
                    position + 0.28,
                    color=palette[group],
                    linewidth=2.0,
                    zorder=1,
                )
            fdr = float(result.table.loc[category, "fdr"])
            ax.set_title(f"{category}\nFDR = {fdr:.2g}")
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(
                [category_label(scale, group) for group in group_order],
                rotation=45 if len(group_order) > 3 else 0,
                ha="right" if len(group_order) > 3 else "center",
            )
            ax.set_xlim(-0.6, len(group_order) - 0.4)
            ax.set_ylabel(y_label)
        if show_legend:
            handles = [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor=palette[group],
                    markeredgecolor=edgecolor,
                    markeredgewidth=0.3,
                    label=category_label(scale, group),
                )
                for group in group_order
            ]
            legend_kwargs = {
                "handles": handles,
                "title": result.condition_by,
                "frameon": False,
            }
            if owns:
                figure.legend(loc="outside right center", **legend_kwargs)
            else:
                axes[selected[-1]].legend(loc="best", **legend_kwargs)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"proportions": long_frame, "tests": tests_table},
        legends=(LegendSpec(kind="categorical", label=result.condition_by),),
        scales=(scale,),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=result.n_cells,
            n_samples=len(result.fit.samples),
            renderer="matplotlib",
            notes=("category_proportions", result.method),
            extras={
                "category_tests": _artifact_record(result.artifact),
                "sample_by": result.sample_by,
                "condition_by": result.condition_by,
                "groups": list(result.groups),
                "transform": result.fit.transform,
                "transformed": transformed,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


@close_figures_on_error
def neighborhood_beeswarm(
    store: Any,
    tests: ArtifactRef | NeighborhoodTestResult,
    *,
    group_by: ArtifactRef | CellField | None = None,
    alpha: float = 0.1,
    min_fraction: float | None = None,
    mixed_label: str = "Mixed",
    group_order: Sequence[Any] | None = None,
    color_scale: ColorScale | None = None,
    point_size: float = 14.0,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot neighbourhood log fold changes as a beeswarm per label.

    The plot follows miloR's ``plotDAbeeswarm``. ``tests`` is a
    ``neighborhood_tests`` artifact from ``ds.abundance.test_neighborhoods``,
    or the returned result. Each point is one tested neighbourhood placed at
    its log2 fold change, colored by it when ``spatial_fdr < alpha`` and grey
    otherwise. ``group_by`` names cell labels that
    ``ds.abundance.annotate_neighborhoods`` assigns to neighbourhoods by
    majority; ``min_fraction`` relabels neighbourhoods whose majority label
    holds a smaller fraction of their labelled cells as ``mixed_label``.
    Neighbourhoods without any labelled cell are left out.
    """
    alpha = _check_alpha(alpha)
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if min_fraction is not None:
        if group_by is None:
            raise ValueError("min_fraction requires group_by")
        if not 0.0 <= min_fraction <= 1.0:
            raise ValueError("min_fraction must be in [0, 1]")
    result = _neighborhood_tests(store, tests)
    table = result.table.copy()
    table.insert(0, "neighborhood", np.arange(len(table)))
    table["significant"] = _significant(result.table, alpha)
    n_unlabelled = 0
    if group_by is None:
        table["group"] = "all neighbourhoods"
    else:
        annotation = _abundance_loader(store, "annotate_neighborhoods")(
            result.neighborhoods, group_by
        )
        if len(annotation) != len(table):
            raise ValueError("The annotation does not match the tested neighbourhoods")
        labels = annotation["label"].to_numpy(dtype=object)
        fractions = annotation["fraction"].to_numpy(dtype=np.float64)
        if min_fraction is not None:
            labels = labels.copy()
            labels[pd.notna(labels) & (fractions < min_fraction)] = mixed_label
        table["group"] = labels
        table["label_fraction"] = fractions
        n_unlabelled = int(pd.isna(labels).sum())
    plotted = table[table["tested"].to_numpy(dtype=bool) & pd.notna(table["group"])]
    plotted = plotted.reset_index(drop=True)
    if plotted.empty:
        raise ValueError("No tested neighbourhoods to plot")
    observed = list(pd.unique(plotted["group"].to_numpy(dtype=object)))
    if group_order is not None:
        order = list(group_order)
        missing = [group for group in observed if group not in order]
        if missing:
            raise ValueError(
                "group_order is missing observed groups: "
                + ", ".join(map(str, missing[:10]))
            )
        order = [group for group in order if group in set(observed)]
    else:
        order = sort_categories([group for group in observed if group != mixed_label])
        if mixed_label in observed:
            order.append(mixed_label)
    color_scale = color_scale or ColorScale(cmap=_DIVERGING_CMAP, vcenter=0.0)
    color_scale = replace(color_scale, cmap=color_scale.cmap or _DIVERGING_CMAP)
    log_fc = plotted["log_fc"].to_numpy(dtype=np.float64)
    significant = plotted["significant"].to_numpy(dtype=bool)
    limits = _log_fc_limits(log_fc[significant], color_scale)
    position = {group: index for index, group in enumerate(order)}
    rows = plotted["group"].map(position).to_numpy(dtype=np.float64)
    offsets = np.zeros(len(plotted), dtype=np.float64)
    for index in range(len(order)):
        members = np.flatnonzero(rows == index)
        offsets[members] = _quasirandom_offsets(log_fc[members], width=0.4)
    # The first group sits at the top, as a reader scans the labels.
    plotted["y"] = (len(order) - 1 - rows) + offsets
    if figsize is None and target is None:
        figsize = (5.0, max(2.4, 0.55 * len(order) + 1.2))
    _, mpl = require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["neighborhood_beeswarm"],
            figsize=figsize,
        )
        ax = axes["neighborhood_beeswarm"]
        norm = continuous_norm(
            mpl,
            vmin=limits[0],
            vmax=limits[1],
            vcenter=0.0 if limits[0] < 0.0 < limits[1] else None,
            scale=color_scale.scale,
        )
        ax.axvline(0.0, color="#9e9e9e", linewidth=0.8, linestyle="--", zorder=0)
        y = plotted["y"].to_numpy(dtype=np.float64)
        ax.scatter(
            log_fc[~significant],
            y[~significant],
            color=color_scale.missing_color,
            s=point_size,
            edgecolors="none",
            zorder=1,
        )
        strongest = np.flatnonzero(significant)
        strongest = strongest[np.argsort(np.abs(log_fc[strongest]), kind="stable")]
        artist = ax.scatter(
            log_fc[strongest],
            y[strongest],
            c=log_fc[strongest],
            cmap=color_scale.cmap,
            norm=norm,
            s=point_size,
            edgecolors="none",
            zorder=2,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([str(group) for group in reversed(order)])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("Log2 fold change")
        if group_by is None:
            ax.set_ylabel("")
        if show_legend:
            colorbar = figure.colorbar(artist, ax=ax, shrink=0.8, pad=0.02)
            colorbar.set_label(f"Log2 fold change (spatial FDR < {alpha:g})")
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"neighborhoods": plotted, "tests": table},
        legends=(
            LegendSpec(
                kind="colorbar",
                label="Log2 fold change",
                extras={"vmin": limits[0], "vmax": limits[1], "alpha": alpha},
            ),
        ),
        scales=(color_scale,),
        provenance=PlotProvenance(
            assay=result.neighborhoods.assay,
            cell_key=None,
            n_cells=0,
            n_samples=len(result.design),
            renderer="matplotlib",
            notes=("neighborhood_beeswarm",),
            extras={
                "neighborhood_tests": _artifact_record(result.artifact),
                "neighborhoods": result.neighborhoods.to_dict(),
                "group_by": (
                    group_by.to_dict()
                    if isinstance(group_by, ArtifactRef)
                    else group_by
                ),
                "alpha": alpha,
                "min_fraction": min_fraction,
                "groups": list(order),
                "n_neighborhoods": len(plotted),
                "n_significant": int(significant.sum()),
                "n_untested": int((~table["tested"].to_numpy(dtype=bool)).sum()),
                "n_unlabelled": n_unlabelled,
                "contrast": dict(result.contrast),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


@close_figures_on_error
def neighborhood_graph(
    store: Any,
    tests: ArtifactRef | NeighborhoodTestResult,
    *,
    layout: ArtifactRef,
    alpha: float = 0.1,
    size_scale: SizeScale | None = None,
    color_scale: ColorScale | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot Milo neighbourhoods at their index cells on a stored embedding.

    The plot follows miloR's ``plotNhoodGraphDA`` without the overlap edges.
    ``tests`` is a ``neighborhood_tests`` artifact or the returned result, and
    ``layout`` a two-dimensional embedding artifact that covers every index
    cell. Marker area follows neighbourhood size; tested neighbourhoods with
    ``spatial_fdr < alpha`` are colored by log2 fold change, and the others
    are grey.
    """
    alpha = _check_alpha(alpha)
    if not isinstance(layout, ArtifactRef):
        raise TypeError("layout must be an embedding ArtifactRef")
    result = _neighborhood_tests(store, tests)
    coordinates, layout_cells, layout_selection = _resolve_layout(store, layout)
    table = result.table.copy()
    table.insert(0, "neighborhood", np.arange(len(table)))
    index_cells = table["index_cell"].to_numpy(dtype=np.int64)
    order = np.argsort(layout_cells, kind="stable")
    sorted_cells = layout_cells[order]
    found = np.searchsorted(sorted_cells, index_cells)
    clipped = np.minimum(found, max(sorted_cells.size - 1, 0))
    covered = (found < sorted_cells.size) & (sorted_cells[clipped] == index_cells)
    if not covered.all():
        raise ValueError(
            f"layout does not cover {int((~covered).sum())} neighbourhood index cells"
        )
    rows = order[clipped]
    table["x"] = coordinates[rows, 0]
    table["y"] = coordinates[rows, 1]
    table["significant"] = _significant(result.table, alpha)
    sizes = table["size"].to_numpy(dtype=np.float64)
    if sizes.size == 0:
        raise ValueError("The test has no neighbourhoods to plot")
    if size_scale is None:
        size_scale = SizeScale(
            vmin=float(sizes.min()),
            vmax=float(sizes.max()),
            size_min=6.0,
            size_max=140.0,
        )
    areas = size_scale.areas(sizes)
    color_scale = color_scale or ColorScale(cmap=_DIVERGING_CMAP, vcenter=0.0)
    color_scale = replace(color_scale, cmap=color_scale.cmap or _DIVERGING_CMAP)
    log_fc = table["log_fc"].to_numpy(dtype=np.float64)
    significant = table["significant"].to_numpy(dtype=bool)
    limits = _log_fc_limits(log_fc[significant], color_scale)
    x = table["x"].to_numpy(dtype=np.float64)
    y = table["y"].to_numpy(dtype=np.float64)
    xlim, ylim = padded_square_limits(x, y)
    if figsize is None and target is None:
        figsize = (5.4, 4.4)
    _, mpl = require_matplotlib()
    edgecolor = scatter_edgecolor(theme)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["neighborhood_graph"],
            figsize=figsize,
        )
        ax = axes["neighborhood_graph"]
        norm = continuous_norm(
            mpl,
            vmin=limits[0],
            vmax=limits[1],
            vcenter=0.0 if limits[0] < 0.0 < limits[1] else None,
            scale=color_scale.scale,
        )
        ax.scatter(
            x[~significant],
            y[~significant],
            s=areas[~significant],
            color=color_scale.missing_color,
            edgecolors=edgecolor,
            linewidths=0.2,
            alpha=0.8,
            zorder=1,
        )
        strongest = np.flatnonzero(significant)
        strongest = strongest[np.argsort(np.abs(log_fc[strongest]), kind="stable")]
        artist = ax.scatter(
            x[strongest],
            y[strongest],
            s=areas[strongest],
            c=log_fc[strongest],
            cmap=color_scale.cmap,
            norm=norm,
            edgecolors=edgecolor,
            linewidths=0.2,
            zorder=2,
        )
        finish_embedding_axes(ax, xlim=xlim, ylim=ylim, frame="minimal")
        legend_sizes = np.unique(
            np.round(np.quantile(sizes, [0.0, 0.5, 1.0])).astype(np.int64)
        )
        if show_legend:
            colorbar = figure.colorbar(artist, ax=ax, shrink=0.7, pad=0.02)
            colorbar.set_label(f"Log2 fold change (spatial FDR < {alpha:g})")
            handles = [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=float(np.sqrt(area)),
                    markerfacecolor=color_scale.missing_color,
                    markeredgecolor=edgecolor,
                    markeredgewidth=0.2,
                    label=str(value),
                )
                for value, area in zip(
                    legend_sizes, size_scale.areas(legend_sizes), strict=True
                )
            ]
            legend_kwargs = {"handles": handles, "title": "Cells", "frameon": False}
            if owns:
                figure.legend(loc="outside right upper", **legend_kwargs)
            else:
                ax.legend(loc="best", **legend_kwargs)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"neighborhoods": table},
        legends=(
            LegendSpec(
                kind="colorbar",
                label="Log2 fold change",
                extras={"vmin": limits[0], "vmax": limits[1], "alpha": alpha},
            ),
            LegendSpec(
                kind="size",
                label="Cells",
                extras={"values": legend_sizes.tolist()},
            ),
        ),
        scales=(color_scale, size_scale),
        provenance=PlotProvenance(
            assay=result.neighborhoods.assay,
            cell_key=None,
            n_cells=0,
            n_samples=len(result.design),
            renderer="matplotlib",
            notes=("neighborhood_graph",),
            extras={
                "neighborhood_tests": _artifact_record(result.artifact),
                "neighborhoods": result.neighborhoods.to_dict(),
                "layout": layout.to_dict(),
                "layout_selection": layout_selection.to_dict(),
                "alpha": alpha,
                "n_neighborhoods": len(table),
                "n_significant": int(significant.sum()),
                "contrast": dict(result.contrast),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
