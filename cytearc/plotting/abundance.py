"""Plots of saved differential-abundance tests.

``category_proportions`` shows the per-sample proportions behind a propeller
test (Phipson et al. 2022). ``neighborhood_beeswarm`` and
``neighborhood_graph`` show a Milo test (Dann et al. 2022) in the spirit of
miloR's ``plotDAbeeswarm`` and ``plotNhoodGraphDA``: neighbourhoods at
spatial FDR below ``alpha`` are coloured by their log2 fold change and the
others are grey. Saved results are read through the store's
``abundance`` loaders, so this module needs no datastore import.
"""

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
    DEFAULT_PANEL_INCHES,
    apply_figure_chrome,
    category_label,
    continuous_norm,
    finish_embedding_axes,
    padded_square_limits,
    resolve_category_scale,
    resolve_color_limits,
    scatter_edgecolor,
    theme_context,
)

_TRANSFORM_LABELS = {
    "logit": "Logit proportion",
    "asin": "Arcsine square-root proportion",
}
_LOG_FC_LABEL = "log2 fold change"
_DEFAULT_CMAP = "RdBu_r"
_UNGROUPED = "all neighbourhoods"
# Half the height of a beeswarm row, so neighbouring rows never touch.
_SWARM_WIDTH = 0.4
# Half the width of a condition's strip of samples.
_STRIP_WIDTH = 0.18
# Grid points of the density that scales beeswarm offsets.
_DENSITY_GRID = 256


def _van_der_corput(count: int) -> np.ndarray:
    """Return the first ``count`` points of the base-2 van der Corput sequence.

    The sequence starts at 1/2, so the first point sits at the centre.
    """
    values = np.zeros(count, dtype=np.float64)
    numbers = np.arange(1, count + 1, dtype=np.int64)
    scale = 0.5
    while np.any(numbers):
        values += (numbers & 1) * scale
        numbers >>= 1
        scale /= 2
    return values


def _quasirandom_offsets(values: np.ndarray, width: float) -> np.ndarray:
    """Spread points sideways in proportion to the density of their values.

    Points are taken in order of value and given evenly spread positions in
    [-1, 1) from the van der Corput sequence, then scaled by ``width`` and by
    the Gaussian kernel density at their value relative to its peak, so a
    swarm is widest where values are most common. Offsets are deterministic
    and lie within ``width``.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    offsets = np.zeros(data.size, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(data))
    if finite.size == 0:
        return offsets
    points = data[finite]
    spread = float(np.std(points))
    if finite.size < 2 or spread == 0:
        density = np.ones(finite.size)
    else:
        # Silverman's rule of thumb for the kernel bandwidth.
        bandwidth = 1.06 * spread * finite.size ** (-1 / 5)
        grid = np.linspace(points.min(), points.max(), _DENSITY_GRID)
        kernel = np.exp(-0.5 * ((grid[:, None] - points[None, :]) / bandwidth) ** 2)
        curve = kernel.sum(axis=1)
        density = np.interp(points, grid, curve / curve.max())
    rank = np.empty(finite.size, dtype=np.int64)
    rank[np.argsort(points, kind="stable")] = np.arange(finite.size)
    positions = 2 * _van_der_corput(finite.size)[rank] - 1
    offsets[finite] = float(width) * density * positions
    return offsets


def _category_tests(store: Any, tests: Any) -> CategoryTestResult:
    if isinstance(tests, CategoryTestResult):
        return tests
    if isinstance(tests, ArtifactRef):
        loader = getattr(getattr(store, "abundance", None), "load_category_tests", None)
        if not callable(loader):
            raise TypeError("store does not provide saved category tests")
        result = loader(tests)
        if not isinstance(result, CategoryTestResult):
            raise TypeError("store returned no CategoryTestResult")
        return result
    raise TypeError("tests must be an ArtifactRef or CategoryTestResult")


def _neighborhood_tests(store: Any, tests: Any) -> NeighborhoodTestResult:
    if isinstance(tests, NeighborhoodTestResult):
        return tests
    if isinstance(tests, ArtifactRef):
        loader = getattr(
            getattr(store, "abundance", None), "load_neighborhood_tests", None
        )
        if not callable(loader):
            raise TypeError("store does not provide saved neighbourhood tests")
        result = loader(tests)
        if not isinstance(result, NeighborhoodTestResult):
            raise TypeError("store returned no NeighborhoodTestResult")
        return result
    raise TypeError("tests must be an ArtifactRef or NeighborhoodTestResult")


def _check_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0 < value <= 1:
        raise ValueError("alpha must be in (0, 1]")
    return value


def _artifact_record(ref: Any) -> dict[str, Any] | None:
    return ref.to_dict() if isinstance(ref, ArtifactRef) else None


def _log_fc_scale(
    log_fc: np.ndarray, significant: np.ndarray, color_scale: ColorScale | None
) -> tuple[ColorScale, tuple[float, float]]:
    """Return the diverging scale of log fold changes and its limits.

    Without explicit limits the scale is symmetric about 0 over the
    significant neighbourhoods, or all of them when none is significant.
    """
    scale = color_scale or ColorScale()
    scale = replace(scale, cmap=scale.cmap or _DEFAULT_CMAP)
    shown = log_fc[significant] if significant.any() else log_fc
    if scale.vmin is not None or scale.vmax is not None:
        return scale, resolve_color_limits(shown, scale)
    _, high = resolve_color_limits(np.abs(shown), replace(scale, vcenter=None))
    limit = high if high > 0 else 1.0
    return scale, (-limit, limit)


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
    """Plot each sample's category proportions by condition, one panel per category.

    Each point is one sample and the bar is the condition's mean. The panel
    title gives the category's propeller FDR. ``transformed`` plots the
    logit or arcsine square-root proportions that the test fits.

    Raises:
        TypeError: If ``tests`` is not a propeller result or its reference.
        KeyError: If ``categories`` names a category the test does not have.
    """
    result = _category_tests(store, tests)
    fit = result.fit
    if categories is None:
        chosen = list(fit.categories)
    else:
        chosen = [str(category) for category in categories]
        unknown = [category for category in chosen if category not in fit.categories]
        if unknown:
            raise KeyError(f"Unknown categories: {unknown}")
        if not chosen:
            raise ValueError("categories must not be empty")
    groups = list(fit.groups)
    scale = categorical_scale or CategoricalScale()
    if scale.order is None:
        scale = replace(scale, order=tuple(groups))
    resolved = resolve_category_scale(np.asarray(groups, dtype=object), scale)
    order = list(resolved.order or ())
    palette = resolved.palette or {}

    proportions = result.proportions()
    values = result.proportions(transformed=True)
    sample_groups = result.sample_groups()
    rows = [
        {
            "category": category,
            "sample": sample,
            "group": sample_groups[sample],
            "proportion": proportions.loc[category, sample],
            "transformed": values.loc[category, sample],
        }
        for category in chosen
        for sample in fit.samples
    ]
    proportion_table = pd.DataFrame(
        rows, columns=["category", "sample", "group", "proportion", "transformed"]
    )
    test_table = result.table.loc[chosen].reset_index()
    column = "transformed" if transformed else "proportion"
    ylabel = _TRANSFORM_LABELS[fit.transform] if transformed else "Proportion"

    _, mpl = require_matplotlib()
    if figsize is None and target is None:
        columns = max(1, min(n_columns or 4, len(chosen)))
        panel_rows = int(np.ceil(len(chosen) / columns))
        width = max(1.6, 0.9 + 0.7 * len(order))
        figsize = (width * columns + 1.4, 3.0 * panel_rows)
    edgecolor = scatter_edgecolor(theme)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=list(chosen),
            figsize=figsize,
            n_columns=n_columns,
        )
        for category in chosen:
            ax = axes[category]
            subset = proportion_table[proportion_table["category"] == category]
            for position, group in enumerate(order):
                points = subset.loc[subset["group"] == group, column].to_numpy(
                    dtype=np.float64
                )
                if points.size == 0:
                    continue
                ax.scatter(
                    position + _quasirandom_offsets(points, _STRIP_WIDTH),
                    points,
                    s=point_size,
                    color=palette[group],
                    edgecolors=edgecolor,
                    linewidths=0.4,
                    zorder=2,
                )
                ax.hlines(
                    float(points.mean()),
                    position - _STRIP_WIDTH * 1.4,
                    position + _STRIP_WIDTH * 1.4,
                    colors=edgecolor,
                    linewidth=1.4,
                    zorder=3,
                )
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(
                [category_label(resolved, group) for group in order],
                rotation=30 if len(order) > 2 else 0,
                ha="right" if len(order) > 2 else "center",
            )
            ax.set_xlim(-0.6, len(order) - 0.4)
            if not transformed:
                ax.set_ylim(bottom=0)
            ax.set_ylabel(ylabel)
            fdr = float(result.table.loc[category, "fdr"])
            ax.set_title(f"{category}\nFDR = {fdr:.3g}")
        if show_legend:
            handles = [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor=palette[group],
                    markeredgecolor=edgecolor,
                    markersize=np.sqrt(point_size),
                    label=category_label(resolved, group),
                )
                for group in order
            ]
            if owns:
                figure.legend(
                    handles=handles,
                    title=result.condition_by,
                    loc="outside right center",
                    frameon=False,
                )
            else:
                axes[chosen[-1]].legend(
                    handles=handles, title=result.condition_by, frameon=False
                )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=dict(axes),
        tables={"proportions": proportion_table, "tests": test_table},
        legends=(LegendSpec(kind="categorical", label=result.condition_by),),
        scales=(resolved,),
        provenance=PlotProvenance(
            cell_key=None,
            n_cells=int(result.n_cells),
            n_samples=len(fit.samples),
            renderer="matplotlib",
            notes=("category_proportions", fit.method),
            extras={
                "category_tests": _artifact_record(result.artifact),
                "categories": list(chosen),
                "groups": groups,
                "sample_by": result.sample_by,
                "condition_by": result.condition_by,
                "transform": fit.transform,
                "transformed": bool(transformed),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _group_record(group_by: Any) -> Any:
    if isinstance(group_by, ArtifactRef):
        return group_by.to_dict()
    if isinstance(group_by, CellField):
        return group_by.key
    return None


def _resolve_group_order(
    observed: Sequence[Any], group_order: Sequence[Any] | None, mixed_label: str
) -> list[Any]:
    labelled = [group for group in observed if group != mixed_label]
    if group_order is None:
        order = sort_categories(labelled)
    else:
        order = [str(group) for group in group_order]
        if len(set(order)) != len(order):
            raise ValueError("group_order must not repeat a group")
        missing = sorted(set(labelled).difference(order), key=str)
        if missing:
            raise ValueError(f"group_order is missing observed groups: {missing}")
        order = [group for group in order if group in set(labelled)]
    if mixed_label in set(observed):
        order = [group for group in order if group != mixed_label] + [mixed_label]
    return list(order)


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
    """Plot tested neighbourhoods' log2 fold changes as a beeswarm per label.

    Each row holds the neighbourhoods whose most common ``group_by`` label
    it names, or every neighbourhood without ``group_by``. Neighbourhoods at
    spatial FDR below ``alpha`` are coloured by log fold change on a scale
    symmetric about 0, and the others are grey. With ``min_fraction``,
    neighbourhoods whose majority label covers less than that share of
    their labelled cells form a last ``mixed_label`` row.

    Raises:
        TypeError: If ``tests`` is not a Milo result or its reference.
        ValueError: If ``alpha`` is not in (0, 1], or ``min_fraction`` is
            set without ``group_by`` or outside [0, 1].
    """
    resolved_alpha = _check_alpha(alpha)
    if min_fraction is not None:
        if group_by is None:
            raise ValueError("min_fraction requires group_by")
        if not 0 <= float(min_fraction) <= 1:
            raise ValueError("min_fraction must be in [0, 1]")
    result = _neighborhood_tests(store, tests)
    table = result.table
    tested = table["tested"].to_numpy(dtype=bool)
    nhoods = np.flatnonzero(tested)
    log_fc = table["log_fc"].to_numpy(dtype=np.float64)[tested]
    spatial_fdr = table["spatial_fdr"].to_numpy(dtype=np.float64)[tested]
    with np.errstate(invalid="ignore"):
        significant = np.asarray(spatial_fdr < resolved_alpha, dtype=bool)

    if group_by is None:
        labels = np.full(nhoods.size, _UNGROUPED, dtype=object)
        fraction = np.full(nhoods.size, np.nan)
        groups_of = labels.copy()
    else:
        annotate = getattr(
            getattr(store, "abundance", None), "annotate_neighborhoods", None
        )
        if not callable(annotate):
            raise TypeError("store does not annotate neighbourhoods")
        annotation = annotate(result.neighborhoods, group_by)
        labels = annotation["label"].to_numpy(dtype=object)[tested]
        fraction = annotation["fraction"].to_numpy(dtype=np.float64)[tested]
        groups_of = np.asarray(
            [mixed_label if label is None else str(label) for label in labels],
            dtype=object,
        )
        if min_fraction is not None:
            with np.errstate(invalid="ignore"):
                mixed = ~(fraction >= float(min_fraction))
            groups_of[mixed] = mixed_label
    groups = _resolve_group_order(list(pd.unique(groups_of)), group_order, mixed_label)
    row_of = {group: len(groups) - 1 - index for index, group in enumerate(groups)}
    rows = np.asarray([row_of[group] for group in groups_of], dtype=np.float64)
    offsets = np.zeros(nhoods.size, dtype=np.float64)
    for group in groups:
        members = np.flatnonzero(groups_of == group)
        offsets[members] = _quasirandom_offsets(log_fc[members], _SWARM_WIDTH)
    y = rows + offsets
    scale, (vmin, vmax) = _log_fc_scale(log_fc, significant, color_scale)
    plotted = pd.DataFrame(
        {
            "neighborhood": nhoods,
            "group": groups_of,
            "label": labels,
            "label_fraction": fraction,
            "log_fc": log_fc,
            "spatial_fdr": spatial_fdr,
            "significant": significant,
            "y": y,
        }
    )

    _, mpl = require_matplotlib()
    if figsize is None and target is None:
        figsize = (5.4, max(2.4, 0.9 + 0.55 * len(groups)))
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["beeswarm"], figsize=figsize
        )
        ax = axes["beeswarm"]
        norm = continuous_norm(
            mpl, vmin=vmin, vmax=vmax, vcenter=None, scale=scale.scale
        )
        ax.axvline(0.0, color=scatter_edgecolor(theme), linewidth=0.8, zorder=0)
        ax.scatter(
            log_fc[~significant],
            y[~significant],
            s=point_size,
            color=scale.missing_color,
            edgecolors="none",
            zorder=1,
        )
        order = np.flatnonzero(significant)
        order = order[np.argsort(np.abs(log_fc[order]), kind="stable")]
        artist = ax.scatter(
            log_fc[order],
            y[order],
            c=log_fc[order],
            s=point_size,
            cmap=scale.cmap,
            norm=norm,
            edgecolors="none",
            zorder=2,
        )
        ax.set_yticks([row_of[group] for group in groups])
        ax.set_yticklabels([str(group) for group in groups])
        ax.set_ylim(-0.6, len(groups) - 0.4)
        ax.set_xlabel(_LOG_FC_LABEL)
        ax.set_title(
            f"{int(significant.sum())} of {nhoods.size} neighbourhoods at "
            f"spatial FDR < {resolved_alpha:g}"
        )
        if show_legend:
            colorbar = figure.colorbar(artist, ax=ax, shrink=0.8, pad=0.02)
            colorbar.set_label(_LOG_FC_LABEL)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=dict(axes),
        tables={"neighborhoods": plotted},
        legends=(
            LegendSpec(
                kind="colorbar",
                label=_LOG_FC_LABEL,
                extras={"vmin": vmin, "vmax": vmax, "cmap": scale.cmap},
            ),
        ),
        scales=(scale,),
        provenance=PlotProvenance(
            cell_key=None,
            n_samples=int(result.design.shape[0]),
            renderer="matplotlib",
            notes=("neighborhood_beeswarm",),
            extras={
                "neighborhood_tests": _artifact_record(result.artifact),
                "group_by": _group_record(group_by),
                "groups": list(groups),
                "alpha": resolved_alpha,
                "min_fraction": min_fraction,
                "mixed_label": mixed_label,
                "contrast": dict(result.contrast),
                "n_neighborhoods": int(nhoods.size),
                "n_significant": int(significant.sum()),
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

    Marker area follows neighbourhood size. Tested neighbourhoods at
    spatial FDR below ``alpha`` are coloured by log2 fold change on a scale
    symmetric about 0, and the others are grey. Overlaps between
    neighbourhoods are not drawn.

    Raises:
        TypeError: If ``tests`` is not a Milo result or its reference, or
            ``layout`` is not an embedding reference.
        ValueError: If ``alpha`` is not in (0, 1] or the layout does not
            place every index cell.
    """
    if not isinstance(layout, ArtifactRef):
        raise TypeError("layout must be an ArtifactRef of an embedding")
    resolved_alpha = _check_alpha(alpha)
    result = _neighborhood_tests(store, tests)
    table = result.table
    values, cell_idx, _ = _resolve_layout(store, layout)
    index_cells = table["index_cell"].to_numpy(dtype=np.int64)
    positions = np.searchsorted(cell_idx, index_cells)
    placed = positions < cell_idx.size
    placed[placed] = cell_idx[positions[placed]] == index_cells[placed]
    if not placed.all():
        raise ValueError(
            f"The layout does not place {int((~placed).sum())} index cells; use an "
            "embedding of the graph's cells"
        )
    x = values[positions, 0]
    y = values[positions, 1]
    sizes = table["size"].to_numpy(dtype=np.int64)
    log_fc = table["log_fc"].to_numpy(dtype=np.float64)
    tested = table["tested"].to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        significant = tested & (
            table["spatial_fdr"].to_numpy(dtype=np.float64) < resolved_alpha
        )
    resolved_sizes = size_scale or SizeScale(
        vmin=float(sizes.min()) if sizes.size else 0.0,
        vmax=float(sizes.max()) if sizes.size else 1.0,
        size_min=8.0,
        size_max=160.0,
    )
    areas = resolved_sizes.areas(sizes)
    scale, (vmin, vmax) = _log_fc_scale(
        log_fc[tested], significant[tested], color_scale
    )
    plotted = pd.DataFrame(
        {
            "neighborhood": np.arange(len(table)),
            "index_cell": index_cells,
            "x": x,
            "y": y,
            "size": sizes,
            "log_fc": log_fc,
            "spatial_fdr": table["spatial_fdr"].to_numpy(dtype=np.float64),
            "tested": tested,
            "significant": significant,
        }
    )

    _, mpl = require_matplotlib()
    if figsize is None and target is None:
        figsize = (DEFAULT_PANEL_INCHES + 1.6, DEFAULT_PANEL_INCHES + 0.4)
    edgecolor = scatter_edgecolor(theme)
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["neighborhoods"], figsize=figsize
        )
        ax = axes["neighborhoods"]
        norm = continuous_norm(
            mpl, vmin=vmin, vmax=vmax, vcenter=None, scale=scale.scale
        )
        grey = np.flatnonzero(~significant)
        grey = grey[np.argsort(-sizes[grey], kind="stable")]
        ax.scatter(
            x[grey],
            y[grey],
            s=areas[grey],
            color=scale.missing_color,
            edgecolors=edgecolor,
            linewidths=0.2,
            zorder=1,
        )
        coloured = np.flatnonzero(significant)
        coloured = coloured[np.argsort(np.abs(log_fc[coloured]), kind="stable")]
        artist = ax.scatter(
            x[coloured],
            y[coloured],
            s=areas[coloured],
            c=log_fc[coloured],
            cmap=scale.cmap,
            norm=norm,
            edgecolors=edgecolor,
            linewidths=0.3,
            zorder=2,
        )
        xlim, ylim = padded_square_limits(values[:, 0], values[:, 1])
        finish_embedding_axes(
            ax,
            xlim=xlim,
            ylim=ylim,
            title=f"Spatial FDR < {resolved_alpha:g}",
            frame="minimal",
        )
        legend_sizes = np.unique(
            np.round(np.linspace(resolved_sizes.vmin, resolved_sizes.vmax, 3))
        )
        if show_legend:
            colorbar = figure.colorbar(
                artist,
                ax=ax,
                location="top",
                orientation="horizontal",
                shrink=0.7,
                fraction=0.05,
                pad=0.04,
            )
            colorbar.set_label(_LOG_FC_LABEL)
            handles = [
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor=scale.missing_color,
                    markeredgecolor=edgecolor,
                    markersize=float(np.sqrt(area)),
                    label=f"{size:g}",
                )
                for size, area in zip(
                    legend_sizes, resolved_sizes.areas(legend_sizes), strict=True
                )
            ]
            ax.legend(
                handles=handles,
                title="Cells",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize="small",
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=dict(axes),
        tables={"neighborhoods": plotted},
        legends=(
            LegendSpec(
                kind="colorbar",
                label=_LOG_FC_LABEL,
                extras={"vmin": vmin, "vmax": vmax, "cmap": scale.cmap},
            ),
            LegendSpec(
                kind="size",
                label="Neighbourhood size",
                extras={"values": legend_sizes.tolist()},
            ),
        ),
        scales=(scale, resolved_sizes),
        provenance=PlotProvenance(
            assay=layout.assay,
            cell_key=None,
            n_cells=int(cell_idx.size),
            n_samples=int(result.design.shape[0]),
            renderer="matplotlib",
            notes=("neighborhood_graph",),
            extras={
                "neighborhood_tests": _artifact_record(result.artifact),
                "layout": layout.to_dict(),
                "alpha": resolved_alpha,
                "contrast": dict(result.contrast),
                "n_neighborhoods": int(len(table)),
                "n_significant": int(significant.sum()),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
