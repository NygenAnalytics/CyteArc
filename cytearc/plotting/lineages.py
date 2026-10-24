"""Plots of lineage profiles, lineage curves, and cluster trees."""

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..storage.artifacts import ArtifactRef, inspect_artifact
from ..trajectory.curves import smooth_against_progress
from ..trajectory.lineages import select_lineages
from ..utils.arguments import float_argument
from ._contracts import CategoricalScale, ColorScale, PlotProvenance
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
    colormap_palette,
    continuous_norm,
    default_point_size,
    finish_embedding_axes,
    padded_square_limits,
    theme_context,
)

__all__ = ["lineage_curves", "lineage_heatmap", "lineage_trends", "lineage_tree"]


def _positive(value: float, name: str) -> float:
    try:
        resolved = float_argument(value, name)
    except ValueError:
        resolved = np.nan
    if not resolved > 0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


def _require_loaded(loaded: ArtifactRef, requested: ArtifactRef, label: str) -> None:
    if loaded != requested:
        raise ValueError(f"Loaded {label} does not match the request")


def _lineage_selection(
    available: tuple[Any, ...],
    lineages: Sequence[Any] | None,
) -> list[int]:
    """Return the columns of the chosen lineages of a result, in order."""
    _, columns = select_lineages(available, lineages, where="in the result")
    return [int(column) for column in columns]


def _lineage_palette(names: Sequence[str], cmap: str) -> dict[str, str]:
    return colormap_palette(list(names), cmap)


@close_figures_on_error
def lineage_heatmap(
    store: Any,
    *,
    aggregation: ArtifactRef,
    lineages: Sequence[Any] | None = None,
    show_features: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    heatmap_cmap: str = "coolwarm",
    pseudotime_cmap: str = "viridis",
    clusterbar_cmap: str = "tab20",
    tick_fontsize: int = 10,
    axis_fontsize: int = 12,
    feature_label_fontsize: int = 10,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot fate-weighted feature profiles side by side for each lineage.

    Rows are features, grouped by module and, within a module, ordered by
    where their profile peaks. Each lineage's panel runs from the shared
    start to its sink, with a strip of the bins' mean pseudotime above it.
    Colours span ``vmin`` to ``vmax``. By default that is -2 to 2 for
    z-scaled profiles, and from the smallest mean to the 99th percentile for
    profiles stored without z-scaling, which the colorbar labels as mean
    expression.
    """
    if not isinstance(aggregation, ArtifactRef):
        raise TypeError("aggregation must be an ArtifactRef")
    for limit in (vmin, vmax):
        if limit is not None and not np.isfinite(limit):
            raise ValueError("vmax must be greater than vmin, and both finite")
    loaded = store.trajectory.load_lineage_aggregation(aggregation)
    _require_loaded(loaded.ref, aggregation, "lineage aggregation")
    z_scaled = bool(
        (inspect_artifact(store.zw, aggregation).parameters or {})["z_scale"]
    )
    columns = _lineage_selection(loaded.lineages, lineages)
    names = [str(loaded.lineages[column]) for column in columns]
    data = np.asarray(loaded.data)[:, columns, :]
    if vmin is None:
        vmin = -2.0 if z_scaled else float(data.min())
    if vmax is None:
        vmax = 2.0 if z_scaled else float(np.percentile(data, 99.0))
        # Retained profiles vary, so their maximum always exceeds their
        # minimum, even where the 99th percentile does not.
        if not z_scaled and not vmax > vmin:
            vmax = float(data.max())
    if not vmax > vmin:
        raise ValueError("vmax must be greater than vmin, and both finite")
    scale_label = "standardized expression" if z_scaled else "mean expression"
    clusters = np.asarray(loaded.feature_clusters)
    labels = np.asarray(loaded.feature_names).astype(str)
    peak = np.argmax(data.reshape(data.shape[0], -1), axis=1)
    row_order = np.lexsort((peak, clusters))
    data = data[row_order]
    clusters = clusters[row_order]
    labels = labels[row_order]
    cluster_order = sorted(set(clusters.tolist()))
    palette = colormap_palette(cluster_order, clusterbar_cmap)
    n_bins = data.shape[2]
    bin_pseudotime = np.asarray(loaded.bin_pseudotime)[columns]
    show_features = [] if show_features is None else list(show_features)
    # Feature labels sit in their own column between the last panel and the
    # colorbar, as wide as the longest label at its font size, so they never
    # cover the colorbar. A panel is 2.4 inches wide at the default size.
    label_inches = (
        0.2
        + max(len(str(feature)) for feature in show_features)
        * feature_label_fontsize
        * 0.65
        / 72.0
        if show_features
        else 0.0
    )
    label_width = max(label_inches / 2.4 * 6.0, 0.05)
    if figsize is None:
        figsize = (1.2 + 2.4 * len(columns) + label_inches, 8.0)

    plt, mpl = require_matplotlib()
    with theme_context(theme):
        figure = plt.figure(figsize=figsize, constrained_layout=False)
        grid = figure.add_gridspec(
            nrows=2,
            ncols=len(columns) + 3,
            height_ratios=(1, 24),
            width_ratios=(0.6, *([6.0] * len(columns)), label_width, 0.4),
            wspace=0.06,
            hspace=0.04,
        )
        axes: dict[Hashable, Any] = {}
        cluster_ax = figure.add_subplot(grid[1, 0])
        axes["feature_clusters"] = cluster_ax
        norm = continuous_norm(mpl, vmin=vmin, vmax=vmax, vcenter=None)
        for position, name in enumerate(names):
            heatmap_ax = figure.add_subplot(grid[1, position + 1])
            strip_ax = figure.add_subplot(grid[0, position + 1])
            image = heatmap_ax.imshow(
                data[:, position, :],
                aspect="auto",
                interpolation="nearest",
                cmap=heatmap_cmap,
                norm=norm,
                rasterized=True,
            )
            heatmap_ax.set_xticks([])
            heatmap_ax.set_yticks([])
            if position == len(names) - 1 and show_features:
                index = {label.lower(): row for row, label in enumerate(labels)}
                visible = [
                    feature for feature in show_features if feature.lower() in index
                ]
                heatmap_ax.yaxis.tick_right()
                heatmap_ax.set_yticks([index[feature.lower()] for feature in visible])
                heatmap_ax.set_yticklabels(visible, fontsize=feature_label_fontsize)
            heatmap_ax.set_xlabel(
                f"{bin_pseudotime[position, 0]:.2g} to "
                f"{bin_pseudotime[position, -1]:.2g}",
                fontsize=tick_fontsize,
            )
            strip_ax.imshow(
                bin_pseudotime[position].reshape(1, -1),
                aspect="auto",
                interpolation="nearest",
                cmap=pseudotime_cmap,
                vmin=float(bin_pseudotime.min()),
                vmax=float(bin_pseudotime.max()),
            )
            strip_ax.set_xticks([])
            strip_ax.set_yticks([])
            strip_ax.set_title(name, fontsize=axis_fontsize)
            axes[("heatmap", name)] = heatmap_ax
            axes[("pseudotime", name)] = strip_ax
        cluster_ax.imshow(
            np.asarray(
                [cluster_order.index(value) for value in clusters], dtype=float
            ).reshape(-1, 1),
            aspect="auto",
            interpolation="nearest",
            cmap=mpl.colors.ListedColormap(
                [palette[cluster] for cluster in cluster_order]
            ),
            vmin=-0.5,
            vmax=max(len(cluster_order) - 0.5, 0.5),
        )
        cluster_ax.set_xticks([])
        cluster_ax.set_yticks([])
        for cluster in cluster_order:
            row = float(np.flatnonzero(clusters == cluster).mean())
            cluster_ax.text(
                0, row, str(cluster), fontsize=tick_fontsize, ha="center", va="center"
            )
        cluster_ax.set_ylabel(f"{data.shape[0]} features", fontsize=axis_fontsize)
        if show_legend:
            colorbar_ax = figure.add_subplot(grid[1, -1])
            colorbar = figure.colorbar(image, cax=colorbar_ax)
            colorbar.ax.tick_params(labelsize=tick_fontsize)
            colorbar.set_label(scale_label, fontsize=axis_fontsize)
            axes["colorbar"] = colorbar.ax
        apply_figure_chrome(figure, theme)

    profiles = pd.DataFrame(
        data.reshape(data.shape[0], -1),
        index=pd.Index(labels, name="feature"),
        columns=pd.MultiIndex.from_product(
            (names, range(n_bins)), names=("lineage", "bin")
        ),
    )
    features = pd.DataFrame(
        {
            "feature_index": np.asarray(loaded.feature_indices)[row_order],
            "feature": labels,
            "cluster": clusters,
        }
    )
    bins = pd.DataFrame(
        {
            "lineage": np.repeat(names, n_bins),
            "bin": np.tile(np.arange(n_bins), len(names)),
            "pseudotime": bin_pseudotime.ravel(),
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"profiles": profiles, "features": features, "bins": bins},
        legends=(
            LegendSpec(
                kind="colorbar",
                label=scale_label,
                extras={"vmin": vmin, "vmax": vmax},
            ),
            LegendSpec(kind="categorical", label="feature modules"),
        ),
        scales=(
            ColorScale(cmap=heatmap_cmap, vmin=vmin, vmax=vmax),
            CategoricalScale(order=tuple(cluster_order), palette=palette),
        ),
        provenance=PlotProvenance(
            assay=loaded.assay,
            renderer="matplotlib",
            notes=("lineage_heatmap", "aggregated"),
            extras={
                "aggregation": aggregation.to_dict(),
                "fate": loaded.fate.to_dict(),
                "cell_selection": loaded.cell_selection.to_dict(),
                "feature_selection": loaded.feature_selection.to_dict(),
                "lineages": names,
                "n_features": int(data.shape[0]),
                "n_bins": int(n_bins),
                "show_features": show_features,
            },
        ),
        owns_figure=True,
        theme=theme,
    )
    if show:
        result.show()
    return result


@close_figures_on_error
def lineage_trends(
    store: Any,
    *,
    aggregation: ArtifactRef,
    features: Sequence[str],
    lineages: Sequence[Any] | None = None,
    axis: Literal["progress", "pseudotime"] = "progress",
    lineage_cmap: str = "tab10",
    line_width: float = 2.0,
    n_columns: int | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot selected features' profiles along each lineage.

    Each panel is one feature, with one line per lineage through the
    lineage's bins, so lineage-specific rises and falls stand side by side.
    With ``axis="progress"`` the bins are spaced by the share of the lineage's
    fate weight they have passed, so every bin is equally wide; with
    ``axis="pseudotime"`` they sit at their mean pseudotime, where bins
    inside an endpoint crowd together.
    """
    if not isinstance(aggregation, ArtifactRef):
        raise TypeError("aggregation must be an ArtifactRef")
    if isinstance(features, str) or not isinstance(features, Sequence) or not features:
        raise TypeError("features must be a non-empty sequence of feature names")
    if len({str(feature).lower() for feature in features}) != len(features):
        raise ValueError("features must not repeat a feature")
    if axis not in {"progress", "pseudotime"}:
        raise ValueError("axis must be 'progress' or 'pseudotime'")
    line_width = _positive(line_width, "line_width")
    loaded = store.trajectory.load_lineage_aggregation(aggregation)
    _require_loaded(loaded.ref, aggregation, "lineage aggregation")
    columns = _lineage_selection(loaded.lineages, lineages)
    names = [str(loaded.lineages[column]) for column in columns]
    labels = np.asarray(loaded.feature_names).astype(str)
    rows: list[int] = []
    for feature in features:
        matches = np.flatnonzero(np.char.lower(labels) == str(feature).lower())
        if not matches.size:
            raise ValueError(
                f"Feature {feature!r} is not a retained feature of the aggregation"
            )
        rows.append(int(matches[0]))
    panel_keys = [str(feature) for feature in features]
    resolved_columns = min(len(panel_keys), 4) if n_columns is None else n_columns
    if (
        isinstance(resolved_columns, bool)
        or not isinstance(resolved_columns, int)
        or resolved_columns < 1
    ):
        raise ValueError("n_columns must be a positive integer")
    if figsize is None and target is None:
        n_rows = int(np.ceil(len(panel_keys) / resolved_columns))
        figsize = (
            DEFAULT_PANEL_INCHES * min(resolved_columns, len(panel_keys)),
            DEFAULT_PANEL_INCHES * 0.8 * n_rows,
        )
    palette = _lineage_palette(names, lineage_cmap)
    data = np.asarray(loaded.data)
    bin_pseudotime = np.asarray(loaded.bin_pseudotime)
    n_bins = data.shape[2]
    progress = (np.arange(n_bins) + 0.5) / n_bins
    records = []
    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            target,
            panel_keys=panel_keys,
            figsize=figsize,
            n_columns=min(resolved_columns, len(panel_keys)),
        )
        for key, row in zip(panel_keys, rows, strict=True):
            panel = axes[key]
            for column, name in zip(columns, names, strict=True):
                panel.plot(
                    progress if axis == "progress" else bin_pseudotime[column],
                    data[row, column],
                    color=palette[name],
                    linewidth=line_width,
                    label=name,
                )
                records.append(
                    pd.DataFrame(
                        {
                            "feature": key,
                            "lineage": name,
                            "progress": progress,
                            "pseudotime": bin_pseudotime[column],
                            "value": data[row, column],
                        }
                    )
                )
            panel.set_title(key)
            panel.set_xlabel("Lineage progress" if axis == "progress" else "Pseudotime")
        if show_legend:
            first = axes[panel_keys[0]]
            first.legend(frameon=False, fontsize="small")
        apply_figure_chrome(figure, theme)
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"trends": pd.concat(records, ignore_index=True)},
        legends=(LegendSpec(kind="categorical", label="lineage"),),
        scales=(CategoricalScale(order=tuple(names), palette=palette),),
        provenance=PlotProvenance(
            assay=loaded.assay,
            renderer="matplotlib",
            notes=("lineage_trends", "aggregated"),
            extras={
                "aggregation": aggregation.to_dict(),
                "lineages": names,
                "features": panel_keys,
                "axis": axis,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    if show:
        result.show()
    return result


def _layout_for(
    store: Any, layout: ArtifactRef, cell_selection: ArtifactRef
) -> tuple[np.ndarray, np.ndarray]:
    coordinates, cell_indices, layout_selection = _resolve_layout(store, layout)
    if layout_selection != cell_selection:
        raise ValueError(
            "layout must use the exact cell-selection artifact of the trajectory"
        )
    return coordinates, cell_indices


@close_figures_on_error
def lineage_curves(
    store: Any,
    *,
    curves: ArtifactRef,
    layout: ArtifactRef,
    lineages: Sequence[Any] | None = None,
    df: float | None = None,
    lineage_cmap: str = "tab10",
    line_width: float = 2.5,
    point_size: float | None = None,
    point_alpha: float = 0.6,
    cell_color: str = "#d0d0d0",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Draw fate-weighted lineage curves over a two-dimensional layout.

    The curves were fitted in their own coordinates, often many reduced
    dimensions. To draw one in a layout such as a UMAP, the layout
    coordinates of the cells are averaged with the lineage's fate weights
    along the cells' positions on the curve and smoothed with the spline
    the curves were fitted with, as Slingshot's ``embedCurves`` does.
    ``df`` overrides the curves' degrees of freedom for this smoothing; like
    theirs, it lies between 2 and the size of their spline basis, 32 by
    default.
    """
    if not isinstance(curves, ArtifactRef):
        raise TypeError("curves must be an ArtifactRef")
    line_width = _positive(line_width, "line_width")
    if point_size is not None:
        point_size = _positive(point_size, "point_size")
    if not np.isfinite(point_alpha) or not 0 <= point_alpha <= 1:
        raise ValueError("point_alpha must be between zero and one")
    requested_df = np.nan
    if df is not None:
        try:
            requested_df = float_argument(df, "df")
        except ValueError:
            requested_df = np.nan
    loaded = store.trajectory.load_curves(curves)
    _require_loaded(loaded.ref, curves, "lineage curves")
    parameters = inspect_artifact(store.zw, curves).parameters or {}
    spline_basis = int(parameters["spline_basis"])
    if df is not None and not 2.0 < requested_df < spline_basis:
        raise ValueError(f"df must be greater than 2 and less than {spline_basis}")
    degrees_of_freedom = float(parameters["df"] if df is None else requested_df)
    coordinates, cell_indices = _layout_for(store, layout, loaded.cell_selection)
    columns = _lineage_selection(loaded.lineages, lineages)
    names = [str(loaded.lineages[column]) for column in columns]
    valid = np.asarray(loaded.valid, dtype=bool)
    scored_coordinates = coordinates[valid]
    drawn: dict[str, np.ndarray] = {}
    for column, name in zip(columns, names, strict=True):
        drawn[name] = smooth_against_progress(
            scored_coordinates,
            loaded.progress[valid, column],
            loaded.weights[valid, column],
            n_points=loaded.points.shape[1],
            degrees_of_freedom=degrees_of_freedom,
            n_bins=int(parameters["fit_bins"]),
            n_basis=spline_basis,
        )
    palette = _lineage_palette(names, lineage_cmap)
    xlim, ylim = padded_square_limits(coordinates[:, 0], coordinates[:, 1])
    if figsize is None and target is None:
        # Room on the right for the legend beside the square panel.
        figsize = (DEFAULT_PANEL_INCHES + 1.4, DEFAULT_PANEL_INCHES)
    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            target,
            panel_keys=["curves"],
            figsize=figsize,
        )
        axis = axes["curves"]
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=default_point_size(len(cell_indices))
            if point_size is None
            else point_size,
            c=cell_color,
            alpha=point_alpha,
            linewidths=0,
            rasterized=len(cell_indices) >= 50_000,
        )
        for name, points in drawn.items():
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=palette[name],
                linewidth=line_width,
                label=name,
            )
            axis.plot(
                points[-1, 0],
                points[-1, 1],
                marker="o",
                color=palette[name],
                markersize=line_width * 2.5,
            )
        finish_embedding_axes(axis, xlim=xlim, ylim=ylim, title="Lineage curves")
        if show_legend:
            axis.legend(
                frameon=False,
                fontsize="small",
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
            )
        apply_figure_chrome(figure, theme)
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "lineage": name,
                    "point": np.arange(len(points)),
                    "x": points[:, 0],
                    "y": points[:, 1],
                }
            )
            for name, points in drawn.items()
        ],
        ignore_index=True,
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"curves": table},
        legends=(LegendSpec(kind="categorical", label="lineage"),),
        scales=(CategoricalScale(order=tuple(names), palette=palette),),
        provenance=PlotProvenance(
            n_cells=len(cell_indices),
            renderer="matplotlib",
            notes=("lineage_curves", "artifact"),
            extras={
                "curves": curves.to_dict(),
                "layout": layout.to_dict(),
                "fate": loaded.fate.to_dict(),
                "cell_selection": loaded.cell_selection.to_dict(),
                "lineages": names,
                "df": degrees_of_freedom,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    if show:
        result.show()
    return result


@close_figures_on_error
def lineage_tree(
    store: Any,
    *,
    tree: ArtifactRef,
    layout: ArtifactRef,
    show_cells: bool = True,
    cell_color: str = "#d0d0d0",
    point_size: float | None = None,
    point_alpha: float = 0.5,
    node_size: float = 80.0,
    node_color: str = "#4e79a7",
    root_color: str = "#59a14f",
    leaf_color: str = "#e15759",
    edge_color: str = "#555555",
    labels: bool = True,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Draw a cluster tree over a two-dimensional layout.

    Each cluster sits at the median layout position of its cells. The root
    and the candidate sinks, the leaves other than the root, are coloured
    apart from the other clusters, and the legend names the colours.
    """
    if not isinstance(tree, ArtifactRef):
        raise TypeError("tree must be an ArtifactRef")
    node_size = _positive(node_size, "node_size")
    if point_size is not None:
        point_size = _positive(point_size, "point_size")
    if not np.isfinite(point_alpha) or not 0 <= point_alpha <= 1:
        raise ValueError("point_alpha must be between zero and one")
    loaded = store.trajectory.load_lineage_tree(tree)
    _require_loaded(loaded.ref, tree, "lineage tree")
    coordinates, cell_indices = _layout_for(store, layout, loaded.cell_selection)
    labels_values = store.artifacts.load_values(
        loaded.labels, cell_selection=loaded.cell_selection
    )
    # One pass factorizes the labels and takes every cluster's median,
    # instead of comparing every cell's label once per cluster.
    codes, uniques = pd.factorize(
        np.asarray(labels_values.values), use_na_sentinel=False
    )
    medians = pd.DataFrame(coordinates[:, :2]).groupby(codes).median()
    code_of = {
        value.item() if isinstance(value, np.generic) else value: code
        for code, value in enumerate(np.asarray(uniques, dtype=object))
    }
    positions: dict[Any, tuple[float, float]] = {}
    for cluster in loaded.clusters:
        x, y = medians.loc[code_of[cluster]]
        positions[cluster] = (float(x), float(y))
    candidates = set(loaded.candidate_sinks)
    role_colors = {
        "root": root_color,
        "candidate sink": leaf_color,
        "cluster": node_color,
    }
    roles = {
        cluster: "root"
        if loaded.root is not None and bool(cluster == loaded.root)
        else "candidate sink"
        if cluster in candidates
        else "cluster"
        for cluster in positions
    }
    present = tuple(role for role in role_colors if role in roles.values())
    palette = {role: role_colors[role] for role in present}
    xlim, ylim = padded_square_limits(coordinates[:, 0], coordinates[:, 1])
    if figsize is None and target is None and show_legend:
        # Room on the right for the legend beside the square panel.
        figsize = (DEFAULT_PANEL_INCHES + 1.6, DEFAULT_PANEL_INCHES)
    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            target,
            panel_keys=["tree"],
            figsize=figsize,
        )
        axis = axes["tree"]
        if show_cells:
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=default_point_size(len(cell_indices))
                if point_size is None
                else point_size,
                c=cell_color,
                alpha=point_alpha,
                linewidths=0,
                rasterized=len(cell_indices) >= 50_000,
            )
        for source, target_cluster in zip(
            loaded.edges["source"], loaded.edges["target"], strict=True
        ):
            first, second = positions[source], positions[target_cluster]
            axis.plot(
                (first[0], second[0]),
                (first[1], second[1]),
                color=edge_color,
                linewidth=1.5,
                zorder=2,
            )
        for cluster, (x, y) in positions.items():
            axis.scatter(
                [x],
                [y],
                s=node_size,
                c=palette[roles[cluster]],
                zorder=3,
                edgecolors="white",
            )
            if labels:
                axis.annotate(
                    str(cluster),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize="small",
                )
        finish_embedding_axes(axis, xlim=xlim, ylim=ylim, title="Cluster tree")
        if show_legend:
            _, mpl = require_matplotlib()
            axis.legend(
                handles=[
                    mpl.lines.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markerfacecolor=palette[role],
                        markeredgecolor="white",
                        markersize=8,
                    )
                    for role in present
                ],
                labels=list(present),
                frameon=False,
                fontsize="small",
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
            )
        apply_figure_chrome(figure, theme)
    nodes = pd.DataFrame(
        {
            "cluster": list(positions),
            "x": [position[0] for position in positions.values()],
            "y": [position[1] for position in positions.values()],
            "size": np.asarray(loaded.sizes),
            "candidate_sink": [cluster in candidates for cluster in positions],
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"nodes": nodes, "edges": loaded.edges.copy()},
        legends=(LegendSpec(kind="categorical", label="cluster role"),),
        scales=(CategoricalScale(order=present, palette=palette),),
        provenance=PlotProvenance(
            n_cells=len(cell_indices),
            renderer="matplotlib",
            notes=("lineage_tree", "artifact"),
            extras={
                "tree": tree.to_dict(),
                "layout": layout.to_dict(),
                "cell_selection": loaded.cell_selection.to_dict(),
                "root": loaded.root,
                "candidate_sinks": list(loaded.candidate_sinks),
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    if show:
        result.show()
    return result
