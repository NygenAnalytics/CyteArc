"""Native diagnostic plots."""

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._contracts import CategoricalScale, PlotProvenance
from ._deps import require_kneed, require_matplotlib, require_seaborn
from ._display import groups_in_stored_order
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._style import (
    apply_figure_chrome,
    colormap_argument,
    positive_argument,
    scatter_edgecolor,
    theme_argument,
    theme_context,
)
from ..utils.arguments import (
    RANDOM_STATE_MAX,
    bool_argument,
    float_argument,
    integer_argument,
)

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
else:
    # Matplotlib is optional, so the hints name Colormap only for type
    # checkers and resolve without it.
    Colormap = Any


def _clean_axis(ax: Any, *, tick_size: float = 10.0) -> None:
    ax.tick_params(axis="both", labelsize=tick_size)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(which="major", linestyle="--", alpha=0.4)


def _return_result(result: PlotResult, *, show: bool) -> PlotResult:
    if show:
        result.show()
    return result


@close_figures_on_error
def qc(
    data: pd.DataFrame,
    color: str = "steelblue",
    cmap: str = "tab20",
    figsize: tuple[float, float] | None = None,
    label_size: float = 10.0,
    title_size: float = 10,
    sup_title: str | None = None,
    sup_title_size: float = 12,
    scatter_size: float = 1.0,
    max_points: int = 10_000,
    seed: int = 0,
    show_on_single_row: bool = True,
    show: bool = True,
    theme: str = "notebook",
    group_title: str | None = None,
) -> PlotResult:
    """Plot grouped distributions for cell-level QC metrics.

    Above ``max_points`` cells, each panel draws the points it shows with
    ``seed``, a seed of NumPy's ``RandomState`` from 0 to 2**32 - 1. The
    groups take the colors of ``cmap``, a palette that seaborn's
    ``color_palette`` reads: the name of a seaborn palette or a Matplotlib
    colormap, or a list of colors. ``group_title`` names what the groups
    of the ``groups`` column are, such as "Sample": it titles each panel's
    axis of the groups and the legend that ``PlotResult.legends`` reports.
    Without it the axis is left blank and the legend is reported as
    ``groups``, the column's name; a panel of one group draws no axis of
    groups to title.

    ``data`` is a pandas DataFrame with a ``groups`` column and one numeric column per metric. The groups of an ordered ``Categorical`` follow its categories, as the groups of a column stored as ordered do in the other plots, and the cells without a group come after them, as ``NA``; other groups take natural order.
    """
    show_on_single_row = bool_argument(show_on_single_row, "show_on_single_row")
    show = bool_argument(show, "show")
    seed = integer_argument(seed, "seed", minimum=0, maximum=RANDOM_STATE_MAX)
    require_matplotlib()
    sns = require_seaborn()
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame with a 'groups' column, not "
            f"{type(data).__name__}"
        )
    if "groups" not in data.columns:
        raise KeyError("data must contain a 'groups' column")
    if group_title is not None and not isinstance(group_title, str):
        raise TypeError(
            f"group_title must be text or None, not {type(group_title).__name__}"
        )
    if data.empty:
        raise ValueError("data must contain at least one row")
    if max_points < 0:
        raise ValueError("max_points must be non-negative")

    metric_columns = [column for column in data.columns if column != "groups"]
    if not metric_columns:
        raise ValueError("data must contain at least one metric column")
    if len(set(metric_columns)) != len(metric_columns):
        raise ValueError("metric column names must be unique")

    numeric_values: dict[Any, np.ndarray] = {}
    for metric in metric_columns:
        try:
            numeric_values[metric] = pd.to_numeric(
                data[metric], errors="raise"
            ).to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"QC metric {metric!r} must be numeric") from exc

    groups = data["groups"]
    group_values = groups.astype(object).where(groups.notna(), "NA")
    observed = list(pd.unique(group_values))
    # An ordered Categorical orders the groups by its categories, as a column
    # stored as ordered orders the groups of the other plots.
    stored = (
        CategoricalScale(order=tuple(groups.cat.categories))
        if isinstance(groups.dtype, pd.CategoricalDtype) and groups.dtype.ordered
        else None
    )
    group_order = [
        observed[at]
        for at in groups_in_stored_order(
            [(str(label),) for label in observed], [stored]
        )
    ]
    n_groups = len(group_order)
    n_panels = len(metric_columns)
    n_columns = n_panels if show_on_single_row else 1
    n_rows = 1 if show_on_single_row else n_panels
    resolved_figsize = figsize
    if resolved_figsize is None:
        resolved_figsize = (
            min(15.0, n_groups + (2.0 * n_columns)),
            1.0 + (2.5 * n_rows),
        )

    try:
        palette_colors = sns.color_palette(cmap, n_colors=n_groups).as_hex()
    except (TypeError, ValueError) as error:
        raised = TypeError if isinstance(error, TypeError) else ValueError
        raise raised(
            "cmap must be a seaborn palette or Matplotlib colormap name, or a "
            f"list of colors, not {cmap!r}"
        ) from error
    palette = dict(zip(group_order, palette_colors, strict=True))
    if n_groups == 1:
        palette[group_order[0]] = color

    long_frames: list[pd.DataFrame] = []
    displayed_points: dict[str, int] = {}
    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=metric_columns,
            figsize=resolved_figsize,
            n_columns=n_columns,
        )
        for metric in metric_columns:
            ax = axes[metric]
            metric_frame = pd.DataFrame(
                {
                    "group": group_values.to_numpy(),
                    "metric": str(metric),
                    "value": numeric_values[metric],
                }
            )
            long_frames.append(metric_frame)
            collections_before = len(ax.collections)
            if n_groups == 1:
                sns.violinplot(
                    data=metric_frame,
                    x="group",
                    y="value",
                    order=group_order,
                    ax=ax,
                    color=color,
                    linewidth=1,
                    inner=None,
                    cut=0,
                    saturation=1,
                )
            else:
                sns.violinplot(
                    data=metric_frame,
                    x="group",
                    y="value",
                    order=group_order,
                    hue="group",
                    hue_order=group_order,
                    palette=palette,
                    legend=False,
                    dodge=False,
                    ax=ax,
                    linewidth=1,
                    inner=None,
                    cut=0,
                    saturation=1,
                )
            for collection in ax.collections[collections_before:]:
                collection.set_alpha(0.6)

            if max_points == 0:
                display_frame = metric_frame.iloc[:0]
            elif len(metric_frame) > max_points:
                display_frame = metric_frame.sample(
                    n=max_points,
                    random_state=seed,
                )
            else:
                display_frame = metric_frame
            displayed_points[str(metric)] = len(display_frame)
            if not display_frame.empty:
                sns.stripplot(
                    data=display_frame,
                    x="group",
                    y="value",
                    order=group_order,
                    jitter=0.4,
                    ax=ax,
                    size=scatter_size,
                    color="black",
                    alpha=0.4,
                )

            ax.set_ylabel(str(metric), fontsize=label_size)
            ax.set_xlabel(
                group_title if group_title and n_groups > 1 else "",
                fontsize=label_size,
            )
            if n_groups == 1:
                ax.set_xticks([])
                finite = numeric_values[metric][np.isfinite(numeric_values[metric])]
                if len(finite):
                    ax.set_title(
                        f"Median: {np.median(finite):.1f}",
                        fontsize=title_size,
                    )
            _clean_axis(ax, tick_size=label_size)

        if sup_title is not None:
            figure.suptitle(sup_title, fontsize=sup_title_size)
        apply_figure_chrome(figure, theme)

    long_data = pd.concat(long_frames, ignore_index=True)
    summary = (
        long_data.groupby(["group", "metric"], observed=False, dropna=False, sort=False)
        .agg(count=("value", "count"), median=("value", "median"))
        .reset_index()
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "data": data.copy(),
            "summary": summary,
        },
        legends=(
            LegendSpec(
                kind="categorical",
                label="groups" if group_title is None else group_title,
                extras={"categories": list(group_order)},
            ),
        ),
        scales=(
            CategoricalScale(
                order=tuple(group_order),
                palette=palette,
            ),
        ),
        provenance=PlotProvenance(
            n_cells=len(data),
            renderer="matplotlib",
            notes=("qc",),
            extras={
                "metrics": [str(metric) for metric in metric_columns],
                "groups": list(group_order),
                "max_points": max_points,
                "seed": seed,
                "displayed_points": displayed_points,
                "show_on_single_row": show_on_single_row,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)


@close_figures_on_error
def elbow(
    variance_explained: np.ndarray | list[float],
    figsize: tuple[float | None, float] = (None, 2),
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot explained variance and mark the detected elbow.

    Principal components are numbered from 1, as ``dims`` counts them, so the elbow line stands at the number of the PC where the curve bends. The ``variance_explained`` table holds ``pc``, ``variance_explained`` and ``is_elbow`` for each PC, and ``provenance.extras["elbow_pc"]`` holds the elbow PC's number, or None when no elbow is detected. Fewer than three values, or values that are all equal, have no elbow. Without a width in ``figsize``, the figure is a quarter inch wide for each PC and at least 2.25 inches wide.
    """
    show = bool_argument(show, "show")
    require_matplotlib()
    values = np.asarray(variance_explained, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("variance_explained must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("variance_explained must contain only finite values")

    pcs = np.arange(1, len(values) + 1, dtype=np.int64)
    elbow_pc = None
    # An elbow lies between the curve's ends, and kneed scales the values by
    # their range, so fewer than three values, or values that are all equal,
    # are not searched.
    if len(values) >= 3 and values.min() < values.max():
        knee_locator = require_kneed()
        with warnings.catch_warnings():
            # kneed 0.8.1 to 0.8.3 warn when they find no knee, which
            # elbow_pc reports; later releases do not warn.
            warnings.filterwarnings(
                "ignore", message="No knee/elbow found$", category=UserWarning
            )
            locator = knee_locator(
                pcs,
                values,
                S=1.0,
                curve="convex",
                direction="decreasing",
            )
        detected = getattr(locator, "elbow", None)
        elbow_pc = None if detected is None else int(detected)
    width, height = figsize
    resolved_figsize = (
        # 2.25 inches fit the x axis label, 1.4 inches wide at 9 points,
        # beside the y axis label and tick labels of up to seven characters.
        max(2.25, 0.25 * len(values)) if width is None else float(width),
        float(height),
    )

    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=["elbow"],
            figsize=resolved_figsize,
        )
        ax = axes["elbow"]
        ax.plot(pcs, values, linewidth=1)
        ax.set_xticks(pcs)
        if elbow_pc is not None:
            ax.axvline(
                elbow_pc,
                linewidth=1,
                color="red",
                label="Elbow",
            )
            ax.legend(frameon=False, fontsize=9)
        ax.set_ylabel("% Variance explained", fontsize=9)
        ax.set_xlabel("Principal components", fontsize=9)
        _clean_axis(ax, tick_size=8)
        apply_figure_chrome(figure, theme)

    table = pd.DataFrame(
        {
            "pc": pcs,
            "variance_explained": values,
            "is_elbow": pcs == elbow_pc,
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"variance_explained": table},
        legends=(
            ()
            if elbow_pc is None
            else (
                LegendSpec(
                    kind="line",
                    label="Elbow",
                    extras={"pc": elbow_pc},
                ),
            )
        ),
        scales=(),
        provenance=PlotProvenance(
            n_cells=0,
            renderer="matplotlib",
            notes=("elbow",),
            extras={
                "elbow_pc": elbow_pc,
                "n_components": len(values),
                "sensitivity": 1.0,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)


@close_figures_on_error
def graph_qc(
    graph: Any,
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot node degree and edge weight distributions for a sparse graph."""
    show = bool_argument(show, "show")
    require_matplotlib()
    shape = getattr(graph, "shape", None)
    if shape is None or len(shape) != 2:
        raise TypeError("graph must be a two-dimensional sparse matrix")
    if not hasattr(graph, "data"):
        raise TypeError("graph must expose sparse edge weights through .data")

    try:
        degrees = np.asarray((graph != 0).sum(axis=0)).ravel().astype(np.int64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "graph must support sparse non-zero degree calculation"
        ) from exc
    edge_weights = np.asarray(graph.data, dtype=np.float64).ravel()
    node_degrees = pd.DataFrame(
        {
            "node": np.arange(len(degrees), dtype=np.int64),
            "degree": degrees,
        }
    )
    degree_frequencies = (
        pd.Series(degrees, name="degree")
        .value_counts(sort=False)
        .sort_index()
        .rename_axis("degree")
        .reset_index(name="frequency")
    )
    degree_clip_limit = (
        float(np.percentile(degrees, 99.5) + 5.0) if len(degrees) else 1.0
    )

    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=["node_degree", "edge_weight"],
            figsize=(12.0, 4.0),
            n_columns=2,
        )
        degree_ax = axes["node_degree"]
        degree_ax.bar(
            degree_frequencies["degree"],
            degree_frequencies["frequency"],
            width=0.5,
        )
        degree_ax.set_xlim((0.0, max(1.0, degree_clip_limit)))
        degree_ax.set_xlabel("Node degree")
        degree_ax.set_ylabel("Frequency")
        if len(degrees) and degrees.max() > degree_clip_limit:
            degree_ax.text(
                max(1.0, degree_clip_limit),
                float(degree_frequencies["frequency"].max()),
                f"plot is clipped (max degree: {degrees.max()})",
                ha="right",
                fontsize=9,
            )
        _clean_axis(degree_ax)

        weight_ax = axes["edge_weight"]
        weight_ax.hist(edge_weights, bins=30)
        weight_ax.set_xlabel("Edge weight")
        weight_ax.set_ylabel("Frequency")
        _clean_axis(weight_ax)
        apply_figure_chrome(figure, theme)

    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "node_degrees": node_degrees,
            "degree_frequencies": degree_frequencies,
            "edge_weights": pd.DataFrame({"edge_weight": edge_weights}),
        },
        legends=(
            LegendSpec(kind="bar", label="Node degree frequency"),
            LegendSpec(
                kind="histogram",
                label="Edge weight",
                extras={"bins": 30},
            ),
        ),
        scales=(),
        provenance=PlotProvenance(
            n_cells=len(degrees),
            renderer="matplotlib",
            notes=("graph_qc",),
            extras={
                "graph_shape": [int(shape[0]), int(shape[1])],
                "n_edges": len(edge_weights),
                "degree_clip_limit": degree_clip_limit,
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)


def _pair(
    value: object, name: str, pair: str, item: Callable[[object, str], float]
) -> tuple[float, float]:
    """Check a pair of numbers, such as a figure's width and height, each with ``item``."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a pair of {pair}")
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    first, second = value
    return item(first, f"{name}[0]"), item(second, f"{name}[1]")


def _marker_area(value: object, name: str) -> float:
    area = float_argument(value, name)
    if area < 0:
        raise ValueError(f"{name} must be at least 0")
    return area


def _label_size_option(value: object) -> float:
    return positive_argument(value, "label_size")


def _figsize_option(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    return _pair(
        value, "figsize", "numbers, the width and height in inches", positive_argument
    )


def _point_sizes_option(value: object) -> tuple[float, float]:
    return _pair(
        value,
        "point_sizes",
        "marker areas in points squared, for the features not selected and "
        "the selected ones",
        _marker_area,
    )


def _colormaps_option(value: object) -> tuple[str | None, str | None]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            "colormaps must be a pair of colormaps, for the features not "
            "selected and the selected ones"
        )
    if len(value) != 2:
        raise ValueError("colormaps must contain two values")
    return (
        colormap_argument(value[0], "colormaps[0]", optional=True),
        colormap_argument(value[1], "colormaps[1]", optional=True),
    )


_HVG_PLOT_OPTIONS: dict[str, Callable[[object], Any]] = {
    "label_size": _label_size_option,
    "figsize": _figsize_option,
    "point_sizes": _point_sizes_option,
    "colormaps": _colormaps_option,
    "theme": theme_argument,
}


def hvg_plot_options(**options: object) -> dict[str, Any]:
    """Check options of :func:`highly_variable_features` without drawing.

    The plot checks its options with it before it draws, and
    ``features.hvgs`` checks the options that it passes to the plot, whose
    names it has checked against the plot's signature, before it selects
    genes or saves anything.

    Returns:
        The options given, as the plot draws them: ``label_size`` as a float,
        ``figsize`` and ``point_sizes`` as pairs of floats, ``colormaps`` as
        a pair of colormap names or None, and ``theme``.

    Raises:
        TypeError: If an option has a wrong type.
        ValueError: If a pair does not hold two values, a size is out of
            range, or a colormap is not one that Matplotlib holds.
        KeyError: If ``theme`` names no theme.
    """
    return {name: _HVG_PLOT_OPTIONS[name](value) for name, value in options.items()}


@close_figures_on_error
def highly_variable_features(
    mean_nonzero: np.ndarray,
    corrected_variance: np.ndarray,
    n_cells: np.ndarray,
    selected: np.ndarray,
    *,
    label_size: float = 12,
    figsize: tuple[float, float] | None = (4.5, 4.0),
    point_sizes: tuple[float, float] = (3, 30),
    colormaps: "tuple[str | Colormap | None, str | Colormap | None]" = (
        "winter",
        "magma_r",
    ),
    theme: str = "notebook",
    show: bool = True,
) -> PlotResult:
    """Plot mean-variance diagnostics for feature selection.

    ``colormaps`` colors the features not selected and the selected ones by
    the number of cells that express them, each a Matplotlib colormap name
    or a registered ``Colormap``, or None for Matplotlib's default. The
    legend marks each group with one marker of one size in the middle color
    of its colormap. ``point_sizes`` holds the two groups' marker areas in
    points squared, ``label_size`` the size in points of the axis labels and
    ticks, and ``figsize`` the figure's width and height in inches, or None
    for CyteArc's default panel size. Every option is checked before the plot
    draws.
    """
    show = bool_argument(show, "show")
    options = hvg_plot_options(
        label_size=label_size,
        figsize=figsize,
        point_sizes=point_sizes,
        colormaps=colormaps,
        theme=theme,
    )
    label_size = options["label_size"]
    figsize = options["figsize"]
    point_sizes = options["point_sizes"]
    colormaps = options["colormaps"]
    theme = options["theme"]
    _, mpl = require_matplotlib()
    means = np.asarray(mean_nonzero, dtype=np.float64)
    variances = np.asarray(corrected_variance, dtype=np.float64)
    cell_counts = np.asarray(n_cells, dtype=np.float64)
    selected_mask = np.asarray(selected)
    arrays = (means, variances, cell_counts, selected_mask)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("feature diagnostic inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("feature diagnostic inputs must have matching lengths")
    selected_mask = selected_mask.astype(bool, copy=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_means = np.log2(means)
        log_variances = np.log2(variances)

    with theme_context(theme):
        figure, axes, owns_figure = normalize_axes_target(
            None,
            panel_keys=["highly_variable_features"],
            figsize=figsize,
        )
        ax = axes["highly_variable_features"]
        not_selected_points = ax.scatter(
            log_means[~selected_mask],
            log_variances[~selected_mask],
            alpha=0.6,
            c=cell_counts[~selected_mask],
            cmap=colormaps[0],
            s=point_sizes[0],
            label="Not selected",
        )
        selected_points = ax.scatter(
            log_means[selected_mask],
            log_variances[selected_mask],
            alpha=0.8,
            c=cell_counts[selected_mask],
            cmap=colormaps[1],
            s=point_sizes[1],
            edgecolor="black",
            linewidth=0.5,
            label="Selected",
        )
        ax.set_xlabel("Log mean non-zero expression", fontsize=label_size)
        ax.set_ylabel("Log corrected variance", fontsize=label_size)
        # A scatter's own legend marker copies its first point, which is
        # tiny for the features not selected, and transparent when no cell
        # expresses the first of them, so each group gets a marker of its own.
        # A thin neutral edge keeps a colormap's pale middle, such as bwr's
        # white, visible beside the selected marker's black edge.
        ax.legend(
            handles=[
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=points.get_cmap()(0.5),
                    markeredgecolor=edgecolor,
                    markeredgewidth=0.5,
                    alpha=points.get_alpha(),
                    label=points.get_label(),
                )
                for points, edgecolor in (
                    (not_selected_points, scatter_edgecolor(theme)),
                    (selected_points, "black"),
                )
            ],
            frameon=False,
            fontsize=max(8.0, label_size - 2.0),
        )
        _clean_axis(ax, tick_size=label_size)
        apply_figure_chrome(figure, theme)

    feature_table = pd.DataFrame(
        {
            "mean_nonzero": means,
            "corrected_variance": variances,
            "n_cells": cell_counts,
            "selected": selected_mask,
            "log2_mean_nonzero": log_means,
            "log2_corrected_variance": log_variances,
        }
    )
    result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"features": feature_table},
        legends=(
            LegendSpec(
                kind="categorical",
                label="Highly variable",
                extras={"categories": ["Not selected", "Selected"]},
            ),
        ),
        scales=(),
        provenance=PlotProvenance(
            n_cells=0,
            renderer="matplotlib",
            notes=("highly_variable_features",),
            extras={
                "n_features": len(means),
                "n_selected": int(selected_mask.sum()),
                "max_expressing_cells": (
                    float(np.nanmax(cell_counts)) if len(cell_counts) else 0.0
                ),
                "point_sizes": list(point_sizes),
                "colormaps": list(colormaps),
            },
        ),
        owns_figure=owns_figure,
        theme=theme,
    )
    return _return_result(result, show=show)
