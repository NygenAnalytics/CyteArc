"""Figures of clones: the cells that descend from one ancestor cell."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..clones.diversity import ORDERS
from ..clones.expansion import DEFAULT_RANK_SPLITS, rank_window_labels, rank_windows
from ..storage.refs import ArtifactRef
from ..utils.arguments import integer_argument
from ..utils.arrays import sort_categories, sorted_unique
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._groups import cell_groups
from ._style import (
    apply_figure_chrome,
    categorical_color_map,
    colormap_palette,
    theme_context,
)

__all__ = ["clonal_expansion", "clonal_rarefaction", "clone_abundance"]

_ORDER_TITLES = {
    0: "Clones (order 0)",
    1: "Shannon diversity (order 1)",
    2: "Simpson diversity (order 2)",
}


def _clones_method(store: Any, name: str) -> Any:
    """Return a method of the store's clones namespace, or raise."""
    method = getattr(getattr(store, "clones", None), name, None)
    if not callable(method):
        raise TypeError(f"store does not provide clones.{name}")
    return method


def _load_clones(store: Any, clones: ArtifactRef) -> Any:
    return _clones_method(store, "load_clones")(clones)


@close_figures_on_error
def clone_abundance(
    store: Any,
    clones: ArtifactRef,
    *,
    group_by: str | None = None,
    groups: ArtifactRef | None = None,
    top: int = 20,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot how many clones have each size, and the largest clones.

    ``clones`` is a ``clones`` artifact, such as one from
    ``ds.clones.import_clones``. The first panel counts clones by their
    number of cells on log scales: in most experiments many clones hold one
    or a few cells, and expanded clones form the tail. The second panel draws
    the ``top`` clones with the most cells, labelled with their IDs.

    ``group_by`` names a cell metadata column, such as the sample or time
    point, and ``groups`` a cell-label artifact, such as a clustering; with
    either, the first panel draws one line per group, counting the cells of
    each clone within the group, and the second stacks each clone's cells by
    group. Cells without a group are left out. The plot raises ``ValueError``
    when no cell with a group has a clone.

    Tables: ``sizes`` (``group``, ``clone_size``, ``n_clones``) and ``top``
    (``clone``, then the cells of each group, or ``cells``).
    """
    top = integer_argument(top, "top", minimum=1)
    loaded = _load_clones(store, clones)
    group_labels, names, group_name = cell_groups(
        store, clones, group_by=group_by, groups=groups
    )
    codes = np.asarray(loaded.cells["clone"].cat.codes, dtype=np.int64)
    if group_labels is None:
        cell_group = np.zeros(len(codes), dtype=np.int64)
        names = ["all cells"]
    else:
        cell_group = group_labels
    keep = (codes >= 0) & (cell_group >= 0)
    if not keep.any():
        raise ValueError(
            "No cell with a group has a clone, so there is nothing to plot"
            if group_labels is not None
            else "No cell has a clone, so there is nothing to plot"
        )
    # Each clone's cells in each group, from the codes alone.
    n_clones = len(loaded.clones)
    pairs, pair_cells = np.unique(
        cell_group[keep] * n_clones + codes[keep], return_counts=True
    )
    held = sorted_unique(pairs // n_clones)
    group_order = [names[code] for code in held.tolist()]
    pair_groups = np.searchsorted(held, pairs // n_clones)
    pair_codes = pairs % n_clones
    width = int(pair_cells.max()) + 1
    size_keys, size_counts = np.unique(
        pair_groups * width + pair_cells, return_counts=True
    )
    sizes = pd.DataFrame(
        {
            "group": np.asarray(group_order, dtype=object)[size_keys // width],
            "clone_size": (size_keys % width).astype(np.int64),
            "n_clones": size_counts.astype(np.int64),
        }
    )
    totals = np.bincount(pair_codes, weights=pair_cells, minlength=n_clones)
    present = np.flatnonzero(totals > 0)
    # The largest clones, ties in the clones' rank order.
    chosen = present[np.lexsort((present, -totals[present]))[:top]]
    rows = np.full(n_clones, -1, dtype=np.int64)
    rows[chosen] = np.arange(len(chosen))
    taken = rows[pair_codes] >= 0
    matrix = np.zeros((len(chosen), len(group_order)), dtype=np.int64)
    matrix[rows[pair_codes[taken]], pair_groups[taken]] = pair_cells[taken]
    ids = loaded.clones["clone"].to_numpy(dtype=object)
    top_table = pd.DataFrame(
        matrix, columns=pd.Index([str(name) for name in group_order])
    )
    top_table.insert(0, "clone", ids[chosen])
    if group_labels is None:
        top_table = top_table.rename(columns={"all cells": "cells"})
    if figsize is None and target is None:
        figsize = (12.0, max(3.6, 0.28 * len(top_table) + 1.2))
    require_matplotlib()
    from matplotlib.ticker import LogFormatter

    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["sizes", "top"], figsize=figsize, n_columns=2
        )
        ax = axes["sizes"]
        by_group = dict(tuple(sizes.groupby("group", sort=False)))
        for group in group_order:
            group_sizes = by_group[group]
            ax.plot(
                group_sizes["clone_size"],
                group_sizes["n_clones"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=str(group),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(LogFormatter())
            axis.set_minor_formatter(LogFormatter())
        ax.set_xlabel("Cells of the clone")
        ax.set_ylabel("Clones")
        ax.set_title("Clone sizes")
        if show_legend and group_labels is not None:
            ax.legend(frameon=False, fontsize="small")
        ax = axes["top"]
        positions = np.arange(len(top_table))[::-1]
        left = np.zeros(len(top_table))
        for column in top_table.columns[1:]:
            widths = top_table[column].to_numpy(dtype=np.float64)
            ax.barh(positions, widths, left=left, label=str(column))
            left += widths
        ax.set_yticks(
            positions, [str(clone) for clone in top_table["clone"]], fontsize="small"
        )
        ax.set_xlabel("Cells")
        ax.set_title(f"Largest {len(top_table)} clones")
        if show_legend and group_labels is not None:
            ax.legend(frameon=False, fontsize="small", loc="lower right")
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"sizes": sizes, "top": top_table},
        legends=(LegendSpec(kind="categorical", label=group_name),)
        if group_labels is not None
        else (),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(keep.sum()),
            renderer="matplotlib",
            notes=("clones",),
            extras={
                "clones": clones.to_dict(),
                "grouping": group_name,
                "top": top,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _stacked_shares(
    ax: Any,
    table: pd.DataFrame,
    colors: dict[str, str],
    *,
    normalize: bool,
    xlabel: str,
    title: str,
    show_legend: bool,
    legend_title: str,
) -> None:
    """Draw one horizontal stacked bar per group, the first group on top."""
    totals = table.sum(axis=1).to_numpy(dtype=np.float64)
    positions = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for name in table.columns:
        widths = table[name].to_numpy(dtype=np.float64)
        if normalize:
            widths = np.divide(
                widths, totals, out=np.zeros_like(widths), where=totals > 0
            )
        ax.barh(positions, widths, left=left, color=colors[str(name)], label=str(name))
        left += widths
    ax.set_yticks(positions, [str(name) for name in table.index])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if show_legend and table.size:
        ax.legend(
            title=legend_title,
            frameon=False,
            fontsize="small",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )


def _long_table(wide: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Turn a table of groups by columns into rows of group, column, value, fraction."""
    totals = wide.sum(axis=1).to_numpy(dtype=np.float64)
    shares = np.divide(
        wide.to_numpy(dtype=np.float64),
        totals[:, None],
        out=np.full(wide.shape, np.nan),
        where=totals[:, None] > 0,
    )
    return pd.DataFrame(
        {
            "group": np.repeat(wide.index.to_numpy(dtype=object), wide.shape[1]),
            column: np.tile(wide.columns.to_numpy(dtype=object), wide.shape[0]),
            value: wide.to_numpy(dtype=np.int64).reshape(-1),
            "fraction": shares.reshape(-1),
        }
    )


@close_figures_on_error
def clonal_expansion(
    store: Any,
    expansion: ArtifactRef,
    *,
    group_by: str | None = None,
    groups: ArtifactRef | None = None,
    count: str = "cells",
    normalize: bool = True,
    splits: Sequence[int] = DEFAULT_RANK_SPLITS,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the clonal expansion and clonal space of groups of cells.

    ``expansion`` is a ``clonal_expansion`` artifact, such as one from
    ``ds.clones.expansion``. The first panel draws one stacked bar per group:
    the share of its cells (``count="cells"``) or of its clones
    (``count="clones"``) in each expansion category, as scirpy's
    ``summarize_clonal_expansion`` and scRepertoire's ``clonalHomeostasis``
    do. The second panel ranks each group's clones by their cells in the
    group and draws the share of its cells that the clones of each window of
    ranks hold, as scRepertoire's ``clonalProportion`` does: a sample whose
    ten largest clones hold most of its cells is dominated by a few clones.

    The groups are the expansion's own, such as its samples, unless
    ``group_by`` names a cell metadata column, such as a cell type, or
    ``groups`` a cell-label artifact, such as a clustering. Each cell keeps
    the category measured within its expansion group, so grouping the cells
    by cell type shows which cell types hold the expanded cells of their
    samples; the ranks are counted within the plotted groups. Cells without a
    category or a group are left out.

    Tables: ``categories`` (``group``, ``category``, then ``cells`` or
    ``clones``, and ``fraction``) and ``rank_windows`` (``group``,
    ``window``, ``cells`` and ``fraction``).
    """
    if count not in ("cells", "clones"):
        raise ValueError("count must be 'cells' or 'clones'")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    loaded = _clones_method(store, "load_expansion")(expansion, splits=splits)
    labels, names, group_name = cell_groups(
        store, expansion, group_by=group_by, groups=groups
    )
    category_names = list(loaded.labels)
    window_names = list(rank_window_labels(splits))
    if labels is None:
        # The expansion's own groups, which the loaded tables count; their
        # key columns lie between the group's name and the category.
        columns = list(loaded.categories.columns)
        group_name = ", ".join(columns[1 : columns.index("category")]) or None
        order = sort_categories(pd.unique(loaded.categories["group"].to_numpy(object)))
        wide = (
            loaded.categories.pivot(index="group", columns="category", values=count)
            .reindex(index=order, columns=category_names, fill_value=0)
            .astype(np.int64)
        )
        windows = (
            loaded.rank_windows.pivot(index="group", columns="window", values="cells")
            .reindex(index=order, columns=window_names, fill_value=0)
            .astype(np.int64)
        )
        n_cells = int(loaded.categories["cells"].sum())
    else:
        category = np.asarray(loaded.cells["category"].cat.codes, dtype=np.int64)
        clone = np.asarray(loaded.cells["clone"].cat.codes, dtype=np.int64)
        keep = (category >= 0) & (labels >= 0)
        held = sorted_unique(labels[keep])
        order = [names[code] for code in held.tolist()]
        group_codes = np.searchsorted(held, labels[keep])
        width = len(category_names)
        if count == "cells":
            keys = group_codes * width + category[keep]
        else:
            # Each clone counts once per group and category it holds there.
            n_clones = max(1, len(loaded.cells["clone"].cat.categories))
            triples = sorted_unique(
                (group_codes * n_clones + clone[keep]) * width + category[keep]
            )
            keys = triples // (n_clones * width) * width + triples % width
        wide = pd.DataFrame(
            np.bincount(keys, minlength=len(order) * width).reshape(len(order), width),
            index=pd.Index(order, name="group"),
            columns=pd.Index(category_names, name="category"),
        )
        windows = pd.DataFrame(
            rank_windows(clone[keep], group_codes, len(order), splits),
            index=pd.Index(order),
            columns=pd.Index(window_names),
        )
        n_cells = int(keep.sum())
    # Windows that hold no cell of any group are left out of the figure.
    shown_windows = windows.loc[:, windows.sum(axis=0) > 0]
    if figsize is None and target is None:
        figsize = (13.0, max(3.2, 0.32 * len(order) + 1.6))
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target,
            panel_keys=["categories", "rank_windows"],
            figsize=figsize,
            n_columns=2,
        )
        _stacked_shares(
            axes["categories"],
            wide,
            colormap_palette(list(loaded.labels), "viridis_r"),
            normalize=normalize,
            xlabel=f"Fraction of {count}" if normalize else count.capitalize(),
            title="Clonal expansion",
            show_legend=show_legend,
            legend_title=(
                "Clone size" if loaded.breakpoints is not None else "Clone frequency"
            ),
        )
        _stacked_shares(
            axes["rank_windows"],
            shown_windows,
            colormap_palette(window_names, "viridis"),
            normalize=normalize,
            xlabel="Fraction of cells" if normalize else "Cells",
            title="Clonal space by clone rank",
            show_legend=show_legend,
            legend_title="Clone rank",
        )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={
            "categories": _long_table(wide, "category", count),
            "rank_windows": _long_table(windows, "window", "cells"),
        },
        legends=(
            LegendSpec(kind="categorical", label="Expansion category"),
            LegendSpec(kind="categorical", label="Clone rank"),
        ),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=n_cells,
            renderer="matplotlib",
            notes=("clonal_expansion",),
            extras={
                "clonal_expansion": expansion.to_dict(),
                "grouping": group_name,
                "count": count,
                "normalize": normalize,
                "splits": list(loaded.splits),
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


@close_figures_on_error
def clonal_rarefaction(
    store: Any,
    clones: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    cell_selection: ArtifactRef | None = None,
    orders: Sequence[int] = ORDERS,
    base: str = "size",
    endpoint: int | None = None,
    knots: int = 40,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the rarefaction and extrapolation curves of groups of cells.

    ``clones`` is a ``clones`` artifact. Each panel draws one order of Hill
    numbers, the effective number of clones, against the cells sampled
    (``base="size"``) or the sample coverage (``base="coverage"``), as
    iNEXT's ``ggiNEXT`` draws them: one line per group, solid where the group
    is rarefied, dashed where it is extrapolated, and a point at its observed
    size. With ``base="size"`` a last panel draws each group's sample
    coverage against its size, which shows how complete each sample is.
    Curves that keep rising at the observed point have not reached their
    plateau: more cells would find more clones.

    The groups are those of ``ds.clones.diversity``: ``group_by`` cell
    metadata columns and a ``groups`` label artifact, among the cells of
    ``cell_selection``. ``endpoint`` and ``knots`` set the sizes of each
    curve, as ``ds.clones.rarefaction`` does.

    Tables: ``curves`` (``group``, the groups' key columns, ``order``,
    ``size``, ``method``, ``diversity``, ``coverage``).
    """
    if base not in ("size", "coverage"):
        raise ValueError("base must be 'size' or 'coverage'")
    result = _clones_method(store, "rarefaction")(
        clones,
        group_by=group_by,
        groups=groups,
        cell_selection=cell_selection,
        orders=orders,
        endpoint=endpoint,
        knots=knots,
    )
    curves = result.curves
    # The key columns lie between the group's name and the order.
    key_columns = list(curves.columns[1 : curves.columns.get_loc("order")])
    names = sort_categories(pd.unique(curves["group"]))
    colors = categorical_color_map(list(names))
    by_curve = dict(tuple(curves.groupby(["group", "order"], sort=False)))
    panels = [f"order_{order}" for order in result.orders]
    if base == "size":
        panels.append("coverage")
    if figsize is None and target is None:
        figsize = (4.2 * len(panels) + 1.5, 3.8)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=panels, figsize=figsize, n_columns=len(panels)
        )
        for panel in panels:
            ax = axes[panel]
            order = (
                int(panel.split("_")[1]) if panel != "coverage" else result.orders[0]
            )
            for name in names:
                rows = by_curve[(name, order)]
                x = rows["size" if base == "size" else "coverage"].to_numpy()
                y = rows["coverage" if panel == "coverage" else "diversity"].to_numpy()
                method = rows["method"].to_numpy()
                inner = method != "extrapolation"
                outer = method != "rarefaction"
                ax.plot(
                    x[inner], y[inner], color=colors[name], linewidth=1.4, label=name
                )
                ax.plot(x[outer], y[outer], color=colors[name], linewidth=1.4, ls="--")
                ax.plot(
                    x[method == "observed"],
                    y[method == "observed"],
                    "o",
                    color=colors[name],
                    markersize=4,
                )
            ax.set_xlabel("Cells" if base == "size" else "Sample coverage")
            if panel == "coverage":
                ax.set_ylabel("Sample coverage")
                ax.set_title("Sample completeness")
            else:
                ax.set_ylabel("Hill number")
                ax.set_title(_ORDER_TITLES[int(panel.split("_")[1])])
        if show_legend and len(names) > 1:
            axes[panels[-1]].legend(
                frameon=False, fontsize="small", loc="upper left", bbox_to_anchor=(1, 1)
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"curves": curves},
        legends=(LegendSpec(kind="categorical", label="Group"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(
                curves.loc[curves["method"] == "observed"]
                .drop_duplicates("group")["size"]
                .sum()
            ),
            renderer="matplotlib",
            notes=("clones",),
            extras={
                "clones": clones.to_dict(),
                "grouping": ", ".join(key_columns) or None,
                "orders": list(result.orders),
                "base": base,
                "endpoint": endpoint,
                "knots": knots,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
