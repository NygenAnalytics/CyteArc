"""Figures of clones: the cells that descend from one ancestor cell."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..clones.diversity import ORDERS
from ..clones.fate import CloneCoupling, dominant_fates
from ..clones.expansion import DEFAULT_RANK_SPLITS, rank_window_labels, rank_windows
from ..clones.groups import group_names
from ..clones.overlap import OVERLAP_INDICES
from ..storage.refs import ArtifactRef
from ..utils.arguments import integer_argument
from ..utils.arrays import sort_categories, sorted_unique, unique_rows
from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._figure import (
    LegendSpec,
    PlotResult,
    close_figures_on_error,
    normalize_axes_target,
)
from ._groups import cell_groups
from ._heatmap_utils import draw_dendrogram
from ._style import (
    LEGEND_SIDE_MAX_ENTRIES,
    MAX_FIGURE_WIDTH_INCHES,
    apply_figure_chrome,
    categorical_color_map,
    colormap_palette,
    foreground_color,
    legend_side_columns,
    legend_side_rows,
    theme_context,
)

__all__ = [
    "clonal_expansion",
    "clonal_overlap",
    "clonal_rarefaction",
    "clone_abundance",
    "clone_coupling",
    "clone_fates",
    "clone_tracking",
]

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


def _figure_labels(store: Any, clones: ArtifactRef, ids: np.ndarray) -> list[str]:
    """The labels of some clones in figures: their IDs, with junctions for clonotypes."""
    method = getattr(store, "_clone_figure_labels", None)
    if not callable(method):
        raise TypeError("store does not provide the figure labels of clones")
    return [str(label) for label in method(clones, ids.tolist())]


def _kept_apart_groups(
    cells: pd.DataFrame,
    kept_apart: Sequence[str],
    labels: np.ndarray | None,
    names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Each cell's group by its plotted group and the labels its clones keep apart.

    The groups are the combinations of a cell's plotted group, when there is
    a grouping, and its labels in each ``kept_apart`` column of ``cells``,
    categoricals such as its receptor type, in the natural order of the
    labels; each is named by its labels joined by commas.

    Returns:
        Each cell's group, a position in the names, -1 without one, and the
        names of the groups that some cell holds.
    """
    coded: list[tuple[np.ndarray, list[str]]] = []
    if labels is not None:
        coded.append((np.asarray(labels, dtype=np.int64), list(names)))
    for column in kept_apart:
        values = cells[column]
        coded.append(
            (
                np.asarray(values.cat.codes, dtype=np.int64),
                [str(label) for label in values.cat.categories],
            )
        )
    stacked = np.column_stack([codes for codes, _names in coded])
    keep = (stacked >= 0).all(axis=1)
    combinations, group_codes = unique_rows(stacked[keep])
    keys = [
        {
            str(position): column_names[code]
            for position, ((_codes, column_names), code) in enumerate(
                zip(coded, row, strict=True)
            )
        }
        for row in combinations.tolist()
    ]
    cell_group = np.full(len(stacked), -1, dtype=np.int64)
    cell_group[keep] = group_codes
    return cell_group, group_names(keys)


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

    ``clones`` is a clone artifact, such as one from
    ``ds.clones.import_clones`` or ``ds.repertoire.clonotypes``. The first
    panel counts clones by their number of cells on log scales: in most
    experiments many clones hold one or a few cells, and expanded clones form
    the tail. The second panel draws the ``top`` clones with the most cells,
    labelled with their IDs; a clonotype's label adds the junction amino
    acids of its representative cell's ``VDJ_1`` chain, or of its ``VJ_1``
    chain without a VDJ chain, such as ``clonotype_1 (CASSLGQAYEQYF)``.

    ``group_by`` names a cell metadata column, such as the sample or time
    point, and ``groups`` a cell-label artifact, such as a clustering; with
    either, the first panel draws one line per group, counting the cells of
    each clone within the group, and the second stacks each clone's cells by
    group. Cells without a group are left out. The plot raises ``ValueError``
    when no cell with a group has a clone. Clonotypes are counted within each
    receptor type, as every clone summary counts them, so their groups are
    the receptor types, or each group's receptor types, named like
    ``"s1, TCR"``.

    Tables: ``sizes`` (``group``, ``clone_size``, ``n_clones``) and ``top``
    (``clone``, ``label``, then the cells of each group, or ``cells``).
    """
    top = integer_argument(top, "top", minimum=1)
    loaded = _load_clones(store, clones)
    if group_by in loaded.kept_apart:
        raise ValueError(
            f"group_by names {group_by!r}, whose labels the clones keep apart "
            "already; leave it out, or copy the cell metadata column under "
            "another name"
        )
    group_labels, names, group_name = cell_groups(
        store, clones, group_by=group_by, groups=groups
    )
    codes = np.asarray(loaded.cells["clone"].cat.codes, dtype=np.int64)
    grouped = group_labels is not None or bool(loaded.kept_apart)
    if not grouped:
        cell_group = np.zeros(len(codes), dtype=np.int64)
        names = ["all cells"]
    else:
        cell_group, names = _kept_apart_groups(
            loaded.cells, loaded.kept_apart, group_labels, names
        )
        plotted = [] if group_name is None else [group_name]
        group_name = ", ".join([*plotted, *loaded.kept_apart])
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
    clashing = sorted({"clone", "label"} & {str(name) for name in group_order})
    if clashing:
        raise ValueError(
            f"A group is named {clashing[0]!r}, which the table of the largest "
            "clones uses for a column of its own; rename that label"
        )
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
    ids = loaded.clones["clone"].to_numpy(dtype=object)[chosen]
    top_table = pd.DataFrame(
        matrix, columns=pd.Index([str(name) for name in group_order])
    )
    top_table.insert(0, "clone", ids)
    top_table.insert(1, "label", _figure_labels(store, clones, ids))
    if not grouped:
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
        if show_legend and grouped:
            ax.legend(frameon=False, fontsize="small")
        ax = axes["top"]
        positions = np.arange(len(top_table))[::-1]
        left = np.zeros(len(top_table))
        for column in top_table.columns[2:]:
            widths = top_table[column].to_numpy(dtype=np.float64)
            ax.barh(positions, widths, left=left, label=str(column))
            left += widths
        ax.set_yticks(positions, top_table["label"].tolist(), fontsize="small")
        ax.set_xlabel("Cells")
        ax.set_title(f"Largest {len(top_table)} clones")
        if show_legend and grouped:
            ax.legend(frameon=False, fontsize="small", loc="lower right")
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"sizes": sizes, "top": top_table},
        legends=(LegendSpec(kind="categorical", label=group_name),) if grouped else (),
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
    samples; the ranks are counted within the plotted groups. Labels that the
    clones keep apart stay apart: the plotted groups of clonotypes are each
    group's receptor types, named like ``"blood, TCR"``. Cells without a
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
    # The expansion's key columns lie between the group's name and the
    # category; those of neither its group_by nor its label artifact hold the
    # labels that its clones keep apart, such as the receptor type.
    columns = list(loaded.categories.columns)
    key_columns = columns[1 : columns.index("category")]
    own = {*loaded.group_by, *([loaded.groups.kind] if loaded.groups else [])}
    kept_apart = [column for column in key_columns if column not in own]
    if group_by in kept_apart:
        raise ValueError(
            f"group_by names {group_by!r}, whose labels the clones keep apart "
            "already; leave it out, or copy the cell metadata column under "
            "another name"
        )
    labels, names, group_name = cell_groups(
        store, expansion, group_by=group_by, groups=groups
    )
    if labels is not None and kept_apart:
        labels, names = _kept_apart_groups(loaded.cells, kept_apart, labels, names)
        group_name = ", ".join([*([group_name] if group_name else []), *kept_apart])
    category_names = list(loaded.labels)
    window_names = list(rank_window_labels(splits))
    if labels is None:
        # The expansion's own groups, which the loaded tables count.
        group_name = ", ".join(key_columns) or None
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

    ``clones`` is a clone artifact. Each panel draws one order of Hill
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


# Overlap matrices of at most this many groups print their values, and the
# overlap and tracking plots of at most this many groups name them on their
# axes; their tables name every group.
_ANNOTATED_GROUPS = 12
_NAMED_GROUPS = 40
# Room in inches beside an overlap matrix for its colour bar.
_COLORBAR_INCHES = 1.2
_OVERLAP_TITLES = {
    "shared": "Shared clones",
    "jaccard": "Jaccard index",
    "sorensen": "Sørensen index",
    "overlap": "Overlap coefficient",
    "cosine": "Cosine of presence",
    "morisita_horn": "Morisita-Horn index",
    "abundance_cosine": "Cosine of clone sizes",
}


@close_figures_on_error
def clonal_overlap(
    store: Any,
    clones: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    cell_selection: ArtifactRef | None = None,
    index: str = "morisita_horn",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot an overlap index between every pair of groups as a heatmap.

    ``clones`` is a clone artifact, such as ``clones`` or ``clonotypes``.
    ``index`` is one of ``"shared"``, ``"jaccard"``, ``"sorensen"``,
    ``"overlap"``, ``"cosine"``, ``"morisita_horn"`` (the default) and
    ``"abundance_cosine"``, as ``ds.clones.overlap`` computes them for its
    groups: the ``group_by`` cell metadata columns and a ``groups`` label
    artifact, among the cells of ``cell_selection``. The diagonal is left blank
    so that the colours span the pairs, and so are pairs that were not
    compared; the ``matrix`` table holds each group's clones on its diagonal
    for ``"shared"`` and 1 otherwise. Matrices of at most 12 groups print their
    values, and of at most 40 name their groups on the axes. By default the
    figure grows with the groups up to the width of a page.

    Tables: ``matrix`` (groups by groups) and ``pairs``.
    """
    if index not in OVERLAP_INDICES:
        raise ValueError(f"index must be one of {', '.join(OVERLAP_INDICES)}")
    result = _clones_method(store, "overlap")(
        clones, group_by=group_by, groups=groups, cell_selection=cell_selection
    )
    names = list(result.group_sizes["group"])
    positions = pd.Index(names)
    first = positions.get_indexer(result.pairs["group_1"])
    second = positions.get_indexer(result.pairs["group_2"])
    pair_values = result.pairs[index].to_numpy(dtype=np.float64)
    values = np.full((len(names), len(names)), np.nan)
    values[first, second] = pair_values
    values[second, first] = pair_values
    values[np.diag_indices(len(names))] = (
        result.group_sizes["clones"].to_numpy(dtype=np.float64)
        if index == "shared"
        else 1.0
    )
    matrix = pd.DataFrame(values, index=pd.Index(names), columns=pd.Index(names))
    if figsize is None and target is None:
        side = min(
            max(4.0, 0.55 * len(names) + 2.5),
            MAX_FIGURE_WIDTH_INCHES - _COLORBAR_INCHES,
        )
        figsize = (side + _COLORBAR_INCHES, side)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["matrix"], figsize=figsize, n_columns=1
        )
        ax = axes["matrix"]
        drawn = values.copy()
        drawn[np.diag_indices(len(names))] = np.nan
        image = ax.imshow(np.ma.masked_invalid(drawn), cmap="viridis")
        if len(names) <= _NAMED_GROUPS:
            ax.set_xticks(np.arange(len(names)), names, rotation=90)
            ax.set_yticks(np.arange(len(names)), names)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(_OVERLAP_TITLES[index])
        if len(names) <= _ANNOTATED_GROUPS:
            for row in range(len(names)):
                for column in range(len(names)):
                    value = drawn[row, column]
                    if np.isnan(value):
                        continue
                    # Dark text on light cells, light text on dark ones.
                    red, green, blue, _alpha = image.cmap(image.norm(value))
                    light = 0.299 * red + 0.587 * green + 0.114 * blue > 0.5
                    ax.text(
                        column,
                        row,
                        f"{value:.0f}" if index == "shared" else f"{value:.3g}",
                        ha="center",
                        va="center",
                        color="black" if light else "white",
                        fontsize="x-small",
                    )
        if show_legend:
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"matrix": matrix, "pairs": result.pairs},
        legends=(LegendSpec(kind="continuous", label=index),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(result.group_sizes["cells"].sum()),
            renderer="matplotlib",
            notes=("clones",),
            extras={"clones": clones.to_dict(), "index": index, "groups": names},
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


@close_figures_on_error
def clone_tracking(
    store: Any,
    clones: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None = None,
    groups: ArtifactRef | None = None,
    cell_selection: ArtifactRef | None = None,
    top: int = 10,
    labels: Sequence[str] | None = None,
    order: Sequence[str] | None = None,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the share of cells of chosen clones across groups of cells.

    ``clones`` is a clone artifact, such as ``clones`` or ``clonotypes``. The
    plot follows the clones that ``ds.clones.track`` chooses: the ``top`` with
    the most cells in each group, or ``labels``. Each draws one line, labelled
    with its clone ID, across the groups, such as time points or cell types, of
    its share of each group's cells with a clone. Groups are placed by their
    labels in the ``group_by`` columns and the ``groups`` artifact, joined by
    commas; ``order`` lists those names from left to right, or values whose
    text is the name, such as time points as integers, and by default the names
    follow their natural order. Plots of at most 40 groups name them on the
    axis, and the legend lists at most 80 clones, the largest first. By default
    the figure grows with the groups up to the width of a page.

    Tables: ``cells`` (``clone``, ``group``, the key columns, ``cells`` and
    ``fraction``).
    """
    if order is not None:
        if isinstance(order, np.ndarray) and order.ndim == 1:
            order = order.tolist()
        if isinstance(order, str | bytes) or not isinstance(order, Sequence):
            raise TypeError("order must be a sequence of group names")
    tracked = _clones_method(store, "track")(
        clones,
        group_by=group_by,
        groups=groups,
        cell_selection=cell_selection,
        top=top,
        labels=labels,
    )
    cells = tracked.cells
    keys = [
        *tracked.group_by,
        *((tracked.groups.kind,) if tracked.groups is not None else ()),
    ]
    # Groups that differ only in labels that the clones keep apart share a
    # position, named by their labels in the grouping columns alone.
    positions_of = cells["group"]
    if keys:
        positions_of = cells[keys[0]].astype(str)
        for column in keys[1:]:
            positions_of = positions_of + ", " + cells[column].astype(str)
    placed = cells.assign(position=positions_of.to_numpy(dtype=object))
    present = sort_categories(pd.unique(placed["position"]))
    if order is None:
        positions = list(present)
    else:
        # Groups are placed by name, so an order of numbers names them by
        # their text.
        positions = [str(name) for name in order]
        if sorted(positions) != sorted(map(str, present)):
            raise ValueError(f"order must list the groups {present}")
    by_clone = dict(tuple(placed.groupby("clone", sort=False)))
    if figsize is None and target is None:
        figsize = (
            min(max(6.0, 0.7 * len(positions) + 4.0), MAX_FIGURE_WIDTH_INCHES),
            4.2,
        )
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["tracking"], figsize=figsize, n_columns=1
        )
        ax = axes["tracking"]
        colors = categorical_color_map(list(tracked.tracked))
        x = np.arange(len(positions))
        for name in tracked.tracked:
            rows = by_clone.get(name, placed.iloc[:0])
            y = (
                rows.set_index("position")["fraction"]
                .reindex(positions)
                .to_numpy(dtype=np.float64)
            )
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=colors[name],
                label=name,
            )
        if len(positions) <= _NAMED_GROUPS:
            ax.set_xticks(x, positions, rotation=45, ha="right")
        else:
            ax.set_xticks([])
        ax.set_ylabel("Fraction of the group's cells")
        ax.set_title("Clones across groups")
        if show_legend:
            # The largest clones, in columns that fit the figure's height.
            lines = ax.get_lines()
            shown = min(len(lines), LEGEND_SIDE_MAX_ENTRIES)
            rows = legend_side_rows(float(figure.get_size_inches()[1]), title=True)
            ax.legend(
                handles=lines[:shown],
                frameon=False,
                fontsize="small",
                title=(
                    "Clone"
                    if shown == len(lines)
                    else f"Clone ({shown} of {len(lines)})"
                ),
                title_fontsize="small",
                loc="upper left",
                bbox_to_anchor=(1, 1),
                ncols=legend_side_columns(shown, rows=rows),
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"cells": cells},
        legends=(LegendSpec(kind="categorical", label="Clone"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(cells["cells"].sum()),
            renderer="matplotlib",
            notes=("clones",),
            extras={
                "clones": clones.to_dict(),
                "tracked": list(tracked.tracked),
                "order": positions,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


_COUPLING_TITLES = {
    "share_cosine": "Cosine of clone shares",
    "abundance_cosine": "Cosine of clone sizes",
    "normalized_covariance": "Normalized covariance",
    "jaccard": "Jaccard index",
}
_COUPLING_VALUES = ("z_score", "coupling")


@close_figures_on_error
def clone_coupling(
    store: Any,
    coupling: CloneCoupling,
    *,
    value: str = "z_score",
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the fate coupling of every pair of groups as a heatmap with its dendrogram.

    ``coupling`` is the result of ``ds.clones.coupling``. ``value`` is
    ``"z_score"`` (the default), the coupling's distance from its null in
    null standard deviations, on a colour scale centred on 0, or
    ``"coupling"``, the method's value. The groups follow the result's
    average-linkage order, whose dendrogram is drawn above the matrix. The
    diagonal is left blank, and so are pairs that were not compared and
    z-scores that are not finite; a coupling measured without permutations
    has no z-scores and is refused. Matrices of at most 12 groups print their
    values, and of at most 40 name their groups on the axes. By default the
    figure grows with the groups up to the width of a page.

    Tables: ``matrix`` (groups by groups, in the drawn order) and ``pairs``.
    """
    if not isinstance(coupling, CloneCoupling):
        raise TypeError("coupling must be the CloneCoupling of ds.clones.coupling")
    if value not in _COUPLING_VALUES:
        raise ValueError(f"value must be one of {', '.join(_COUPLING_VALUES)}")
    if value == "z_score" and not coupling.permutations:
        raise ValueError(
            "The coupling was measured without permutations, so it has no "
            "z-scores; plot value='coupling', or measure it with permutations"
        )
    names = list(coupling.order)
    positions = pd.Index(names)
    first = positions.get_indexer(coupling.pairs["group_1"])
    second = positions.get_indexer(coupling.pairs["group_2"])
    pair_values = coupling.pairs[value].to_numpy(dtype=np.float64)
    values = np.full((len(names), len(names)), np.nan)
    values[first, second] = pair_values
    values[second, first] = pair_values
    matrix = pd.DataFrame(values, index=pd.Index(names), columns=pd.Index(names))
    drawn = np.where(np.isfinite(values), values, np.nan)
    finite = drawn[np.isfinite(drawn)]
    centred = value == "z_score" or bool((finite < 0).any())
    if figsize is None and target is None:
        side = min(
            max(4.0, 0.55 * len(names) + 2.5),
            MAX_FIGURE_WIDTH_INCHES - _COLORBAR_INCHES,
        )
        figsize = (side + _COLORBAR_INCHES, side + 0.8)
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["matrix"], figsize=figsize, n_columns=1
        )
        ax = axes["matrix"]
        limit = float(np.abs(finite).max()) if finite.size else 1.0
        image = ax.imshow(
            np.ma.masked_invalid(drawn),
            cmap="RdBu_r" if centred else "viridis",
            vmin=-limit if centred else None,
            vmax=limit if centred else None,
        )
        if len(names) <= _NAMED_GROUPS:
            ax.set_xticks(np.arange(len(names)), names, rotation=90)
            ax.set_yticks(np.arange(len(names)), names)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        title = _COUPLING_TITLES.get(coupling.method, str(coupling.method))
        ax.set_title(f"{title}, z-score" if value == "z_score" else title, y=1.18)
        if len(names) <= _ANNOTATED_GROUPS:
            for row in range(len(names)):
                for column in range(len(names)):
                    shown = drawn[row, column]
                    if np.isnan(shown):
                        continue
                    # Dark text on light cells, light text on dark ones.
                    red, green, blue, _alpha = image.cmap(image.norm(shown))
                    light = 0.299 * red + 0.587 * green + 0.114 * blue > 0.5
                    ax.text(
                        column,
                        row,
                        f"{shown:.2g}" if value == "z_score" else f"{shown:.3g}",
                        ha="center",
                        va="center",
                        color="black" if light else "white",
                        fontsize="x-small",
                    )
        axes["dendrogram"] = draw_dendrogram(
            ax,
            _ordered_linkage(coupling),
            side="top",
            color=foreground_color(theme),
        )
        if show_legend:
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"matrix": matrix, "pairs": coupling.pairs},
        legends=(LegendSpec(kind="continuous", label=value),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(coupling.group_sizes["cells"].sum()),
            renderer="matplotlib",
            notes=("clones",),
            extras={
                "clones": coupling.clones.to_dict(),
                "method": coupling.method,
                "value": value,
                "groups": names,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result


def _ordered_linkage(coupling: Any) -> np.ndarray:
    """The coupling's tree, checked to place its leaves in the drawn order."""
    from scipy.cluster.hierarchy import leaves_list

    tree = np.asarray(coupling.linkage, dtype=np.float64)
    names = coupling.group_sizes["group"].to_numpy(dtype=object)
    if [str(name) for name in names[leaves_list(tree)].tolist()] != list(
        coupling.order
    ):
        raise ValueError("The coupling's order does not follow its linkage")
    return tree


@close_figures_on_error
def clone_fates(
    store: Any,
    bias: ArtifactRef,
    *,
    top: int = 40,
    normalize: bool = False,
    target: Any | None = None,
    figsize: tuple[float, float] | None = None,
    theme: str = "notebook",
    show_legend: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot the fates of the clones of early cells, each clone's late cells by fate.

    ``bias`` is a ``clone_fate_bias`` artifact, such as one from
    ``ds.clones.fate_bias``. The plot draws the ``top`` clones with the most
    late cells, ties to the larger clone overall, grouped by their dominant
    fate, the fate that holds the most of their late cells, in the order of
    the fates, and within a fate from the most late cells to the fewest;
    clones whose late cells tie between fates come last. The lower panel
    stacks each clone's late cells by fate, or their shares with
    ``normalize=True``, and the upper panel draws its early cells, whose
    fractions follow the clone's late cells. Plots of at most 40 clones name
    them on the axis. By default the figure grows with the clones up to the
    width of a page.

    Tables: ``clones`` (``clone``, ``fate``, the dominant fate, missing for a
    tie, ``early_cells``, ``late_cells``, then the late cells of each fate).
    """
    top = integer_argument(top, "top", minimum=1)
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    loaded = _clones_method(store, "load_fate_bias")(bias)
    fate_names = [str(name) for name in loaded.fates["fate"].tolist()]
    rows = loaded.clone_fates
    clone_codes = np.asarray(rows["clone"].cat.codes, dtype=np.int64)
    fate_codes = np.asarray(rows["fate"].cat.codes, dtype=np.int64)
    # One row per clone, in the order of the clone codes, with its late cells
    # in each fate.
    codes, at = np.unique(clone_codes, return_inverse=True)
    cells = np.zeros((len(codes), len(fate_names)), dtype=np.int64)
    np.add.at(cells, (at.reshape(-1), fate_codes), rows["cells"].to_numpy(np.int64))
    early_cells = np.zeros(len(codes), dtype=np.int64)
    early_cells[at.reshape(-1)] = rows["early_cells"].to_numpy(np.int64)
    late_cells = cells.sum(axis=1)
    dominant = dominant_fates(cells)
    # The clones with the most late cells, ties to the lower code, the larger
    # clone overall, drawn by dominant fate and then by late cells.
    chosen = np.lexsort((codes, -late_cells))[:top]
    rank = np.where(dominant[chosen] >= 0, dominant[chosen], len(fate_names))
    chosen = chosen[np.lexsort((codes[chosen], -late_cells[chosen], rank))]
    labels = rows["clone"].cat.categories.to_numpy(dtype=object)
    shown_labels = [str(label) for label in labels[codes[chosen]].tolist()]
    table = pd.DataFrame(cells[chosen], columns=pd.Index(fate_names, dtype=object))
    table.insert(0, "clone", shown_labels)
    table.insert(
        1,
        "fate",
        pd.Categorical.from_codes(dominant[chosen], categories=pd.Index(fate_names)),
    )
    table.insert(2, "early_cells", early_cells[chosen])
    table.insert(3, "late_cells", late_cells[chosen])
    if figsize is None and target is None:
        figsize = (
            min(max(6.0, 0.22 * len(chosen) + 3.5), MAX_FIGURE_WIDTH_INCHES),
            5.0,
        )
    require_matplotlib()
    with theme_context(theme):
        figure, axes, owns = normalize_axes_target(
            target, panel_keys=["early", "fates"], figsize=figsize, n_columns=1
        )
        if owns:
            # The early cells take a quarter of the height.
            axes["early"].get_gridspec().set_height_ratios([1, 3])
        colors = categorical_color_map(fate_names)
        x = np.arange(len(chosen))
        ax = axes["early"]
        ax.bar(x, early_cells[chosen], color=foreground_color(theme), width=0.8)
        ax.set_ylabel("Early cells")
        ax.set_xticks([])
        ax.set_title("Fates of the clones of early cells")
        ax = axes["fates"]
        drawn = cells[chosen].astype(np.float64)
        if normalize:
            totals = drawn.sum(axis=1, keepdims=True)
            drawn = np.divide(drawn, totals, out=np.zeros_like(drawn), where=totals > 0)
        bottom = np.zeros(len(chosen))
        for position, name in enumerate(fate_names):
            ax.bar(
                x,
                drawn[:, position],
                bottom=bottom,
                color=colors[name],
                label=name,
                width=0.8,
            )
            bottom += drawn[:, position]
        if len(chosen) <= _NAMED_GROUPS:
            ax.set_xticks(x, shown_labels, rotation=90, fontsize="small")
        else:
            ax.set_xticks([])
        ax.set_xlabel("Clone")
        ax.set_ylabel("Share of late cells" if normalize else "Late cells")
        axes["early"].set_xlim(ax.get_xlim())
        if show_legend:
            ax.legend(
                title="Fate",
                frameon=False,
                fontsize="small",
                title_fontsize="small",
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )
        apply_figure_chrome(figure, theme)
    plot_result = PlotResult(
        figure=figure,
        axes=axes,
        tables={"clones": table},
        legends=(LegendSpec(kind="categorical", label="Fate"),),
        scales=(),
        provenance=PlotProvenance(
            assay=None,
            cell_key=None,
            n_cells=int(late_cells[chosen].sum()),
            renderer="matplotlib",
            notes=("clones",),
            extras={
                "bias": bias.to_dict(),
                "top": top,
                "normalize": normalize,
                "clones": shown_labels,
            },
        ),
        owns_figure=owns,
        theme=theme,
    )
    if show:
        plot_result.show()
    return plot_result
